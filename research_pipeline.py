"""Reproducible analyses for the Student Depression outcome-proxy audit.

The module is intentionally importable from both the executable notebook and
the command line.  It writes only derived artefacts under ``artifacts/``.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
TARGET = "Depression"
GROUP_COLUMN = "Home_division"
FEATURE_SETS = {
    "full": None,
    "proxy_free": ["Health_condition"],
}


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "figures": root / "figures",
        "tables": root / "tables",
        "results": root / "results",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    raw = pd.read_csv(csv_path)
    required = {
        "Gender",
        "Age",
        "Family_background",
        "Relationship_status",
        "Marital_status",
        "Living_style",
        "Current_semester",
        "CGPA",
        "friend_circle",
        "Health_condition",
        "Cost_without_study",
        "Home_division",
        TARGET,
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not set(raw[TARGET].dropna().unique()).issubset({"Yes", "No"}):
        raise ValueError("Depression must contain only Yes/No labels")
    X = raw.drop(columns=TARGET).copy()
    y = (raw[TARGET] == "Yes").astype(int).to_numpy()
    return raw, X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        verbose_feature_names_out=False,
    )


def model_spec(name: str, X: pd.DataFrame) -> tuple[Pipeline, dict[str, list]]:
    preprocessor = build_preprocessor(X)
    if name == "Dummy":
        estimator = DummyClassifier(strategy="prior")
        grid: dict[str, list] = {}
    elif name == "Logistic":
        estimator = LogisticRegression(max_iter=3000, random_state=SEED)
        grid = {"model__C": [0.1, 1.0, 10.0]}
    elif name == "RandomForest":
        estimator = RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1
        )
        grid = {
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 5],
        }
    elif name == "ExtraTrees":
        estimator = ExtraTreesClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1
        )
        grid = {
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 5],
        }
    elif name == "HistGradientBoosting":
        estimator = HistGradientBoostingClassifier(max_iter=250, random_state=SEED)
        grid = {
            "model__learning_rate": [0.05, 0.1],
            "model__max_leaf_nodes": [15, 31],
            "model__l2_regularization": [0.0, 1.0],
        }
    else:
        raise KeyError(name)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)]), grid


def threshold_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "AUROC": roc_auc_score(y, probability),
        "AUPRC": average_precision_score(y, probability),
        "Brier": brier_score_loss(y, probability),
        "Balanced_accuracy": balanced_accuracy_score(y, prediction),
        "F1": f1_score(y, prediction, zero_division=0),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "FNR": 1.0 - sensitivity if np.isfinite(sensitivity) else np.nan,
        "FPR": 1.0 - specificity if np.isfinite(specificity) else np.nan,
    }


def _best_estimator(
    pipeline: Pipeline,
    grid: dict[str, list],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    inner_cv: StratifiedKFold,
) -> tuple[Pipeline, dict]:
    if not grid:
        fitted = clone(pipeline).fit(X_train, y_train)
        return fitted, {}
    search = GridSearchCV(
        pipeline,
        grid,
        scoring="average_precision",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def nested_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    outer_splits: int = 5,
) -> tuple[np.ndarray, pd.DataFrame, list[dict]]:
    pipeline, grid = model_spec(model_name, X)
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=SEED)
    probability = np.full(len(y), np.nan, dtype=float)
    rows: list[dict] = []
    best_params: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fold)
        fitted, params = _best_estimator(
            pipeline, grid, X.iloc[train_idx], y[train_idx], inner
        )
        fold_probability = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        probability[test_idx] = fold_probability
        rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                **threshold_metrics(y[test_idx], fold_probability),
            }
        )
        best_params.append(params)
    if np.isnan(probability).any():
        raise RuntimeError("OOF prediction vector is incomplete")
    return probability, pd.DataFrame(rows), best_params


def bootstrap_interval(
    y: np.ndarray,
    p: np.ndarray,
    metric_name: str,
    repetitions: int = 1000,
    seed: int = SEED,
) -> tuple[float, float]:
    metric = {
        "AUROC": roc_auc_score,
        "AUPRC": average_precision_score,
        "Brier": brier_score_loss,
    }[metric_name]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y)
    for _ in range(repetitions):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        values.append(metric(y[idx], p[idx]))
    return tuple(np.quantile(values, [0.025, 0.975]))


def audit_tables(raw: pd.DataFrame, paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    audit = pd.DataFrame(
        {
            "item": [
                "Rows",
                "Predictors",
                "Depression positive",
                "Depression prevalence",
                "Missing cells",
                "Duplicate rows",
            ],
            "value": [
                len(raw),
                raw.shape[1] - 1,
                int((raw[TARGET] == "Yes").sum()),
                float((raw[TARGET] == "Yes").mean()),
                int(raw.isna().sum().sum()),
                int(raw.duplicated().sum()),
            ],
        }
    )
    health = (
        raw.assign(target=(raw[TARGET] == "Yes").astype(int))
        .groupby("Health_condition", observed=True)["target"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"sum": "depressed_n", "mean": "depression_rate"})
        .sort_values("depression_rate", ascending=False)
        .reset_index()
    )
    group_rows: list[pd.DataFrame] = []
    for column in raw.columns.drop(TARGET):
        if raw[column].nunique(dropna=False) <= 20:
            table = (
                raw.assign(target=(raw[TARGET] == "Yes").astype(int))
                .groupby(column, observed=True)["target"]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={column: "level", "mean": "depression_rate"})
            )
            table.insert(0, "feature", column)
            group_rows.append(table)
    univariate = pd.concat(group_rows, ignore_index=True)
    audit.to_csv(paths["tables"] / "data_audit.csv", index=False)
    health.to_csv(paths["tables"] / "health_condition_target_overlap.csv", index=False)
    univariate.to_csv(paths["tables"] / "univariate_target_rates.csv", index=False)
    return {"audit": audit, "health": health, "univariate": univariate}


def run_model_benchmark(
    raw: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], list[dict]]]:
    models = ["Dummy", "Logistic", "RandomForest", "ExtraTrees", "HistGradientBoosting"]
    summary_rows: list[dict] = []
    fold_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    params_by_run: dict[tuple[str, str], list[dict]] = {}
    for feature_set, dropped in FEATURE_SETS.items():
        X_current = X.drop(columns=dropped or [])
        for model in models:
            print(f"Nested CV: {feature_set} / {model}", flush=True)
            probability, folds, params = nested_oof(X_current, y, model)
            params_by_run[(feature_set, model)] = params
            aggregate = threshold_metrics(y, probability)
            row = {"feature_set": feature_set, "model": model, **aggregate}
            for metric in ["AUROC", "AUPRC", "Brier"]:
                low, high = bootstrap_interval(y, probability, metric)
                row[f"{metric}_low"] = low
                row[f"{metric}_high"] = high
            summary_rows.append(row)
            fold_frame = folds.copy()
            fold_frame.insert(0, "model", model)
            fold_frame.insert(0, "feature_set", feature_set)
            fold_rows.append(fold_frame)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "row_id": np.arange(len(y)),
                        "feature_set": feature_set,
                        "model": model,
                        "y_true": y,
                        "probability": probability,
                        "Gender": raw["Gender"].to_numpy(),
                        "Family_background": raw["Family_background"].to_numpy(),
                        "Home_division": raw["Home_division"].to_numpy(),
                    }
                )
            )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["feature_set", "AUPRC"], ascending=[True, False]
    )
    folds = pd.concat(fold_rows, ignore_index=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    summary.to_csv(paths["tables"] / "nested_cv_performance.csv", index=False)
    folds.to_csv(paths["results"] / "nested_cv_fold_metrics.csv", index=False)
    predictions.to_csv(paths["results"] / "oof_predictions.csv", index=False)
    with (paths["results"] / "best_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {f"{key[0]}::{key[1]}": value for key, value in params_by_run.items()},
            handle,
            indent=2,
        )
    return summary, predictions, params_by_run


def choose_models(summary: pd.DataFrame) -> dict[str, str]:
    non_dummy = summary[summary["model"] != "Dummy"]
    return (
        non_dummy.sort_values("AUPRC", ascending=False)
        .groupby("feature_set", sort=False)
        .first()["model"]
        .to_dict()
    )


def geographic_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    feature_set: str,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    dropped = FEATURE_SETS[feature_set] or []
    X_current = X.drop(columns=dropped)
    pipeline, grid = model_spec(model_name, X_current)
    logo = LeaveOneGroupOut()
    probability = np.full(len(y), np.nan)
    rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X_current, y, groups=groups), start=1
    ):
        held_out = str(groups.iloc[test_idx[0]])
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fold)
        fitted, _ = _best_estimator(
            pipeline, grid, X_current.iloc[train_idx], y[train_idx], inner
        )
        fold_probability = fitted.predict_proba(X_current.iloc[test_idx])[:, 1]
        probability[test_idx] = fold_probability
        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "held_out_division": held_out,
                "n": len(test_idx),
                "prevalence": y[test_idx].mean(),
                **threshold_metrics(y[test_idx], fold_probability),
            }
        )
    overall = pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "model": model_name,
                "held_out_division": "Overall",
                "n": len(y),
                "prevalence": y.mean(),
                **threshold_metrics(y, probability),
            }
        ]
    )
    return overall, pd.DataFrame(rows), probability


def subgroup_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    groups: pd.Series,
    group_name: str,
    feature_set: str,
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for level in sorted(groups.astype(str).unique()):
        mask = groups.astype(str).eq(level).to_numpy()
        if len(np.unique(y[mask])) < 2:
            continue
        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "group": group_name,
                "level": level,
                "n": int(mask.sum()),
                "prevalence": float(y[mask].mean()),
                **threshold_metrics(y[mask], probability[mask]),
            }
        )
    return pd.DataFrame(rows)


def fairness_audit(
    raw: pd.DataFrame,
    y: np.ndarray,
    predictions: pd.DataFrame,
    selected: dict[str, str],
    paths: dict[str, Path],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for feature_set, model in selected.items():
        subset = predictions[
            (predictions["feature_set"] == feature_set)
            & (predictions["model"] == model)
        ].sort_values("row_id")
        p = subset["probability"].to_numpy()
        for group_name in ["Gender", "Family_background", "Home_division"]:
            frames.append(
                subgroup_metrics(
                    y, p, raw[group_name], group_name, feature_set, model
                )
            )
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(paths["tables"] / "subgroup_fairness_metrics.csv", index=False)
    gaps = (
        result.groupby(["feature_set", "model", "group"])[["FNR", "FPR", "Brier"]]
        .agg(lambda s: s.max() - s.min())
        .reset_index()
        .rename(columns={"FNR": "FNR_gap", "FPR": "FPR_gap", "Brier": "Brier_gap"})
    )
    gaps.to_csv(paths["tables"] / "subgroup_metric_gaps.csv", index=False)
    return result


def conformal_experiment(
    X: pd.DataFrame,
    y: np.ndarray,
    raw: pd.DataFrame,
    model_name: str,
    paths: dict[str, Path],
    alpha: float = 0.10,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_current = X.drop(columns=FEATURE_SETS["proxy_free"] or [])
    pipeline, grid = model_spec(model_name, X_current)
    run_rows: list[dict] = []
    subgroup_rows: list[dict] = []
    for seed in seeds:
        train_idx, temp_idx = train_test_split(
            np.arange(len(y)), test_size=0.40, stratify=y, random_state=seed
        )
        cal_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            stratify=y[temp_idx],
            random_state=seed + 1000,
        )
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        fitted, _ = _best_estimator(
            clone(pipeline), grid, X_current.iloc[train_idx], y[train_idx], inner
        )
        p_cal = fitted.predict_proba(X_current.iloc[cal_idx])
        p_test = fitted.predict_proba(X_current.iloc[test_idx])
        scores = 1.0 - p_cal[np.arange(len(cal_idx)), y[cal_idx]]
        quantile_level = min(1.0, math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
        qhat = float(np.quantile(scores, quantile_level, method="higher"))
        prediction_sets = (1.0 - p_test) <= qhat
        covered = prediction_sets[np.arange(len(test_idx)), y[test_idx]]
        set_size = prediction_sets.sum(axis=1)
        singleton = set_size == 1
        singleton_prediction = prediction_sets.argmax(axis=1)
        run_rows.append(
            {
                "seed": seed,
                "alpha": alpha,
                "qhat": qhat,
                "n_test": len(test_idx),
                "coverage": covered.mean(),
                "mean_set_size": set_size.mean(),
                "singleton_rate": singleton.mean(),
                "empty_rate": (set_size == 0).mean(),
                "singleton_accuracy": (
                    (singleton_prediction[singleton] == y[test_idx][singleton]).mean()
                    if singleton.any()
                    else np.nan
                ),
                "AUROC": roc_auc_score(y[test_idx], p_test[:, 1]),
                "AUPRC": average_precision_score(y[test_idx], p_test[:, 1]),
            }
        )
        for group_name in ["Gender", "Home_division"]:
            values = raw.iloc[test_idx][group_name].astype(str).to_numpy()
            for level in sorted(np.unique(values)):
                mask = values == level
                subgroup_rows.append(
                    {
                        "seed": seed,
                        "group": group_name,
                        "level": level,
                        "n": int(mask.sum()),
                        "coverage": float(covered[mask].mean()),
                        "mean_set_size": float(set_size[mask].mean()),
                    }
                )
    runs = pd.DataFrame(run_rows)
    subgroups = pd.DataFrame(subgroup_rows)
    runs.to_csv(paths["tables"] / "conformal_runs.csv", index=False)
    subgroups.to_csv(paths["tables"] / "conformal_subgroup_coverage.csv", index=False)
    return runs, subgroups


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def create_figures(
    raw: pd.DataFrame,
    y: np.ndarray,
    audit: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    selected: dict[str, str],
    geo_detail: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = raw[TARGET].value_counts().reindex(["No", "Yes"])
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], color="#4C78A8")
    axes[0].set(title="Outcome distribution", xlabel="Depression indicator", ylabel="Students")
    health = audit["health"].sort_values("depression_rate")
    sns.barplot(
        data=health,
        x="Health_condition",
        y="depression_rate",
        ax=axes[1],
        color="#E45756",
    )
    axes[1].set(title="Outcome rate by health condition", xlabel="Health condition", ylabel="Depression rate", ylim=(0, 1.05))
    axes[1].tick_params(axis="x", rotation=25)
    _savefig(paths["figures"] / "figure_1_data_and_proxy_audit.png")

    plot_data = summary[summary["model"] != "Dummy"].copy()
    plt.figure(figsize=(9, 4.8))
    ax = sns.barplot(data=plot_data, x="model", y="AUPRC", hue="feature_set")
    ax.axhline(y.mean(), color="black", linestyle="--", linewidth=1, label="Prevalence")
    ax.set(title="Nested cross-validated precision-recall performance", xlabel="Model", ylabel="AUPRC", ylim=(0, 1.03))
    ax.tick_params(axis="x", rotation=20)
    _savefig(paths["figures"] / "figure_2_nested_cv_auprc.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for feature_set, model in selected.items():
        subset = predictions[(predictions.feature_set == feature_set) & (predictions.model == model)].sort_values("row_id")
        p = subset.probability.to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{feature_set}: {model} ({roc_auc_score(y,p):.3f})")
        axes[1].plot(recall, precision, label=f"{feature_set}: {model} ({average_precision_score(y,p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(title="ROC curves", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].axhline(y.mean(), color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
    _savefig(paths["figures"] / "figure_3_discrimination_curves.png")

    plt.figure(figsize=(5.5, 5))
    for feature_set, model in selected.items():
        subset = predictions[(predictions.feature_set == feature_set) & (predictions.model == model)].sort_values("row_id")
        fraction, mean_prediction = calibration_curve(y, subset.probability, n_bins=10, strategy="quantile")
        plt.plot(mean_prediction, fraction, marker="o", label=f"{feature_set}: {model}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed outcome rate")
    plt.title("OOF calibration")
    plt.legend(fontsize=8)
    _savefig(paths["figures"] / "figure_4_calibration.png")

    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=geo_detail, x="held_out_division", y="AUPRC", hue="feature_set")
    plt.xlabel("Held-out division")
    plt.ylabel("AUPRC")
    plt.title("Leave-one-division-out transportability")
    plt.xticks(rotation=30, ha="right")
    _savefig(paths["figures"] / "figure_5_geographic_transportability.png")

    feature_set = "proxy_free"
    model = selected[feature_set]
    subset = predictions[(predictions.feature_set == feature_set) & (predictions.model == model)].sort_values("row_id")
    p = subset.probability.to_numpy()
    confidence = np.abs(p - 0.5) * 2
    rows = []
    for threshold in np.linspace(0.0, 0.9, 19):
        retained = confidence >= threshold
        if retained.sum() < 50 or np.unique(y[retained]).size < 2:
            continue
        metric = threshold_metrics(y[retained], p[retained])
        rows.append({"confidence_threshold": threshold, "retained_fraction": retained.mean(), **metric})
    selective = pd.DataFrame(rows)
    selective.to_csv(paths["tables"] / "selective_prediction.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(selective.confidence_threshold, selective.Balanced_accuracy, marker="o", label="Balanced accuracy")
    ax1.plot(selective.confidence_threshold, selective.Sensitivity, marker="s", label="Sensitivity")
    ax1.set(xlabel="Minimum confidence", ylabel="Performance", ylim=(0, 1.02))
    ax2 = ax1.twinx()
    ax2.plot(selective.confidence_threshold, selective.retained_fraction, color="black", linestyle="--", label="Retained fraction")
    ax2.set(ylabel="Retained fraction", ylim=(0, 1.02))
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")
    ax1.set_title("Proxy-free selective prediction trade-off")
    _savefig(paths["figures"] / "figure_6_selective_prediction.png")


def run_all(
    csv_path: str | Path = "Student_Depression_Dataset.csv",
    artifact_root: str | Path = "artifacts",
) -> dict[str, object]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    csv_path = Path(csv_path)
    paths = ensure_dirs(Path(artifact_root))
    raw, X, y = load_dataset(csv_path)
    audit = audit_tables(raw, paths)
    summary, predictions, params = run_model_benchmark(raw, X, y, paths)
    selected = choose_models(summary)

    geo_overall_frames: list[pd.DataFrame] = []
    geo_detail_frames: list[pd.DataFrame] = []
    geo_probabilities: dict[str, np.ndarray] = {}
    for feature_set, model in selected.items():
        print(f"Geographic validation: {feature_set} / {model}", flush=True)
        overall, detail, probability = geographic_validation(
            X, y, raw[GROUP_COLUMN], feature_set, model
        )
        geo_overall_frames.append(overall)
        geo_detail_frames.append(detail)
        geo_probabilities[feature_set] = probability
    geo_overall = pd.concat(geo_overall_frames, ignore_index=True)
    geo_detail = pd.concat(geo_detail_frames, ignore_index=True)
    geo_overall.to_csv(paths["tables"] / "geographic_validation_overall.csv", index=False)
    geo_detail.to_csv(paths["tables"] / "geographic_validation_by_division.csv", index=False)

    fairness = fairness_audit(raw, y, predictions, selected, paths)
    conformal_runs, conformal_subgroups = conformal_experiment(
        X, y, raw, selected["proxy_free"], paths
    )
    create_figures(raw, y, audit, summary, predictions, selected, geo_detail, paths)

    manifest = {
        "dataset": str(csv_path.resolve()),
        "rows": int(len(raw)),
        "positive_n": int(y.sum()),
        "prevalence": float(y.mean()),
        "selected_models": selected,
        "nested_cv_results": summary.to_dict(orient="records"),
        "geographic_overall": geo_overall.to_dict(orient="records"),
        "conformal_mean": conformal_runs.mean(numeric_only=True).to_dict(),
    }
    with (paths["results"] / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return {
        "raw": raw,
        "X": X,
        "y": y,
        "audit": audit,
        "summary": summary,
        "predictions": predictions,
        "selected": selected,
        "geo_overall": geo_overall,
        "geo_detail": geo_detail,
        "fairness": fairness,
        "conformal_runs": conformal_runs,
        "conformal_subgroups": conformal_subgroups,
        "params": params,
        "manifest": manifest,
    }


if __name__ == "__main__":
    output = run_all()
    print("\nSelected models:", output["selected"])
    print("\nNested CV summary:")
    print(output["summary"].to_string(index=False))
