"""Build the executable research notebook from version-controlled cell sources."""

from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        r"""
# High Accuracy, Low Assurance
## Outcome-proxy sensitivity, geographic transportability, fairness, and uncertainty in student-depression prediction

This notebook is the executable companion to the manuscript. It evaluates a public survey dataset of 4,464 students using leakage-safe nested cross-validation, leave-one-division-out validation, subgroup audits, and split-conformal prediction.

**Interpretation boundary.** The public data card describes a binary `Depression` field but does not document a validated diagnostic instrument or clinician assessment. We therefore analyze a **self-reported depression indicator**, not a clinical diagnosis. `Health_condition` is treated as a potential outcome proxy because its temporal relationship and questionnaire wording are undocumented.

Dataset: Haque and Alom, *Student Depression Detection Dataset*, Version 2, CC BY 4.0, DOI: [10.17632/xn9cb5zpgg.2](https://doi.org/10.17632/xn9cb5zpgg.2).
"""
    ),
    markdown(
        r"""
## Study design

1. Audit missingness, duplicates, prevalence, and conditional outcome rates.
2. Compare a **full** setting with a **proxy-free** setting that excludes `Health_condition`.
3. Use 5-fold outer cross-validation; tune hyperparameters only inside each training fold using 3-fold inner cross-validation and AUPRC.
4. Select one model per setting by pooled out-of-fold AUPRC.
5. Hold out each home division in turn to approximate geographic transportability.
6. Report gender, family-background, and division subgroup errors without claiming that observational parity proves fairness.
7. Evaluate 90% split-conformal prediction over five independent splits.

The primary metric is AUPRC because only 26.9% of records are positive. AUROC, Brier score, sensitivity, specificity, F1, and balanced accuracy are secondary metrics.
"""
    ),
    code(
        """
from pathlib import Path
import json
import platform

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from IPython.display import Image, display

from research_pipeline import run_all

print("Python:", platform.python_version())
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
"""
    ),
    markdown("## Execute the complete reproducible pipeline"),
    code(
        """
results = run_all(
    csv_path=Path("Student_Depression_Dataset.csv"),
    artifact_root=Path("artifacts"),
)
print("Selected models:", results["selected"])
"""
    ),
    markdown("## End-to-end study architecture"),
    code(
        """
display(Image(filename="artifacts/figures/figure_7_study_architecture.png", width=1050))
"""
    ),
    markdown("## Dataset and potential outcome-proxy audit"),
    code(
        """
display(results["audit"]["audit"])
display(results["audit"]["health"].style.format({"depression_rate": "{:.3f}"}))
display(Image(filename="artifacts/figures/figure_1_data_and_proxy_audit.png", width=900))
"""
    ),
    markdown(
        """
`Critical` health status perfectly co-occurs with a positive outcome, while `Good` health status perfectly co-occurs with a negative outcome. This does not by itself prove data leakage; it demonstrates severe construct overlap or outcome-proxy risk that must be stress-tested.
"""
    ),
    markdown("## Nested cross-validated model performance"),
    code(
        """
performance_columns = [
    "feature_set", "model", "AUROC", "AUROC_low", "AUROC_high",
    "AUPRC", "AUPRC_low", "AUPRC_high", "Brier",
    "Balanced_accuracy", "Sensitivity", "Specificity", "F1"
]
display(
    results["summary"][performance_columns]
    .sort_values(["feature_set", "AUPRC"], ascending=[True, False])
    .style.format(precision=3)
)
display(Image(filename="artifacts/figures/figure_2_nested_cv_auprc.png", width=900))
display(Image(filename="artifacts/figures/figure_8_all_model_comparison.png", width=1000))
display(Image(filename="artifacts/figures/figure_3_discrimination_curves.png", width=900))
display(Image(filename="artifacts/figures/figure_4_calibration.png", width=550))
"""
    ),
    markdown(
        """
Model selection uses AUPRC, not accuracy. The full model and proxy-free model answer different questions: the former estimates performance when all recorded variables are available; the latter estimates robustness to removing the strongest potential outcome proxy.
"""
    ),
    markdown("## Leave-one-division-out geographic validation"),
    code(
        """
display(results["geo_overall"].style.format(precision=3))
display(
    results["geo_detail"][[
        "feature_set", "model", "held_out_division", "n", "prevalence",
        "AUROC", "AUPRC", "Brier", "Sensitivity", "Specificity"
    ]].style.format(precision=3)
)
display(Image(filename="artifacts/figures/figure_5_geographic_transportability.png", width=900))
"""
    ),
    markdown("## Subgroup performance audit"),
    code(
        """
gaps = pd.read_csv("artifacts/tables/subgroup_metric_gaps.csv")
display(gaps.style.format(precision=3))

gender_rows = results["fairness"][results["fairness"]["group"] == "Gender"]
display(gender_rows.style.format(precision=3))
display(Image(filename="artifacts/figures/figure_9_division_error_disparity.png", width=1000))
"""
    ),
    markdown(
        """
These are descriptive subgroup audits. They do not establish individual fairness, causal fairness, or clinical safety. Differences can reflect sample size, outcome prevalence, measurement differences, or model behavior.
"""
    ),
    markdown("## Uncertainty and selective prediction"),
    code(
        """
display(results["conformal_runs"].style.format(precision=3))
display(results["conformal_runs"].mean(numeric_only=True).to_frame("mean").T.style.format(precision=3))
display(Image(filename="artifacts/figures/figure_6_selective_prediction.png", width=750))
display(Image(filename="artifacts/figures/figure_10_conformal_reliability.png", width=900))
"""
    ),
    markdown(
        r"""
At nominal 90% coverage, split-conformal prediction is evaluated over five independent data splits. Coverage is marginal under exchangeability; it is not guaranteed for every subgroup or after deployment shift. Non-singleton prediction sets should be interpreted as cases requiring further assessment, not automated decisions.
"""
    ),
    markdown("## Reproducibility checks"),
    code(
        """
required_files = [
    "artifacts/results/run_manifest.json",
    "artifacts/results/oof_predictions.csv",
    "artifacts/tables/nested_cv_performance.csv",
    "artifacts/tables/geographic_validation_overall.csv",
    "artifacts/tables/subgroup_fairness_metrics.csv",
    "artifacts/tables/conformal_runs.csv",
    "artifacts/figures/figure_7_study_architecture.png",
    "artifacts/figures/figure_8_all_model_comparison.png",
    "artifacts/figures/figure_9_division_error_disparity.png",
    "artifacts/figures/figure_10_conformal_reliability.png",
]
for filename in required_files:
    path = Path(filename)
    assert path.exists() and path.stat().st_size > 0, filename

assert len(results["raw"]) == 4464
assert int(results["y"].sum()) == 1201
assert results["predictions"].groupby(["feature_set", "model"]).size().eq(4464).all()
assert results["predictions"]["probability"].between(0, 1).all()
assert set(results["selected"]) == {"full", "proxy_free"}

with open("artifacts/results/run_manifest.json", encoding="utf-8") as handle:
    manifest = json.load(handle)
display(manifest["selected_models"])
print("All reproducibility checks passed.")
"""
    ),
    markdown(
        r"""
## Conclusions

The near-perfect full-model result is not sufficient evidence of deployable depression screening. Performance falls sharply when `Health_condition` is removed and falls further under division-held-out testing. The defensible contribution is therefore an outcome-proxy sensitivity and transportability audit, not a claim of clinical diagnosis.

External validation using a documented instrument such as PHQ-9 or CES-D is required before clinical or institutional use. No result in this notebook should be used to diagnose, rank, or discipline students.
"""
    ),
]


notebook = nbf.v4.new_notebook(cells=cells)
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.13"},
}

output = Path("student_depression_proxy_audit.ipynb")
nbf.write(notebook, output)
print(output.resolve())
