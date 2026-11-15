# Student depression outcome-proxy audit

This repository contains an executed, reproducible analysis and an anonymous-review manuscript for the public **Student Depression Detection Dataset** (Version 2, DOI `10.17632/xn9cb5zpgg.2`, CC BY 4.0).

## Main finding

The full model achieved near-perfect internal performance, but the result was highly dependent on `Health_condition`, a potential outcome proxy. Excluding that variable reduced nested-CV AUPRC from 0.998 to 0.520; leave-one-home-division-out AUPRC was 0.452.

These results concern a binary self-reported indicator. They are not evidence of clinical diagnosis or deployment readiness.

## Files

- `student_depression_proxy_audit.ipynb` — fully executed notebook with saved outputs.
- `research_pipeline.py` — reusable analysis implementation.
- `build_notebook.py` — notebook source builder.
- `artifacts/` — figures, tables, OOF predictions, and run manifest.
- `paper/manuscript.tex` — full anonymous IEEE-style paper draft.
- `paper/references.bib` — references.

## Reproduce

```powershell
python -m pip install -r requirements.txt
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 student_depression_proxy_audit.ipynb
```

The analyzed CSV SHA-256 is `563CF3A1890312822244BD4A6F12C2D78318CB7036E063B3D0C97A358C690556`.

## Responsible-use boundary

Do not use the notebook or its predictions to diagnose, rank, discipline, or automatically refer students. Prospective use requires a documented outcome instrument, external validation, ethics review, informed consent, privacy safeguards, and a human-led support pathway.
