# Manuscript build

The manuscript is written in IEEE conference format and is ready for anonymous review after venue-specific formatting checks.

Build from this directory with a TeX distribution that includes `IEEEtran`:

```powershell
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

The figures are loaded from `../artifacts/figures/`. Author names and affiliations are intentionally anonymous because none were supplied and many conferences use double-blind review.

Before submission:

1. Select the venue and apply its exact template/page limit.
2. Add genuine author metadata only when the anonymity policy permits it.
3. Confirm whether the venue requires an ethics-review determination for secondary public data.
4. Add external validation or primary-data evidence if available.
5. Do not describe the binary field as a clinical diagnosis unless the dataset creators document a validated reference standard.
