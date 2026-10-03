# `artifacts/figures/` — generated, not committed

Figures are written here by the pipeline and by the standalone figure tool:

```bash
python tools/make_paper_figures.py
```

That tool expects a **cost** result set (`artifacts/results/*/cost.csv` or
`artifacts/results/e*_cost.csv`) to draw the cost-scaling panel. No cost run is
shipped in this repository, so the tool currently produces nothing for that
panel — run `configs/e4_cost.yaml` first.
