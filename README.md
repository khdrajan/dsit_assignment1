# Assignment 1 — Predictive Modelling of Eating-Out Problem

This repository contains the full data science workflow (EDA → Modelling → PySpark → Reproducibility) for the Sydney restaurants dataset.

## Structure
```
.
├── data
│   ├── raw/               # Provided CSV + GeoJSON
│   └── processed/         # Cleaned / engineered datasets (DVC outputs)
├── models/                # Saved models (tracked via DVC/Git LFS)
├── notebooks/             # 01_eda.ipynb, 02_modelling.ipynb, 03_pyspark.ipynb
├── reports/
│   └── figures/           # Plots exported from notebooks
├── src/
│   ├── preprocess.py
│   ├── featurize.py
│   ├── train_regression.py
│   ├── train_classification.py
│   └── evaluate.py
├── dvc.yaml               # Pipeline definition
├── params.yaml            # Hyperparameters/shared config
├── requirements.txt
└── README.md
```

## How to run (local)
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. (Optional) Initialize Git, Git LFS and DVC:
   ```bash
   git init
   git lfs install
   dvc init
   dvc add data/raw/zomato_df_final_data.csv
   dvc add data/raw/sydney.geojson
   git add . && git commit -m "Init repo with data and pipeline"
   ```
3. Reproduce the pipeline with DVC:
   ```bash
   dvc repro
   ```
4. Work through notebooks in `notebooks/` starting with `01_eda.ipynb`.

## Notes
- PySpark steps are in `03_pyspark.ipynb`. 
- Figures are exported to `reports/figures` for inclusion in the PDF report.
- Large artifacts (models, processed data) should be tracked by DVC and/or Git LFS.
