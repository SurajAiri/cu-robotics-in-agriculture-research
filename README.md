# Robotics In Agriculture Research

This repository documents the research pipeline behind a crop yield prediction system for a future robotics workflow. The current scope focuses on the machine learning side: preparing agricultural data, training and evaluating regression models, and using the best model to power a phase-II decision support app that recommends environment adjustments for higher predicted yield.

## What This Project Covers

- Agricultural data collection and cleaning
- Feature engineering and preprocessing
- Training and comparing multiple regression models
- Hyperparameter tuning and model selection
- A Streamlit-based phase-II optimization app for crop and environment recommendations

## Documentation Map

- [Architecture overview](docs/architecture.md)
- [Phase I: Model training and selection](docs/phase-1-model-training.md)
- [Phase II: Robotic decision support](docs/phase-2-robotic-decision-support.md)

## Quick Start

### Prerequisites

- Python 3.12+
- uv

### Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Run the Phase-II app

```bash
uv run streamlit run app.py
```

### Run tests

```bash
uv run pytest
```

## End-to-End Flow

1. Agricultural data is collected and selected from the Kaggle crop yield dataset.
2. Raw data is cleaned, validated, encoded, and normalized into processed datasets.
3. Multiple regression models are trained and tuned on the processed data.
4. The best models are stored as production artifacts in `models/production`.
5. The Streamlit app loads the production pipelines and evaluates environment variations.
6. The app recommends the configuration with the highest predicted yield.

## Final Model Summary

The latest evaluation in [results/hypertuned_info.md](results/hypertuned_info.md) records the following test-set performance:

| Model         |   RMSE |    MAE |     R2 |
| ------------- | -----: | -----: | -----: |
| Random Forest | 0.3885 | 0.2151 | 0.8315 |
| Extra Trees   | 0.3636 | 0.1926 | 0.8524 |
| XGBoost       | 0.3519 | 0.1938 | 0.8617 |
| CatBoost      | 0.3653 | 0.2084 | 0.8511 |

The current champion model used by the app is XGBoost, with Extra Trees kept as the rollback option.

## Repository Layout

```text
├── app.py
├── configs/
├── data/
├── docs/
├── models/
├── notebooks/
├── reports/
├── results/
├── scripts/
├── src/
└── tests/
```

## Data Source

The dataset documentation lives in [data/source.md](data/source.md). It uses the Agricultural Crop Yield in Indian States dataset and covers crop, year, season, state, area, production, rainfall, fertilizer, pesticide, and yield.

## Production Assets

- Champion model: [models/production/champion_xgboost_pipeline.joblib](models/production/champion_xgboost_pipeline.joblib)
- Rollback model: [models/production/rollback_extratrees_pipeline.joblib](models/production/rollback_extratrees_pipeline.joblib)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Author

Suraj Airi - surajairi.ml@gmail.com
