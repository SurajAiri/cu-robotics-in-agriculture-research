# Phase I: Model Training and Selection

Phase I is the research and training layer of the project. It takes the agricultural dataset, prepares it for modeling, trains multiple regressors, and selects the best-performing pipeline for production use.

## Goal

Build a reliable crop yield prediction model that can later support the phase-II decision support system.

## Dataset

The project uses the [Agricultural Crop Yield in Indian States dataset](../data/source.md), which contains crop, crop year, season, state, area, production, rainfall, fertilizer, pesticide, and yield.

## Training Flow

1. Collect and inspect the raw agricultural dataset.
2. Clean the data and create a base dataset.
3. Engineer features and prepare the preprocessing pipeline.
4. Train multiple regression models.
5. Evaluate the candidates using RMSE, MAE, and R2.
6. Tune the strongest models and save the production-ready pipelines.

## Main Processing Steps

- Standardize and clean categorical and numeric fields.
- Validate the processed data before training.
- Encode categorical variables.
- Normalize or scale numeric features where needed.
- Build reusable preprocessing and model pipelines.

## Models Explored

The repository history and training artifacts show comparisons across:

- Random Forest
- Extra Trees
- XGBoost
- CatBoost

## Final Evaluation

The latest summary in [results/hypertuned_info.md](../results/hypertuned_info.md) records the following test-set metrics:

| Model         |   RMSE |    MAE |     R2 |
| ------------- | -----: | -----: | -----: |
| Random Forest | 0.3885 | 0.2151 | 0.8315 |
| Extra Trees   | 0.3636 | 0.1926 | 0.8524 |
| XGBoost       | 0.3519 | 0.1938 | 0.8617 |
| CatBoost      | 0.3653 | 0.2084 | 0.8511 |

XGBoost is the current champion model in the production flow, while Extra Trees is retained as a rollback option.

## Production Artifacts

- Champion pipeline: [models/production/champion_xgboost_pipeline.joblib](../models/production/champion_xgboost_pipeline.joblib)
- Rollback pipeline: [models/production/rollback_extratrees_pipeline.joblib](../models/production/rollback_extratrees_pipeline.joblib)

## What This Phase Does Not Cover

- It does not control agricultural hardware directly.
- It does not validate live robot actions in the field.
- It does not claim autonomous execution beyond the prediction and recommendation layer.
