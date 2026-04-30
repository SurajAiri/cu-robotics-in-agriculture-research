# Results Summary

This report summarizes the final model evaluation for the crop yield prediction research project. The goal of this phase was to compare several regression models on the processed agricultural dataset and select the strongest candidate for production use in the phase-II decision support app.

## Evaluation Snapshot

The final comparison used three metrics:

- RMSE, where lower is better
- MAE, where lower is better
- R2, where higher is better

## Final Ranking

| Rank | Model         |   RMSE |    MAE |     R2 |
| ---- | ------------- | -----: | -----: | -----: |
| 1    | XGBoost       | 0.3519 | 0.1938 | 0.8617 |
| 2    | Extra Trees   | 0.3636 | 0.1926 | 0.8524 |
| 3    | CatBoost      | 0.3653 | 0.2084 | 0.8511 |
| 4    | Random Forest | 0.3885 | 0.2151 | 0.8315 |

XGBoost is the current champion model. Extra Trees is kept as a rollback model because its performance remains close to the top score.

## Best Hyperparameters

- XGBoost: `n_estimators=975`, `learning_rate=0.11`, `max_depth=9`, `subsample=0.82`, `colsample_bytree=0.87`
- CatBoost: `depth=9`, `iterations=808`, `l2_leaf_reg=2`, `learning_rate=0.1544`
- Random Forest: `n_estimators=591`, `min_samples_leaf=3`, `min_samples_split=2`
- Extra Trees: `n_estimators=800`, `bootstrap=True`, `min_samples_leaf=2`, `min_samples_split=5`

## Key Takeaways

- The boosting models outperformed the simpler baselines on test performance.
- XGBoost achieved the best overall balance of low error and high explanatory power.
- The model gap between XGBoost, Extra Trees, and CatBoost is small, which makes the rollback model a practical fallback.
- The results are suitable for the current recommendation workflow, but the robot actions themselves remain conceptual and require separate field validation.

## Figures

The following figures summarize the final model comparison:

- [R2 score comparison](figures/final_model_r2_score.png)
- [Error metrics comparison](figures/final_model_error_metrics.png)
- [Combined performance comparison](figures/final_model_performance_comparison.png)

## Supporting Files

- [Detailed tuning notes](../results/hypertuned_info.md)
- [Model comparison CSV](../results/top_model_result.csv)
