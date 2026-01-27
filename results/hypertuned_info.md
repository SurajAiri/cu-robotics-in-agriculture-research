**Best Hyperparameters Found**

1. Catboost (champion)
   regressor_depth: 9
   regressor_iterations: 808
   regressor_l2_leaf_reg: 2
   regressor_learning_rate: np.float64(0.15439975445336496)
2. XGBoost
   n_estimators: 975
   learning_rate: 0.11
   max_depth: 9
   subsample: 0.82
   colsample_bytree: 0.87

3. Random Forest
   'regressor_max_depth': None, 'regressor_max_features': None, 'regressor_min_samples_leaf': 3, 'regressor_min_samples_split': 2, 'regressor_n_estimators': 591

4. Extra Trees
   'regressor_bootstrap': True, 'regressor_max_depth': None, 'regressor_max_features': None, 'regressor_min_samples_leaf': 2, 'regressor_min_samples_split': 5, 'regressor_n_estimators': 800

```
1. catboost

{'regressor__depth': 9, 'regressor__iterations': 808, 'regressor__l2_leaf_reg': 2, 'regressor__learning_rate': np.float64(0.15439975445336496)}

--- Tuning Random Forest ---
Fitting 3 folds for each of 30 candidates, totalling 90 fits
Best params for Random Forest: {'regressor__max_depth': None, 'regressor__max_features': None, 'regressor__min_samples_leaf': 3, 'regressor__min_samples_split': 2, 'regressor__n_estimators': 591}

--- Tuning Extra Trees ---
Fitting 3 folds for each of 30 candidates, totalling 90 fits
Best params for Extra Trees: {'regressor__bootstrap': True, 'regressor__max_depth': None, 'regressor__max_features': None, 'regressor__min_samples_leaf': 2, 'regressor__min_samples_split': 5, 'regressor__n_estimators': 800}
```

--- Final Evaluation on Test Set ---
Random Forest: RMSE=0.3885, MAE=0.2151, R2=0.8315
Extra Trees: RMSE=0.3636, MAE=0.1926, R2=0.8524
XGBoost: RMSE=0.3519, MAE=0.1938, R2=0.8617
CatBoost: RMSE=0.3653, MAE=0.2084, R2=0.8511

Saved performance summary to results/top_model_result.csv
Model RMSE MAE R2
0 Random Forest 0.388463 0.215093 0.831538
1 Extra Trees 0.363633 0.192573 0.852386
2 XGBoost 0.351934 0.193767 0.861731
3 CatBoost 0.365259 0.208427 0.851062
