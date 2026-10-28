"""
Hyperparameter Tuning Script for PLGA Nanoparticle Dataset
Optimizes Regression Architectures for particle_size, EE, and LC.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

warnings.filterwarnings('ignore')

def run_tuning():
    df = pd.read_csv('plga_dataset.csv')
    features = [
        'polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA',
        'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms',
        'drug/polymer', 'surfactant_concentration', 'surfactant_HLB',
        'aqueous/organic', 'pH', 'solvent_polarity_index'
    ]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for target in ['particle_size', 'EE', 'LC']:
        y = df[target].values
        print(f"\n=======================================================")
        print(f"       HYPERPARAMETER TUNING FOR: {target}")
        print(f"=======================================================")

        # 1. Random Forest
        rf_params = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 0.6, 0.8, 1.0]
        }
        rf_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=kf, scoring='r2', n_jobs=-1)
        rf_search.fit(X_scaled, y)
        print(f"Random Forest Best CV R2: {rf_search.best_score_:.4f}")
        print(f"  Params: {rf_search.best_params_}")

        # 2. Extra Trees
        et_params = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 0.6, 0.8, 1.0]
        }
        et_search = GridSearchCV(ExtraTreesRegressor(random_state=42), et_params, cv=kf, scoring='r2', n_jobs=-1)
        et_search.fit(X_scaled, y)
        print(f"Extra Trees Best CV R2:   {et_search.best_score_:.4f}")
        print(f"  Params: {et_search.best_params_}")

        # 3. Gradient Boosting
        gb_params = {
            'n_estimators': [50, 80, 120, 150],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
            'max_depth': [2, 3, 4],
            'subsample': [0.6, 0.8, 1.0],
            'min_samples_leaf': [1, 2, 3]
        }
        gb_search = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=kf, scoring='r2', n_jobs=-1)
        gb_search.fit(X_scaled, y)
        print(f"Gradient Boost Best CV R2:{gb_search.best_score_:.4f}")
        print(f"  Params: {gb_search.best_params_}")

        # 4. XGBoost
        xgb_params = {
            'n_estimators': [40, 60, 80, 120],
            'learning_rate': [0.02, 0.04, 0.07, 0.1],
            'max_depth': [2, 3, 4, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5],
            'reg_lambda': [0.5, 1.0, 2.0]
        }
        xgb_search = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=kf, scoring='r2', n_jobs=-1)
        xgb_search.fit(X_scaled, y)
        print(f"XGBoost Best CV R2:       {xgb_search.best_score_:.4f}")
        print(f"  Params: {xgb_search.best_params_}")

        # 5. SVR
        svr_params = {
            'C': [1.0, 5.0, 10.0, 20.0, 40.0, 80.0],
            'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
            'gamma': ['scale', 'auto', 0.01, 0.05, 0.1]
        }
        svr_search = GridSearchCV(SVR(), svr_params, cv=kf, scoring='r2', n_jobs=-1)
        svr_search.fit(X_scaled, y)
        print(f"SVR Best CV R2:           {svr_search.best_score_:.4f}")
        print(f"  Params: {svr_search.best_params_}")

if __name__ == '__main__':
    run_tuning()
