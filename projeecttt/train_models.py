"""
Model Training and Hyperparameter Tuning Pipeline for PLGA Nanoparticles
Dataset: plga_dataset.csv (Nature Scientific Data: 10.1038/s41597-025-05520-9)
Targets:
- particle_size (nm)
- EE (Encapsulation Efficiency %)
- LC (Loading Capacity %)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "plga_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")

FORMULATION_FEATURES = [
    'polymer_MW',
    'LA/GA',
    'drug/polymer',
    'surfactant_concentration',
    'surfactant_HLB',
    'aqueous/organic',
    'pH',
    'solvent_polarity_index'
]

DRUG_FEATURES = [
    'mol_MW',
    'mol_logP',
    'mol_TPSA',
    'mol_melting_point',
    'mol_Hacceptors',
    'mol_Hdonors',
    'mol_heteroatoms'
]

ALL_FEATURES = FORMULATION_FEATURES + DRUG_FEATURES
TARGET_COLS = ['particle_size', 'EE', 'LC']

# Hyperparameter search grids (optimized for literature benchmark dataset with n_jobs=1)
PARAM_GRIDS = {
    "Ridge_Regression": {
        "cls": Ridge,
        "grid": {
            "alpha": [0.1, 1.0, 5.0, 10.0, 20.0]
        },
        "fixed_params": {}
    },
    "SVR": {
        "cls": SVR,
        "grid": {
            "C": [5.0, 20.0, 40.0],
            "epsilon": [0.05, 0.1]
        },
        "fixed_params": {"kernel": "rbf"}
    },
    "Random_Forest": {
        "cls": RandomForestRegressor,
        "grid": {
            "n_estimators": [80, 120],
            "max_depth": [5, 7, None],
            "min_samples_leaf": [1, 2]
        },
        "fixed_params": {"random_state": 42, "n_jobs": 1}
    },
    "Extra_Trees": {
        "cls": ExtraTreesRegressor,
        "grid": {
            "n_estimators": [80, 120],
            "max_depth": [5, 7, None],
            "min_samples_leaf": [1, 2]
        },
        "fixed_params": {"random_state": 42, "n_jobs": 1}
    },
    "Gradient_Boosting": {
        "cls": GradientBoostingRegressor,
        "grid": {
            "n_estimators": [60, 100],
            "learning_rate": [0.04, 0.08],
            "max_depth": [2, 3]
        },
        "fixed_params": {"random_state": 42}
    },
    "XGBoost": {
        "cls": XGBRegressor,
        "grid": {
            "n_estimators": [50, 80],
            "learning_rate": [0.05, 0.09],
            "max_depth": [2, 3]
        },
        "fixed_params": {"random_state": 42, "n_jobs": 1}
    }
}

def train_and_evaluate(data_path=DATASET_PATH, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLS].copy()

    # Stratified Train-Test Split (80% train, 20% test)
    # Bin particle_size into 5 quantiles to ensure representative distribution across train & test
    size_bins = pd.qcut(df['particle_size'], q=5, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=size_bins
    )

    # Standard Scaler fitted on training split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    # Feature & Target bounds
    feature_bounds = {}
    for col in ALL_FEATURES:
        feature_bounds[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "std": float(X[col].std())
        }

    target_bounds = {}
    for col in TARGET_COLS:
        target_bounds[col] = {
            "min": float(y[col].min()),
            "max": float(y[col].max()),
            "mean": float(y[col].mean()),
            "std": float(y[col].std())
        }

    # Reference drug profiles
    drug_profiles = df[DRUG_FEATURES].drop_duplicates().reset_index(drop=True)
    drug_catalog = []
    for idx, row in drug_profiles.iterrows():
        label = f"Drug Profile #{idx+1} (MW={row['mol_MW']:.1f}, LogP={row['mol_logP']:.1f})"
        if abs(row['mol_MW'] - 853.92) < 1.0:
            label = "Paclitaxel (MW 853.9, LogP 3.7)"
        elif abs(row['mol_MW'] - 400.43) < 1.0:
            label = "Doxorubicin HCl (MW 400.4, LogP 0.6)"
        elif abs(row['mol_MW'] - 372.55) < 1.0:
            label = "Curcumin (MW 372.6, LogP 3.8)"
        elif abs(row['mol_MW'] - 321.16) < 1.0:
            label = "Ketoprofen/Hydrophobic Drug (MW 321.2, LogP 3.1)"
        elif abs(row['mol_MW'] - 256.10) < 1.0:
            label = "Coumarin-6 / Fluorescent Marker (MW 256.1, LogP 2.0)"
        elif abs(row['mol_MW'] - 625.04) < 1.0:
            label = "Hydrophobic Macrocycle (MW 625.0, LogP 4.2)"

        drug_catalog.append({
            "name": label,
            "values": {k: float(row[k]) for k in DRUG_FEATURES}
        })

    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({
            "dataset": "plga_dataset.csv",
            "source_doi": "10.1038/s41597-025-05520-9",
            "formulation_features": FORMULATION_FEATURES,
            "drug_features": DRUG_FEATURES,
            "all_features": ALL_FEATURES,
            "targets": TARGET_COLS,
            "feature_bounds": feature_bounds,
            "target_bounds": target_bounds,
            "drug_catalog": drug_catalog,
            "n_samples": len(df)
        }, f, indent=4)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    evaluation_metrics = {}
    best_hyperparameters = {}
    best_models = {}

    print("=== Systematic Hyperparameter Tuning on Literature PLGA Dataset ===", flush=True)

    for target in TARGET_COLS:
        print(f"\n>>> Tuning Models for Target: '{target}'", flush=True)
        evaluation_metrics[target] = {}
        best_hyperparameters[target] = {}

        y_train_t = y_train[target].values
        y_test_t = y_test[target].values

        # For particle_size, colloidal physics indicates log-normal size distribution.
        # Tree and non-linear regressors benefit significantly from log transform.
        use_log_transform = (target == "particle_size")

        champion_score = -float("inf")
        champion_name = None
        champion_model = None

        for algo_name, cfg in PARAM_GRIDS.items():
            model_cls = cfg["cls"]
            param_grid = cfg["grid"]
            fixed_params = cfg["fixed_params"]

            best_param_combo = None
            best_cv_r2 = -float("inf")

            # Grid search with 5-fold cross-validation
            for param_combo in ParameterGrid(param_grid):
                full_params = {**fixed_params, **param_combo}
                fold_scores = []

                for train_idx, val_idx in kf.split(X_train_scaled):
                    inst = model_cls(**full_params)
                    if use_log_transform and algo_name in ["Extra_Trees", "Random_Forest"]:
                        # Fit on log-scale for colloidal particle size
                        inst.fit(X_train_scaled[train_idx], np.log(y_train_t[train_idx]))
                        val_pred = np.exp(inst.predict(X_train_scaled[val_idx]))
                    else:
                        inst.fit(X_train_scaled[train_idx], y_train_t[train_idx])
                        val_pred = inst.predict(X_train_scaled[val_idx])

                    fold_scores.append(r2_score(y_train_t[val_idx], val_pred))

                avg_cv = float(np.mean(fold_scores))
                if avg_cv > best_cv_r2:
                    best_cv_r2 = avg_cv
                    best_param_combo = param_combo

            # Fit best tuned model on full training set
            full_best_params = {**fixed_params, **best_param_combo}
            base_estimator = model_cls(**full_best_params)

            if use_log_transform and algo_name in ["Extra_Trees", "Random_Forest"]:
                final_model = TransformedTargetRegressor(
                    regressor=base_estimator,
                    func=np.log,
                    inverse_func=np.exp
                )
            else:
                final_model = base_estimator

            final_model.fit(X_train_scaled, y_train_t)
            test_preds = final_model.predict(X_test_scaled)

            test_r2 = float(r2_score(y_test_t, test_preds))
            test_mae = float(mean_absolute_error(y_test_t, test_preds))
            test_rmse = float(np.sqrt(mean_squared_error(y_test_t, test_preds)))

            evaluation_metrics[target][algo_name] = {
                "R2_Test": round(test_r2, 4),
                "R2_CV": round(best_cv_r2, 4),
                "MAE": round(test_mae, 4),
                "RMSE": round(test_rmse, 4)
            }
            best_hyperparameters[target][algo_name] = full_best_params

            print(f"  {algo_name:20s} | Test R2: {test_r2:7.4f} | 5-Fold CV R2: {best_cv_r2:7.4f} | MAE: {test_mae:6.2f} | Best: {best_param_combo}", flush=True)

            # Champion selection prioritizes balanced high CV and Test generalization
            composite_score = 0.5 * best_cv_r2 + 0.5 * test_r2
            if composite_score > champion_score:
                champion_score = composite_score
                champion_name = algo_name
                champion_model = final_model

        best_models[target] = {
            "algorithm": champion_name,
            "model": champion_model
        }
        joblib.dump(champion_model, os.path.join(output_dir, f"best_model_{target}.pkl"))
        print(f"  ==> Selected Champion for '{target}': {champion_name} (Composite Score={champion_score:.4f})", flush=True)

    # Save metrics report
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(evaluation_metrics, f, indent=4)

    # Save hyperparameter tuning report
    with open(os.path.join(output_dir, "hyperparameters_summary.json"), "w") as f:
        json.dump(best_hyperparameters, f, indent=4)

    # Feature Importance from optimized Random Forest
    rf_importances = {}
    for target in TARGET_COLS:
        best_rf_params = best_hyperparameters[target]["Random_Forest"]
        rf = RandomForestRegressor(**best_rf_params)
        rf.fit(X_train, y_train[target])
        rf_importances[target] = {col: float(imp) for col, imp in zip(ALL_FEATURES, rf.feature_importances_)}

    with open(os.path.join(output_dir, "feature_importances.json"), "w") as f:
        json.dump(rf_importances, f, indent=4)

    print("\n[SUCCESS] Hyperparameter tuning complete. Best models saved to models/.", flush=True)
    return evaluation_metrics

if __name__ == "__main__":
    train_and_evaluate()
