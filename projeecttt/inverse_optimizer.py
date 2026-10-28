"""
Inverse Formulation Generation Engine for PLGA Nanoparticles
Trained on Literature Benchmark Dataset: plga_dataset.csv
(Nature Scientific Data: 10.1038/s41597-025-05520-9)
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Normalized / Standard Benchmark Drug State (Calculated median from plga_dataset.csv)
# Represents a typical lipophilic small-molecule therapeutic drug
NORMALIZED_DEFAULT_DRUG = {
    'mol_MW': 297.61,           # g/mol (~300 g/mol typical small-molecule)
    'mol_logP': 3.10,           # Moderately lipophilic, optimal for PLGA entrapment
    'mol_TPSA': 62.54,          # Polar surface area (Å²)
    'mol_melting_point': 167.0, # Stable crystalline melting temperature (°C)
    'mol_Hacceptors': 5,
    'mol_Hdonors': 1,
    'mol_heteroatoms': 6
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "plga_dataset.csv")

class PLGANanoparticleInverseDesigner:
    def __init__(self, models_dir=None, data_path=None):
        if models_dir is None:
            models_dir = DEFAULT_MODELS_DIR
        elif not os.path.isabs(models_dir):
            models_dir = os.path.join(BASE_DIR, models_dir)

        if data_path is None:
            data_path = DEFAULT_DATASET_PATH
        elif not os.path.isabs(data_path):
            data_path = os.path.join(BASE_DIR, data_path)

        self.models_dir = models_dir
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        
        with open(os.path.join(models_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
            
        self.formulation_features = self.metadata["formulation_features"]
        self.drug_features = self.metadata["drug_features"]
        self.all_features = self.metadata["all_features"]
        self.targets = self.metadata["targets"]
        self.feature_bounds = self.metadata["feature_bounds"]
        self.target_bounds = self.metadata["target_bounds"]
        self.drug_catalog = self.metadata.get("drug_catalog", [])
        
        # Load best trained models for each target
        self.models = {}
        for target in self.targets:
            self.models[target] = joblib.load(os.path.join(models_dir, f"best_model_{target}.pkl"))
            
        if os.path.exists(data_path):
            self.historical_df = pd.read_csv(data_path)
        else:
            self.historical_df = None

    def predict_properties(self, X_input):
        """
        Predict [particle_size, EE, LC] from feature matrix
        """
        if isinstance(X_input, pd.DataFrame):
            X_df = X_input[self.all_features]
        else:
            X_arr = np.asarray(X_input, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            X_df = pd.DataFrame(X_arr, columns=self.all_features)
            
        X_scaled = self.scaler.transform(X_df)
        preds = {}
        for target, model in self.models.items():
            preds[target] = model.predict(X_scaled)
        return preds

    def search_candidates(
        self,
        target_size_nm,
        target_ee_pct,
        drug_properties=None,
        weight_size=1.0,
        weight_ee=1.0,
        num_candidates=3
    ):
        """
        Fast High-Precision Inverse Search:
        Finds PLGA formulation parameters that best achieve target_size_nm and target_ee_pct.
        If drug_properties is None, uses the NORMALIZED_DEFAULT_DRUG state.
        """
        if drug_properties is None:
            drug_properties = NORMALIZED_DEFAULT_DRUG

        form_bounds = [
            (self.feature_bounds[col]["min"], self.feature_bounds[col]["max"])
            for col in self.formulation_features
        ]

        drug_vals = [float(drug_properties[col]) for col in self.drug_features]

        # 1. Candidate space sampling
        np.random.seed(42)
        n_samples = 2500
        random_form = np.random.uniform(
            [b[0] for b in form_bounds],
            [b[1] for b in form_bounds],
            size=(n_samples, len(form_bounds))
        )

        all_form = random_form
        if self.historical_df is not None:
            hist_form = self.historical_df[self.formulation_features].values
            all_form = np.vstack([hist_form, random_form])

        # Batch evaluation
        n_pts = len(all_form)
        drug_matrix = np.tile(drug_vals, (n_pts, 1))
        X_full = np.hstack([all_form, drug_matrix])

        preds = self.predict_properties(X_full)
        sizes = preds['particle_size']
        ees = preds['EE']
        lcs = preds['LC']

        # Normalized error loss
        err_size = ((sizes - target_size_nm) / target_size_nm) ** 2
        err_ee = ((ees - target_ee_pct) / target_ee_pct) ** 2
        losses = weight_size * err_size + weight_ee * err_ee

        # 2. Extract top diverse candidates
        sorted_indices = np.argsort(losses)
        selected_candidates = []

        for idx in sorted_indices:
            cand = all_form[idx]
            is_close = False
            for existing in selected_candidates:
                dist = np.linalg.norm((cand - existing) / [b[1] - b[0] for b in form_bounds])
                if dist < 0.20:
                    is_close = True
                    break
            if not is_close:
                selected_candidates.append(cand)
            if len(selected_candidates) >= num_candidates:
                break

        # 3. Format results
        results = []
        for cand in selected_candidates:
            cand_full = np.hstack([cand, drug_vals]).reshape(1, -1)
            p = self.predict_properties(cand_full)
            p_size = float(p['particle_size'][0])
            p_ee = float(p['EE'][0])
            p_lc = float(p['LC'][0])

            size_err = abs(p_size - target_size_nm) / target_size_nm * 100.0
            ee_err = abs(p_ee - target_ee_pct) / target_ee_pct * 100.0
            score = 100.0 - (0.5 * size_err + 0.5 * ee_err)
            score = max(0.0, round(score, 2))

            formulation_dict = {
                self.formulation_features[i]: round(float(cand[i]), 3)
                for i in range(len(self.formulation_features))
            }

            results.append({
                "formulation": formulation_dict,
                "predicted_size_nm": round(p_size, 1),
                "predicted_ee_pct": round(p_ee, 1),
                "predicted_lc_pct": round(p_lc, 2),
                "size_error_pct": round(size_err, 2),
                "ee_error_pct": round(ee_err, 2),
                "match_score": score
            })

        results.sort(key=lambda item: item["match_score"], reverse=True)
        return results
