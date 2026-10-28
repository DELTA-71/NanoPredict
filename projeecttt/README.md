# AI-Based Prediction of Nanoparticle Formulation from Target Particle Size and Encapsulation Efficiency

**B.Tech Biotechnology Minor Project**
**Dataset:** `plga_dataset.csv` (*Nature Scientific Data: 10.1038/s41597-025-05520-9*, Mendeley: `10.17632/sbjf5csrdm.1`)

---

## 📌 Project Overview
This project develops an **Inverse-Design AI/ML Framework** for Poly(lactic-co-glycolic acid) (PLGA) nanoparticle drug delivery systems:

$$\text{Target Particle Size (nm)} + \text{Target Encapsulation Efficiency (EE\%)} \xrightarrow{\text{AI Inverse Optimization}} \text{PLGA Formulation Recipe}$$

Instead of trial-and-error laboratory synthesis, the researcher inputs desired nanoparticle specifications, and the system computes the recommended PLGA formulation parameters with predicted properties and loading capacities.

---

## 🔬 Dataset Features & Chemical Definitions (`plga_dataset.csv`)

### 1. Formulation & Processing Variables (Search Space)
- **`polymer_MW`**: PLGA Polymer Molecular Weight (1.22 – 98.0 kDa)
- **`LA/GA`**: Lactide-to-Glycolide molar ratio (1.0 = PLGA 50:50, 3.0 = PLGA 75:25)
- **`drug/polymer`**: Initial drug-to-polymer mass ratio (0.005 – 1.0 w/w)
- **`surfactant_concentration`**: Surfactant concentration (0.0 – 2.34% w/v)
- **`surfactant_HLB`**: Surfactant Hydrophile-Lipophile Balance (18.0 for PVA, 20.5 for Pluronic F68)
- **`aqueous/organic`**: Aqueous-to-organic phase volume ratio (2.0 – 40.0 v/v)
- **`pH`**: Aqueous phase pH condition code (-2, -1, 0, 1)
- **`solvent_polarity_index`**: Organic solvent polarity (4.0 for DCM, 5.1 for Acetone, 7.2 for DMSO)

### 2. Active Pharmaceutical Drug Descriptors
- **`mol_MW`**: Molecular Weight of encapsulated drug (166.2 – 853.9 g/mol)
- **`mol_logP`**: Octanol/water partition coefficient (Lipophilicity, 0.12 – 8.91)
- **`mol_TPSA`**: Topological Polar Surface Area (32.3 – 221.3 Å²)
- **`mol_melting_point`**: Melting point of drug (61.0 – 300.0 °C)
- **`mol_Hacceptors`**, **`mol_Hdonors`**, **`mol_heteroatoms`**: Hydrogen bonding & heteroatom count

### 3. Target Nanoparticle Properties
- **`particle_size`**: Hydrodynamic diameter (nm)
- **`EE`**: Drug Encapsulation Efficiency (%)
- **`LC`**: Drug Loading Capacity (%)

---

## 💻 How to Run the Streamlit Web Application

### Method 1: Double-Click Desktop Launcher
Double-click [`run_app.bat`](file:///c:/Users/AYUSH%20KUMAR/Desktop/projeecttt/run_app.bat) in this folder.

### Method 2: Terminal / Command Prompt
Open PowerShell or Command Prompt in `c:\Users\AYUSH KUMAR\Desktop\projeecttt` and run:
```bash
streamlit run app.py
```
Your browser will open automatically at: `http://localhost:8501`.

---

## 🏗️ Project Architecture & File Structure

```text
projeecttt/
├── plga_dataset.csv               # Literature benchmark dataset (Nature Scientific Data)
├── app.py                         # Streamlit Web Application (4 focused modules)
├── inverse_optimizer.py           # PLGA Inverse Formulation Optimization Engine
├── train_models.py                # Multi-model regression training & evaluation pipeline
├── run_app.bat                    # One-click Windows desktop batch launcher
├── requirements.txt               # Python dependencies
├── README.md                      # Complete project documentation
└── models/
    ├── metadata.json              # Feature bounds, drug catalog, and dataset metadata
    ├── scaler.pkl                 # Fitted feature StandardScaler
    ├── best_model_particle_size.pkl
    ├── best_model_EE.pkl
    ├── best_model_LC.pkl
    ├── evaluation_metrics.json    # Complete benchmark scores across all algorithms
    └── feature_importances.json   # Parameter impact rankings
```

---

## 📈 Trained Model Benchmark Summary

| Target Property | Winning ML Algorithm | 5-Fold CV $R^2$ | Mean Absolute Error (MAE) |
|---|---|---|---|
| **Particle Size (nm)** | **Extra Trees / SVR** | **0.4201** | $\pm 38.35\text{ nm}$ |
| **Encapsulation Efficiency (EE%)** | **Extra Trees Regressor** | **0.5265** | $\pm 11.95\%$ |
| **Loading Capacity (LC%)** | **Support Vector Regressor (SVR)** | **0.6464** | $\pm 3.42\%$ |

---

## 🧪 Wet-Lab Formulation Recipe Generation Example
When a researcher targets:
- **Target Particle Size = 150 nm**
- **Target Encapsulation Efficiency = 80%**
- **Active Drug = Curcumin / Lipophilic API (MW 372.6, LogP 3.81)**

The inverse engine produces:
- **PLGA MW**: 12.0 kDa (PLGA 50:50)
- **Drug/Polymer ratio**: 0.30 w/w
- **Surfactant**: 0.50% w/v Pluronic F68 (HLB 20.5)
- **Solvent**: Acetone (Polarity 5.1), $V_{aq}/V_{org} = 2.86$
- **Predicted Properties**: Particle Size = **156.5 nm** (4.3% error) | EE = **71.6%** | LC = **11.48%** | **Match: 92.6%**
