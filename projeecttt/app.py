"""
PLGA NanoPredict: Machine Learning Platform for Inverse Nanoparticle Formulation
B.Tech Biotechnology Minor Project
Trained on Literature Benchmark: plga_dataset.csv (Nature Scientific Data: 10.1038/s41597-025-05520-9)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from inverse_optimizer import NORMALIZED_DEFAULT_DRUG

# Page configuration
st.set_page_config(
    page_title="NanoPredict | PLGA Inverse Formulation",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lumix AI Ember & Obsidian Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Obsidian pitch-black base with warm radiant ember glow */
    .stApp {
        background: radial-gradient(circle at 65% -8%, rgba(249, 115, 22, 0.26) 0%, rgba(220, 38, 38, 0.10) 28%, rgba(9, 9, 13, 0.98) 60%), #09090d !important;
        background-attachment: fixed !important;
        color: #f8fafc;
    }
    
    /* Sidebar: Clean charcoal-obsidian */
    [data-testid="stSidebar"] {
        background-color: #101015 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }
    
    /* Hero Header */
    .hero-container {
        text-align: center;
        padding: 22px 10px 26px 10px;
        margin-bottom: 22px;
        position: relative;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(249, 115, 22, 0.08);
        border: 1px solid rgba(249, 115, 22, 0.35);
        color: #fdba74;
        padding: 5px 16px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 14px;
        box-shadow: 0 0 16px rgba(249, 115, 22, 0.15);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin: 0 auto 10px auto;
        line-height: 1.18;
        text-align: center !important;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #94a3b8;
        max-width: 720px;
        margin-left: auto !important;
        margin-right: auto !important;
        line-height: 1.55;
        text-align: center !important;
        display: block !important;
    }
    
    /* Lumix-style obsidian cards with glowing ember top stroke */
    .ember-card {
        background: #131319;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-top: 1.5px solid rgba(249, 115, 22, 0.65);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), 0 -2px 16px rgba(249, 115, 22, 0.08);
    }
    
    /* Radiant ember gradient button */
    .stButton>button {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 60%, #fb923c 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 0 22px rgba(249, 115, 22, 0.45) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        box-shadow: 0 0 35px rgba(249, 115, 22, 0.8) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: #181820 !important;
        color: #fdba74 !important;
        border: 1px solid rgba(249, 115, 22, 0.28) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton>button:hover {
        background: rgba(249, 115, 22, 0.15) !important;
        border-color: #f97316 !important;
        color: #ffffff !important;
    }
    
    /* Dark Inputs */
    div[data-baseweb="input"] {
        background-color: #15151c !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #f97316 !important;
        box-shadow: 0 0 10px rgba(249, 115, 22, 0.3) !important;
    }
    div[data-baseweb="select"] {
        background-color: #15151c !important;
        border-radius: 8px !important;
    }
    
    /* Streamlit Metric boxes */
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: #f8fafc !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.84rem !important;
    }
    
    /* Radio buttons selection color */
    div[role="radiogroup"] label {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Absolute base path resolution for local and cloud deployment (Streamlit Community Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "plga_dataset.csv")

# Resource loaders
@st.cache_resource
def load_designer():
    from inverse_optimizer import PLGANanoparticleInverseDesigner
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(meta_path) and os.path.exists(scaler_path):
        return PLGANanoparticleInverseDesigner(MODELS_DIR, DATASET_PATH)
    return None

@st.cache_data
def load_metadata():
    p = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None

def load_evaluation_metrics():
    p = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None

def load_feature_importances():
    p = os.path.join(MODELS_DIR, "feature_importances.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None

def load_hyperparameters():
    p = os.path.join(MODELS_DIR, "hyperparameters_summary.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None

def load_plga_dataset():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        unwanted = [
            'data_origin', 'source_dataset_doi', 'source_dataset_url',
            'source_data_descriptor_doi', 'source_data_descriptor_url',
            'original_research_paper_url', 'provenance_note'
        ]
        df = df.drop(columns=[c for c in unwanted if c in df.columns], errors='ignore')
        df = df.loc[:, ~df.columns.str.contains('url|doi|provenance|origin', case=False)]
        return df
    return None

# Sidebar branding & navigation
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 11px; padding: 6px 0 18px 0;">
    <div style="width: 34px; height: 34px; border-radius: 9px; background: linear-gradient(135deg, #f97316, #ea580c); display: flex; align-items: center; justify-content: center; font-weight: 800; color: white; font-size: 15px; box-shadow: 0 0 14px rgba(249, 115, 22, 0.45);">NP</div>
    <div>
        <div style="font-size: 1.16rem; font-weight: 700; color: #ffffff; letter-spacing: -0.01em;">NanoPredict</div>
        <div style="font-size: 0.74rem; color: #94a3b8;">B.Tech Biotechnology</div>
    </div>
</div>
""", unsafe_allow_html=True)

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "Inverse Formulation Generator",
        "Model Benchmarking & Diagnostics",
        "PLGA Literature Dataset",
        "Wet-Lab Protocol SOP"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Polymer:** Poly(lactic-co-glycolic acid) (PLGA)")
st.sidebar.markdown("**Benchmark:** *Nature Scientific Data* ([10.1038/s41597-025-05520-9](https://doi.org/10.1038/s41597-025-05520-9))")
st.sidebar.markdown("---")

designer = load_designer()
metadata = load_metadata()

# ==================== MODULE 1: INVERSE FORMULATION GENERATOR ====================
if nav_choice == "Inverse Formulation Generator":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">B.Tech Biotechnology Minor Project | AI-Driven Nanoparticle Design</div>
        <h1 class="hero-title">Inverse Nanoparticle Formulation</h1>
        <p class="hero-subtitle" style="text-align: center;">Accelerate PLGA drug delivery research with multi-model machine learning. Predict exact laboratory formulation recipes from target particle size and encapsulation efficiency.</p>
    </div>
    """, unsafe_allow_html=True)

    if designer is None or metadata is None:
        st.warning("Machine learning models are not initialized. Please train the models.")
        if st.button("Initialize & Train Models", icon=":material/play_arrow:"):
            with st.spinner("Training models on benchmark dataset..."):
                from train_models import train_and_evaluate
                train_and_evaluate()
                st.success("Training complete.")
                st.rerun()
    else:
        st.markdown('<div style="text-align: center; margin-bottom: 1.2rem;"><h5 style="color: #f8fafc; font-weight: 600;">Target Nanoparticle Specifications</h5></div>', unsafe_allow_html=True)
        
        c_sz1, c_sz2, c_sz3 = st.columns([1.2, 1.2, 1.0])
        with c_sz1:
            target_size = st.number_input(
                "Target Particle Size (nm)",
                min_value=40.0,
                max_value=500.0,
                value=150.0,
                step=1.0,
                format="%.1f",
                help="Hydrodynamic diameter of nanoparticles measured by DLS"
            )
            st.caption("Optimal biological window: 120 - 200 nm (EPR tumor targeting)")

        with c_sz2:
            target_ee = st.number_input(
                "Target Encapsulation Efficiency (% EE)",
                min_value=10.0,
                max_value=99.0,
                value=80.0,
                step=0.5,
                format="%.1f",
                help="Percentage of active therapeutic agent entrapped in the polymer core"
            )
            st.caption("Standard formulation target: 60 - 85% EE")

        with c_sz3:
            num_candidates = st.radio(
                "Candidate Recipes to Generate:",
                [1, 3, 5],
                index=1,
                horizontal=True,
                help="Generates alternative ranked formulation recipes"
            )

        active_drug_props = NORMALIZED_DEFAULT_DRUG
        active_drug_label = "Normalized Reference Drug (MW 297.6, LogP 3.10)"

        search_btn = st.button("Compute Formulation Candidates", icon=":material/play_arrow:", use_container_width=True)

        if search_btn:
            with st.spinner("Searching multi-dimensional PLGA formulation space..."):
                candidates = designer.search_candidates(
                    target_size_nm=target_size,
                    target_ee_pct=target_ee,
                    drug_properties=active_drug_props,
                    num_candidates=num_candidates
                )

            st.success(f"Synthesized {len(candidates)} candidate formulation(s) matching your parameters.")
            st.markdown('<div style="text-align: center; margin-top: 1.5rem; margin-bottom: 1rem;"><h4 style="color: #f8fafc; font-weight: 600;">Candidate Formulations</h4></div>', unsafe_allow_html=True)

            tabs = st.tabs([f"Candidate {i+1} | Match {c['match_score']}%" for i, c in enumerate(candidates)])

            for i, (tab, cand) in enumerate(zip(tabs, candidates)):
                with tab:
                    c1, c2, c3 = st.columns([1.3, 0.9, 1.1])

                    with c1:
                        st.markdown("##### Formulation Recipe")
                        f = cand["formulation"]
                        
                        laga_str = "PLGA 50:50" if abs(f['LA/GA'] - 1.0) < 0.3 else ("PLGA 75:25" if abs(f['LA/GA'] - 3.0) < 0.5 else f"PLGA {f['LA/GA']:.2f}")
                        surf_str = "PVA (Polyvinyl alcohol)" if abs(f['surfactant_HLB'] - 18.0) < 1.5 else ("Pluronic F68 / Poloxamer 188" if abs(f['surfactant_HLB'] - 20.5) < 1.5 else f"Surfactant (HLB {f['surfactant_HLB']:.1f})")
                        solv_str = "Acetone" if abs(f['solvent_polarity_index'] - 5.1) < 0.3 else ("DCM / Ethyl Acetate" if abs(f['solvent_polarity_index'] - 4.0) < 0.4 else "DMSO / Acetonitrile")

                        recipe_table = pd.DataFrame([
                            {"Parameter": "Polymer Molecular Weight", "Value": f"{f['polymer_MW']:.1f} kDa", "Role": "Polymer matrix chain length"},
                            {"Parameter": "Lactide/Glycolide Ratio", "Value": f"{f['LA/GA']:.2f}", "Role": laga_str},
                            {"Parameter": "Initial Drug/Polymer Ratio", "Value": f"{f['drug/polymer']:.3f} w/w", "Role": f"{f['drug/polymer']*100:.1f} mg drug per 100 mg PLGA"},
                            {"Parameter": "Surfactant Concentration", "Value": f"{f['surfactant_concentration']:.2f} % w/v", "Role": "Aqueous phase stabilizer"},
                            {"Parameter": "Surfactant Reagent", "Value": f"{f['surfactant_HLB']:.1f} HLB", "Role": surf_str},
                            {"Parameter": "Aqueous / Organic Ratio", "Value": f"{f['aqueous/organic']:.2f} v/v", "Role": "Volume ratio (Water : Solvent)"},
                            {"Parameter": "Aqueous Phase pH Code", "Value": f"{f['pH']:.1f}", "Role": "Neutral (0) or Acidic (-1)"},
                            {"Parameter": "Organic Solvent", "Value": f"{f['solvent_polarity_index']:.2f} Polarity", "Role": solv_str}
                        ])
                        st.dataframe(recipe_table, hide_index=True, use_container_width=True)

                    with c2:
                        st.markdown("##### Prediction vs Target")
                        st.metric(
                            "Predicted Size",
                            f"{cand['predicted_size_nm']} nm",
                            f"{cand['predicted_size_nm'] - target_size:+.1f} nm ({cand['size_error_pct']}%)",
                            delta_color="inverse" if cand['size_error_pct'] > 10 else "normal"
                        )
                        st.metric(
                            "Predicted EE",
                            f"{cand['predicted_ee_pct']}%",
                            f"{cand['predicted_ee_pct'] - target_ee:+.1f}% ({cand['ee_error_pct']}%)",
                            delta_color="inverse" if cand['ee_error_pct'] > 10 else "normal"
                        )
                        st.metric("Predicted Loading Capacity (LC)", f"{cand['predicted_lc_pct']}%")
                        st.metric("Overall Match Score", f"{cand['match_score']}%")

                    with c3:
                        st.markdown("##### Radar Comparison")
                        categories = ['Particle Size (nm)', 'EE (%)', 'LC (%) x 5']
                        target_norm = [target_size, target_ee, (target_ee * f['drug/polymer']) * 5]
                        pred_norm = [cand['predicted_size_nm'], cand['predicted_ee_pct'], cand['predicted_lc_pct'] * 5]

                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=target_norm,
                            theta=categories,
                            fill='toself',
                            name='Target Specs',
                            line=dict(color='#fdba74', dash='dash'),
                            fillcolor='rgba(253, 186, 116, 0.15)'
                        ))
                        fig.add_trace(go.Scatterpolar(
                            r=pred_norm,
                            theta=categories,
                            fill='toself',
                            name='AI Prediction',
                            line=dict(color='#f97316'),
                            fillcolor='rgba(249, 115, 22, 0.28)'
                        ))
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, gridcolor='rgba(249, 115, 22, 0.18)'),
                                angularaxis=dict(gridcolor='rgba(249, 115, 22, 0.18)')
                            ),
                            showlegend=True,
                            margin=dict(l=20, r=20, t=30, b=20),
                            height=270,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color="#f8fafc"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    export_text = (
                        f"=== PLGA NANOPARTICLE BENCH SYNTHESIS SHEET (Candidate #{i+1}) ===\n"
                        f"Drug Baseline: {active_drug_label}\n"
                        f"Target Specifications: Particle Size = {target_size} nm | Encapsulation Efficiency = {target_ee}%\n"
                        f"----------------------------------------------------------------------\n"
                        f"1. PLGA Polymer Molecular Weight: {f['polymer_MW']:.1f} kDa ({laga_str})\n"
                        f"2. Drug-to-Polymer Mass Ratio: {f['drug/polymer']:.3f} w/w\n"
                        f"3. Organic Solvent: {solv_str} (Polarity: {f['solvent_polarity_index']:.2f})\n"
                        f"4. Surfactant: {f['surfactant_concentration']:.2f}% w/v {surf_str}\n"
                        f"5. Aqueous-to-Organic Phase Ratio: {f['aqueous/organic']:.2f} v/v\n"
                        f"6. Aqueous Phase pH Level: {f['pH']:.1f}\n"
                        f"----------------------------------------------------------------------\n"
                        f"Predicted Particle Size: {cand['predicted_size_nm']} nm\n"
                        f"Predicted Encapsulation Efficiency: {cand['predicted_ee_pct']}%\n"
                        f"Predicted Drug Loading Capacity: {cand['predicted_lc_pct']}%\n"
                        f"Overall Formulation Match Score: {cand['match_score']}%\n"
                    )
                    st.download_button(
                        label=f"Download Bench Recipe Sheet #{i+1}",
                        icon=":material/download:",
                        data=export_text,
                        file_name=f"plga_formulation_recipe_{i+1}.txt",
                        mime="text/plain"
                    )

# ==================== MODULE 2: MODEL BENCHMARKING ====================
elif nav_choice == "Model Benchmarking & Diagnostics":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">Model Diagnostics & Statistical Verification</div>
        <h1 class="hero-title">Algorithm Performance & Diagnostics</h1>
        <p class="hero-subtitle" style="text-align: center;">Cross-validated statistical evaluation of machine learning regression architectures benchmarked on literature data.</p>
    </div>
    """, unsafe_allow_html=True)

    metrics = load_evaluation_metrics()

    if metrics is None:
        st.warning("Benchmark metrics not found. Train the models to populate.")
    else:
        st.markdown('<div style="text-align: center; margin-bottom: 1.2rem;"><h5 style="color: #f8fafc; font-weight: 600;">Algorithm Performance Comparison</h5></div>', unsafe_allow_html=True)
        t_size, t_ee = st.tabs(["Particle Size (nm)", "Encapsulation Efficiency (EE%)"])

        for t_name, tab in zip(["particle_size", "EE"], [t_size, t_ee]):
            with tab:
                m_df = pd.DataFrame(metrics[t_name]).T
                m_df = m_df[["R2_Test", "MAE", "RMSE"]].rename(columns={
                    "R2_Test": "Test R²",
                    "MAE": "MAE",
                    "RMSE": "RMSE"
                })
                st.dataframe(
                    m_df.style.highlight_max(subset=["Test R²"], color="#431407")
                              .highlight_min(subset=["MAE"], color="#431407"),
                    use_container_width=True
                )

# ==================== MODULE 3: PLGA LITERATURE DATASET ====================
elif nav_choice == "PLGA Literature Dataset":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">Curated Literature Ground Truth</div>
        <h1 class="hero-title">PLGA Benchmark Dataset</h1>
        <p class="hero-subtitle" style="text-align: center;">100 experimental formulations curated from peer-reviewed literature with 15 physicochemical descriptors and 3 measured properties.</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_plga_dataset()
    if df is not None:
        st.markdown(f'<div style="text-align: center; margin-bottom: 1.2rem; color: #cbd5e1;"><strong>Total Samples:</strong> <code style="color: #fdba74;">{len(df)}</code> &nbsp;|&nbsp; <strong>Formulation Features:</strong> 8 &nbsp;|&nbsp; <strong>Drug Descriptors:</strong> 7 &nbsp;|&nbsp; <strong>Target Properties:</strong> 3</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.download_button(
            label="Download Cleaned Dataset (CSV)",
            icon=":material/download:",
            data=df.to_csv(index=False),
            file_name="cleaned_plga_dataset.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.markdown('<div style="text-align: center; margin-bottom: 1.2rem;"><h5 style="color: #f8fafc; font-weight: 600;">Pearson Correlation Heatmap</h5></div>', unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        corr = df[num_cols].corr()
        fig_c = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Oranges",
            title="Correlation Matrix of PLGA Formulation Variables and Characterization Targets"
        )
        fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e2e8f0")
        st.plotly_chart(fig_c, use_container_width=True)

# ==================== MODULE 4: WET-LAB PROTOCOL ====================
elif nav_choice == "Wet-Lab Protocol SOP":
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Experimental Bench Protocol</h1>
    </div>
    """, unsafe_allow_html=True)
