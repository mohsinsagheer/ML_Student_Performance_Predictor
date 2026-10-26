"""
Student Performance Prediction System - Streamlit Dashboard
===========================================================
Interactive machine learning dashboard implementing:
- Decision Tree Classifier
- Naïve Bayes Classifier
- K-Nearest Neighbors (KNN) Classifier
- Majority-Vote Ensemble Classifier

Features:
- Navy blue gradient sidebar with high-contrast text and inputs
- Header hero banner with navy blue gradient
- Pure white main background (in Light Mode) with full Dark Mode support
- Dynamic theme switching
- EDA, data preprocessing, memory profiling, live student prediction, and batch prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

from src.data_loader import (
    load_dataset,
    handle_missing_values,
    create_target_variable,
    get_memory_usage,
    optimize_memory
)
from src.preprocessor import StudentDataPreprocessor
from src.models import StudentModelManager
from src.utils import (
    MODEL_DESCRIPTIONS,
    SAMPLE_STUDENT_PROFILES,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_grade_distribution,
    plot_study_vs_outcome
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Insight | Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Sidebar: Theme Toggle, Dataset, Feature Scope & Hyperparameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/education.png", width=64)
    st.markdown("<h2 style='color: #FFFFFF; margin-top: -10px; margin-bottom: 15px;'>Student Insight</h2>", unsafe_allow_html=True)
    
    # 🌓 Light & Dark Mode Toggle
    st.markdown("<p style='color: #E2E8F0; font-weight: 700; margin-bottom: 4px; font-size: 0.9rem;'>🌓 DISPLAY THEME</p>", unsafe_allow_html=True)
    theme_choice = st.radio(
        "Display Theme",
        options=["☀️ Light Mode", "🌙 Dark Mode"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    is_dark = (theme_choice == "🌙 Dark Mode")
    
    st.markdown("<hr style='margin: 15px 0; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E2E8F0; font-weight: 700; margin-bottom: 6px; font-size: 0.95rem;'>📁 DATASET SELECTION</p>", unsafe_allow_html=True)
    dataset_choice = st.selectbox(
        "Choose Subject Dataset",
        options=["math", "portuguese", "combined"],
        format_func=lambda x: {
            "math": "Mathematics (student-mat.csv)",
            "portuguese": "Portuguese (student-por.csv)",
            "combined": "Combined Cohort (Both Courses)"
        }[x],
        index=0
    )
    
    st.markdown("<hr style='margin: 15px 0; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E2E8F0; font-weight: 700; margin-bottom: 6px; font-size: 0.95rem;'>⚙️ FEATURE CONFIGURATION</p>", unsafe_allow_html=True)
    include_grades = st.checkbox(
        "Include Period Grades (G1, G2)",
        value=True,
        help="When checked, models use first and second period grades (G1, G2) as predictive features. When unchecked, models rely solely on behavioral & demographic factors."
    )
    
    test_size = st.slider(
        "Test Split Ratio",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
        help="Proportion of the dataset reserved for testing model generalization."
    )
    
    random_seed = st.number_input(
        "Random State Seed",
        value=42,
        step=1,
        help="Ensures reproducible train/test splits and model initialization."
    )
    
    st.markdown("<hr style='margin: 15px 0; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E2E8F0; font-weight: 700; margin-bottom: 6px; font-size: 0.95rem;'>🎛️ MODEL HYPERPARAMETERS</p>", unsafe_allow_html=True)
    
    with st.expander("🌲 Decision Tree Settings", expanded=False):
        dt_criterion = st.selectbox("Split Criterion", ["gini", "entropy", "log_loss"], index=0)
        dt_max_depth = st.slider("Max Depth", min_value=2, max_value=20, value=5, help="Controls tree depth to prevent overfitting.")
        dt_min_samples = st.slider("Min Samples Split", min_value=2, max_value=10, value=2)
        
    with st.expander("📊 Naïve Bayes Settings", expanded=False):
        nb_var_smoothing = st.select_slider(
            "Variance Smoothing (var_smoothing)",
            options=[1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            value=1e-9,
            help="Calculation stabilizer portion added to feature variances."
        )
        
    with st.expander("📍 K-Nearest Neighbors Settings", expanded=False):
        knn_k = st.slider("Number of Neighbors (k)", min_value=1, max_value=25, value=5, step=2)
        knn_weights = st.selectbox("Weights", ["uniform", "distance"], index=0)
        knn_metric = st.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan"], index=0)

    st.markdown("<hr style='margin: 15px 0; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.markdown("<div style='color: #CBD5E1; font-size: 0.8rem; text-align: center;'>ML Foundations Lab Project</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Dynamic CSS Injection based on Theme (Light vs Dark)
# -----------------------------------------------------------------------------
if is_dark:
    # DARK MODE STYLES
    body_bg = "#0B0F19"
    content_bg = "#0F172A"
    text_color = "#F8FAFC"
    text_muted = "#94A3B8"
    card_bg = "linear-gradient(180deg, #1E293B 0%, #0F172A 100%)"
    card_border = "#334155"
    tab_active_bg = "rgba(99, 102, 241, 0.15)"
    tab_active_color = "#818CF8"
    tab_border = "#6366F1"
    maj_card_bg = "linear-gradient(180deg, #1E1B4B 0%, #0F172A 100%)"
    maj_card_border = "#6366F1"
    input_bg = "#1E293B"
    input_border = "#475569"
    form_bg = "#111827"
    form_border = "#334155"
    popover_bg = "#1E293B"
    popover_border = "#818CF8"
    popover_hover = "#312E81"
    popover_hover_text = "#C7D2FE"
    popover_text = "#F8FAFC"
    slider_thumb = "#818CF8"
else:
    # LIGHT MODE STYLES (Clean White Background as requested)
    body_bg = "#FFFFFF"
    content_bg = "#FFFFFF"
    text_color = "#0F172A"
    text_muted = "#475569"
    card_bg = "#FFFFFF"
    card_border = "#E2E8F0"
    tab_active_bg = "rgba(7, 25, 82, 0.08)"
    tab_active_color = "#071952"
    tab_border = "#071952"
    maj_card_bg = "linear-gradient(180deg, #F0F9FF 0%, #FFFFFF 100%)"
    maj_card_border = "#0284C7"
    input_bg = "#FFFFFF"
    input_border = "#CBD5E1"
    form_bg = "#F8FAFC"
    form_border = "#E2E8F0"
    popover_bg = "#FFFFFF"
    popover_border = "#1E3A8A"
    popover_hover = "#EFF6FF"
    popover_hover_text = "#1E3A8A"
    popover_text = "#0F172A"
    slider_thumb = "#1E3A8A"

st.markdown(f"""
<style>
    /* ========================================================= */
    /* 1. GLOBAL & MAIN PAGE BACKGROUND                          */
    /* ========================================================= */
    .stApp {{
        background-color: {body_bg} !important;
        color: {text_color} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* All Main Content Headings in Dark & Light Mode */
    section.main h1, section.main h2, section.main h3, 
    section.main h4, section.main h5, section.main h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {{
        color: {text_color} !important;
        font-weight: 700 !important;
    }}

    /* All Main Content Paragraphs, Spans & Lists */
    section.main p, 
    section.main span:not(.pass-badge):not(.fail-badge), 
    section.main li, 
    div.block-container p {{
        color: {text_color} !important;
    }}

    /* Captions & Subtitles */
    [data-testid="stCaptionContainer"] p, .stCaption p, .stCaption {{
        color: {text_muted} !important;
    }}

    /* Form Widget Labels on Main Page */
    [data-testid="stWidgetLabel"] p, label p, label span {{
        color: {text_color} !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }}

    /* ========================================================= */
    /* 2. FORMS & PREDICTOR SECTION (FIX VISIBILITY IN DARK MODE)*/
    /* ========================================================= */
    div[data-testid="stForm"] {{
        background-color: {form_bg} !important;
        border: 1.5px solid {form_border} !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.1) !important;
    }}

    /* Main Page & Form: Selectbox outer container */
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        border: 1.5px solid {input_border} !important;
        border-radius: 8px !important;
        color: {text_color} !important;
    }}

    /* ---- Selected value text: all known BaseWeb/Streamlit class patterns ---- */
    /* ValueContainer, SingleValue, Placeholder, ControlComponent children */
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"] span,
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"] div,
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="singleValue"],
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="placeholder"],
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] div[aria-selected="true"],
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div,
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] span,
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] p,
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"] span,
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"] div,
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="singleValue"],
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="placeholder"],
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] div[aria-selected="true"],
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div,
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] span,
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] p {{
        color: {text_color} !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
        -webkit-text-fill-color: {text_color} !important;
    }}

    /* select_slider also needs text fixes */
    section.main div[data-testid="stSelectSlider"] [data-testid="stThumbValue"],
    div[data-testid="stForm"] div[data-testid="stSelectSlider"] [data-testid="stThumbValue"] {{
        color: {text_color} !important;
    }}

    /* Native <select> fallback (if Streamlit renders one) */
    section.main select,
    div[data-testid="stForm"] select {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        border: 1.5px solid {input_border} !important;
        border-radius: 8px !important;
    }}
    section.main select option,
    div[data-testid="stForm"] select option {{
        background-color: {popover_bg} !important;
        color: {popover_text} !important;
    }}

    /* Search/hidden input inside SelectBox */
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] input,
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] input {{
        background: transparent !important;
        border: none !important;
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
    }}

    /* Selectbox Arrow Icon */
    section.main div[data-testid="stSelectbox"] [data-baseweb="select"] svg,
    div[data-testid="stForm"] div[data-testid="stSelectbox"] [data-baseweb="select"] svg {{
        fill: {text_muted} !important;
        color: {text_muted} !important;
    }}

    /* Dropdown Popover / Menu Options List (Global) */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    ul[data-baseweb="menu"],
    ul[role="listbox"] {{
        background-color: {popover_bg} !important;
        border: 1.5px solid {popover_border} !important;
        border-radius: 8px !important;
        box-shadow: 0 12px 30px rgba(0,0,0,0.5) !important;
    }}
    ul[data-baseweb="menu"] li,
    ul[data-baseweb="menu"] li span,
    ul[data-baseweb="menu"] li div,
    li[role="option"] {{
        background-color: {popover_bg} !important;
        color: {popover_text} !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }}
    ul[data-baseweb="menu"] li:hover,
    li[role="option"]:hover,
    li[aria-selected="true"] {{
        background-color: {popover_hover} !important;
        color: {popover_hover_text} !important;
    }}

    /* Main Page Sliders */
    section.main [data-testid="stSlider"] div[data-testid="stSliderThumbValue"] {{
        color: {slider_thumb} !important;
        font-weight: 800 !important;
    }}
    section.main [data-testid="stSlider"] div[data-testid="stSliderTickBarMin"],
    section.main [data-testid="stSlider"] div[data-testid="stSliderTickBarMax"] {{
        color: {text_muted} !important;
    }}

    /* Expanders on Main Page */
    section.main [data-testid="stExpander"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 10px !important;
    }}
    section.main [data-testid="stExpander"] summary,
    section.main [data-testid="stExpander"] summary p {{
        color: {text_color} !important;
        font-weight: 700 !important;
    }}

    /* File Uploader on Main Page */
    section.main [data-testid="stFileUploader"] {{
        background-color: {card_bg} !important;
        border: 1.5px dashed {card_border} !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }}
    section.main [data-testid="stFileUploader"] * {{
        color: {text_color} !important;
    }}

    /* JSON Viewer on Main Page */
    section.main div[data-testid="stJson"] {{
        background-color: {input_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }}
    section.main div[data-testid="stJson"] * {{
        color: #38BDF8 !important;
    }}

    /* ========================================================= */
    /* 3. SIDEBAR: NAVY BLUE GRADIENT & CRYSTAL-CLEAR CONTROLS   */
    /* ========================================================= */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #071952 0%, #0B2447 50%, #081B3A 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.15) !important;
    }}
    
    /* Sidebar Headers, Labels & Descriptions */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] label span {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }}
    
    /* Sidebar Selectbox Container */
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
        background-color: #0F172A !important;
        border: 1.5px solid #38BDF8 !important;
        border-radius: 8px !important;
    }}
    /* Sidebar Selected Option Text */
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] div[aria-selected="true"],
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] div[class*="singleValue"],
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] span {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] input {{
        background: transparent !important;
        border: none !important;
        color: transparent !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] svg {{
        fill: #38BDF8 !important;
        color: #38BDF8 !important;
    }}

    /* Sidebar Checkbox */
    [data-testid="stSidebar"] [data-testid="stCheckbox"] {{
        background-color: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }}
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label span {{
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}

    /* Sidebar Sliders */
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stSliderThumbValue"] {{
        color: #38BDF8 !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stSliderTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stSliderTickBarMax"],
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div {{
        color: #CBD5E1 !important;
        font-weight: 600 !important;
    }}

    /* Sidebar Number Input */
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
        background-color: #0F172A !important;
        border: 1.5px solid #38BDF8 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 8px !important;
    }}
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {{
        background-color: #1E3A8A !important;
        border: 1px solid #38BDF8 !important;
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {{
        fill: #FFFFFF !important;
    }}

    /* Sidebar Expanders */
    [data-testid="stSidebar"] [data-testid="stExpander"] {{
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1.5px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {{
        fill: #38BDF8 !important;
    }}
    
    /* Sidebar Theme Radio */
    [data-testid="stSidebar"] [data-testid="stRadio"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label span {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}

    /* ========================================================= */
    /* 4. HEADER HERO BANNER                                     */
    /* ========================================================= */
    .header-hero {{
        background: linear-gradient(135deg, #071952 0%, #0B2447 50%, #19376D 100%);
        border-radius: 16px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(7, 25, 82, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }}
    .header-hero .hero-title {{
        font-size: 2.35rem;
        font-weight: 800;
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
        margin: 0 0 0.5rem 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }}
    .header-hero .hero-subtitle {{
        font-size: 1.05rem;
        color: #E2E8F0 !important;
        margin: 0;
        line-height: 1.6;
        font-weight: 400;
    }}
    
    /* ========================================================= */
    /* 5. METRIC CARDS & CONTAINERS                              */
    /* ========================================================= */
    .metric-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 4px 15px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -4px rgba(7, 25, 82, 0.15);
    }}
    .metric-card h3 {{
        margin: 0;
        font-size: 0.82rem;
        color: {text_muted} !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
    }}
    .metric-card p {{
        margin: 0.45rem 0 0 0;
        font-size: 1.85rem;
        font-weight: 800;
        color: {text_color} !important;
    }}
    
    /* Majority Vote Card Highlight */
    .majority-card {{
        background: {maj_card_bg} !important;
        border: 2px solid {maj_card_border} !important;
        box-shadow: 0 0 20px rgba(11, 36, 71, 0.2) !important;
    }}
    
    /* Status Badges */
    .pass-badge {{
        display: inline-block;
        padding: 0.35rem 0.95rem;
        border-radius: 9999px;
        background-color: rgba(16, 185, 129, 0.16);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #10B981 !important;
        font-weight: 800;
        font-size: 1.15rem;
    }}
    .fail-badge {{
        display: inline-block;
        padding: 0.35rem 0.95rem;
        border-radius: 9999px;
        background-color: rgba(239, 68, 68, 0.16);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #EF4444 !important;
        font-weight: 800;
        font-size: 1.15rem;
    }}
    
    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 2px solid {card_border};
        padding-bottom: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: {text_muted} !important;
        border: none;
        background: transparent;
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        color: {tab_active_color} !important;
        background-color: {tab_active_bg} !important;
        border-bottom: 3px solid {tab_border} !important;
    }}
    
    /* Primary Action Buttons */
    div.stButton > button:first-child {{
        background: linear-gradient(135deg, #071952 0%, #0B2447 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 14px 0 rgba(7, 25, 82, 0.3);
        transition: all 0.2s ease;
    }}
    div.stButton > button:first-child:hover {{
        background: linear-gradient(135deg, #0B2447 0%, #19376D 100%) !important;
        box-shadow: 0 6px 20px 0 rgba(7, 25, 82, 0.45);
        transform: translateY(-1px);
    }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loading and Pipeline Execution (Cached for Performance)
# -----------------------------------------------------------------------------
@st.cache_data
def get_processed_data(dataset_type: str, include_period: bool, split_ratio: float, seed: int):
    raw_df = load_dataset(dataset_type)
    cleaned_df, missing_summary = handle_missing_values(raw_df)
    labeled_df = create_target_variable(cleaned_df)
    
    preprocessor = StudentDataPreprocessor(include_period_grades=include_period)
    X, y, X_df = preprocessor.fit_transform(labeled_df)
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X, y, test_size=split_ratio, random_state=seed, stratify=True
    )
    
    return labeled_df, missing_summary, preprocessor, X, y, X_df, X_train, X_test, y_train, y_test


labeled_df, missing_summary, preprocessor, X, y, X_df, X_train, X_test, y_train, y_test = get_processed_data(
    dataset_choice, include_grades, test_size, random_seed
)

# Train Models Manager
model_manager = StudentModelManager()
model_manager.train_all_models(
    X_train, y_train,
    dt_params={"criterion": dt_criterion, "max_depth": dt_max_depth, "min_samples_split": dt_min_samples, "random_state": random_seed},
    nb_params={"var_smoothing": nb_var_smoothing},
    knn_params={"n_neighbors": knn_k, "weights": knn_weights, "metric": knn_metric},
    feature_names=preprocessor.feature_names_out
)
comparison_df, evals = model_manager.evaluate_all(X_test, y_test)
feature_importances_df = model_manager.get_feature_importances()


# -----------------------------------------------------------------------------
# Main Application Header Section: Navy Blue Gradient Hero Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-hero">
    <div class="hero-title">🎓 Student Performance Prediction System</div>
    <div class="hero-subtitle">
        An interactive machine learning dashboard comparing <b>Decision Tree</b>, <b>Naïve Bayes</b>, 
        and <b>K-Nearest Neighbors (KNN)</b> classifiers with majority-voting consensus.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Top KPI Metric Cards (Clean White in Light Mode, Dark Slate in Dark Mode)
# -----------------------------------------------------------------------------
total_students = len(labeled_df)
pass_count = (labeled_df["passed"] == 1).sum()
fail_count = (labeled_df["passed"] == 0).sum()
pass_rate = (pass_count / total_students) * 100
mem_stats = get_memory_usage(labeled_df)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(f'<div class="metric-card"><h3>Total Students</h3><p>{total_students}</p></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown(f'<div class="metric-card"><h3>Passed (≥ 10)</h3><p style="color:#10B981;">{pass_count}</p></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'<div class="metric-card"><h3>Failed (< 10)</h3><p style="color:#EF4444;">{fail_count}</p></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown(f'<div class="metric-card"><h3>Pass Rate</h3><p style="color:#2563EB;">{pass_rate:.1f}%</p></div>', unsafe_allow_html=True)
with kpi5:
    st.markdown(f'<div class="metric-card"><h3>Memory Footprint</h3><p>{mem_stats["total_kb"]:.1f} KB</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Tab Navigation
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset & Memory Profiling",
    "🧠 Model Evaluation & Comparison",
    "🎯 Live Student Predictor",
    "📁 Batch CSV Prediction",
    "📚 Educational Guide & Theory"
])


# -----------------------------------------------------------------------------
# TAB 1: Dataset Exploration & Memory Profiling
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("📋 Dataset Overview & Statistics")
    
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.write(f"Displaying **{labeled_df.shape[0]} rows** and **{labeled_df.shape[1]} columns**.")
    with col_t2:
        show_all_cols = st.checkbox("Show all columns in preview", value=False)
        
    preview_df = labeled_df if show_all_cols else labeled_df.head(10)
    st.dataframe(preview_df, use_container_width=True)
    
    st.markdown("#### 🔍 Summary Statistics & Missing Values")
    col_stat1, col_stat2 = st.columns([2, 1])
    
    with col_stat1:
        st.markdown("**Numeric Features Descriptive Statistics**")
        st.dataframe(labeled_df.describe().round(2), use_container_width=True)
        
    with col_stat2:
        st.markdown("**Missing Value Verification**")
        total_missing = sum(missing_summary.values())
        if total_missing == 0:
            st.success("✅ **0 Missing Values Detected!** Dataset is clean and ready for machine learning.")
        else:
            st.warning(f"⚠️ {total_missing} missing values imputed using robust statistics.")
        
        st.markdown("**Target Class Breakdown (Pass vs Fail)**")
        target_counts = labeled_df["pass_label"].value_counts().reset_index()
        target_counts.columns = ["Outcome", "Count"]
        fig_donut = px.pie(
            target_counts, 
            values="Count", 
            names="Outcome",
            hole=0.55,
            color="Outcome",
            color_discrete_map={"Pass": "#10B981", "Fail": "#EF4444"}
        )
        fig_donut.update_layout(
            height=240, 
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC" if is_dark else "#0F172A")
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")
    st.subheader("💾 Memory Usage & Optimization Analysis")
    st.markdown(
        "Demonstrates deep memory profiling in Python and Pandas. "
        "Explore how data types affect system RAM and how integer downcasting and categorical conversion optimize memory consumption."
    )
    
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        opt_df, mem_savings = optimize_memory(labeled_df)
        st.markdown(f"""
        - **Raw DataFrame Memory:** `{mem_savings['before_kb']} KB`
        - **Downcasted Optimized Memory:** `{mem_savings['after_kb']} KB`
        - **Memory Reduction Achieved:** `🔻 {mem_savings['reduction_pct']}%`
        """)
        st.progress(max(0.0, min(1.0, 1.0 - (mem_savings['after_kb'] / mem_savings['before_kb']))))
        
    with opt_col2:
        st.markdown("**Data Type Breakdown**")
        st.json(mem_stats["dtype_breakdown"])

    st.markdown("---")
    st.subheader("📈 Interactive Visualizations")
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.plotly_chart(plot_grade_distribution(labeled_df, is_dark=is_dark), use_container_width=True)
        
    with v_col2:
        st.plotly_chart(plot_study_vs_outcome(labeled_df, is_dark=is_dark), use_container_width=True)
        
    # Additional correlation exploration
    with st.expander("🔎 Explore Alcohol Consumption & Absences vs Academic Outcomes", expanded=False):
        fig_alc = px.scatter(
            labeled_df,
            x="absences",
            y="G3",
            color="pass_label",
            size="Walc",
            hover_data=["age", "studytime", "failures"],
            color_discrete_map={"Pass": "#10B981", "Fail": "#EF4444"},
            labels={"absences": "Number of School Absences", "G3": "Final Grade G3", "Walc": "Weekend Alcohol"},
            title="<b>Absences vs Final Grade (Point Size = Weekend Alcohol Consumption)</b>",
            template="plotly_dark" if is_dark else "plotly_white"
        )
        fig_alc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC" if is_dark else "#0F172A"),
            height=380
        )
        st.plotly_chart(fig_alc, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: Model Evaluation & Comparison
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🧠 Model Training & Performance Comparison")
    st.write(
        f"All three classifiers — **Decision Tree**, **Naïve Bayes**, and **K-Nearest Neighbors (KNN)** — were trained on "
        f"**{X_train.shape[0]} training instances** and tested on **{X_test.shape[0]} holdout test instances**."
    )
    
    # Model comparison table
    st.dataframe(
        comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Model Size (Bytes)"]],
        use_container_width=True,
        hide_index=True
    )
    
    # Comparison Bar Chart
    st.plotly_chart(plot_model_comparison(comparison_df, is_dark=is_dark), use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔲 Confusion Matrices")
    st.caption("Examine True Negatives (Correct Fails), False Positives, False Negatives, and True Positives (Correct Passes).")
    
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.plotly_chart(plot_confusion_matrix(evals["Decision Tree"]["confusion_matrix"], "Decision Tree", is_dark=is_dark), use_container_width=True)
        st.plotly_chart(plot_confusion_matrix(evals["K-Nearest Neighbors"]["confusion_matrix"], "K-Nearest Neighbors (KNN)", is_dark=is_dark), use_container_width=True)
        
    with cm_col2:
        st.plotly_chart(plot_confusion_matrix(evals["Naïve Bayes"]["confusion_matrix"], "Naïve Bayes", is_dark=is_dark), use_container_width=True)
        if "Majority Vote" in evals:
            st.plotly_chart(plot_confusion_matrix(evals["Majority Vote"]["confusion_matrix"], "Majority Vote Ensemble", is_dark=is_dark), use_container_width=True)

    # Feature Importance Section
    if feature_importances_df is not None and not feature_importances_df.empty:
        st.markdown("---")
        st.subheader("🌲 Decision Tree Feature Importance")
        st.markdown(
            "Identifies which features contributed most to splitting nodes and reducing impurity in the Decision Tree."
        )
        st.plotly_chart(plot_feature_importance(feature_importances_df, top_n=10, is_dark=is_dark), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: Live Single Student Predictor
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🎯 Real-Time Student Performance Predictor")
    st.markdown("Input student academic, demographic, and behavioral details to receive real-time predictions from all three models.")

    # Persona Preloader
    st.markdown("#### ⚡ Quick Load Preset Student Personas")
    persona_col1, persona_col2, persona_col3, persona_col4 = st.columns(4)
    
    if "student_input" not in st.session_state:
        st.session_state.student_input = SAMPLE_STUDENT_PROFILES["🌟 High Achieving Scholar"].copy()

    with persona_col1:
        if st.button("🌟 High Achieving Scholar", use_container_width=True):
            st.session_state.student_input = SAMPLE_STUDENT_PROFILES["🌟 High Achieving Scholar"].copy()
            st.rerun()
    with persona_col2:
        if st.button("⚠️ At-Risk Student", use_container_width=True):
            st.session_state.student_input = SAMPLE_STUDENT_PROFILES["⚠️ At-Risk / Struggling Student"].copy()
            st.rerun()
    with persona_col3:
        if st.button("📚 Average Hardworking", use_container_width=True):
            st.session_state.student_input = SAMPLE_STUDENT_PROFILES["📚 Average Hardworking Student"].copy()
            st.rerun()
    with persona_col4:
        if st.button("⚡ High Absences", use_container_width=True):
            st.session_state.student_input = SAMPLE_STUDENT_PROFILES["⚡ High Potential, High Absences"].copy()
            st.rerun()

    cur = st.session_state.student_input
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction Form with 3 Columns
    with st.form("student_prediction_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            st.markdown("##### 📝 Academic Performance")
            if include_grades:
                in_g1 = st.slider("First Period Grade (G1: 0-20)", 0, 20, int(cur.get("G1", 12)))
                in_g2 = st.slider("Second Period Grade (G2: 0-20)", 0, 20, int(cur.get("G2", 12)))
            else:
                in_g1 = 12
                in_g2 = 12
                st.info("Period grades (G1, G2) are disabled via sidebar settings.")
                
            in_studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                format_func=lambda x: {1: "< 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "> 10 hours"}[x],
                index=int(cur.get("studytime", 2)) - 1
            )
            in_failures = st.selectbox("Past Class Failures", options=[0, 1, 2, 3], index=int(cur.get("failures", 0)))
            in_schoolsup = st.selectbox("School Support", ["no", "yes"], index=0 if cur.get("schoolsup", "no") == "no" else 1)
            in_famsup = st.selectbox("Family Support", ["no", "yes"], index=0 if cur.get("famsup", "yes") == "no" else 1)
            in_paid = st.selectbox("Extra Paid Classes", ["no", "yes"], index=0 if cur.get("paid", "no") == "no" else 1)
            
        with col_f2:
            st.markdown("##### 👤 Demographics & Family")
            in_school = st.selectbox("School", ["GP", "MS"], format_func=lambda x: "Gabriel Pereira (GP)" if x == "GP" else "Mousinho da Silveira (MS)", index=0 if cur.get("school", "GP") == "GP" else 1)
            in_sex = st.selectbox("Sex", ["F", "M"], format_func=lambda x: "Female" if x == "F" else "Male", index=0 if cur.get("sex", "F") == "F" else 1)
            in_age = st.slider("Age", 15, 22, int(cur.get("age", 17)))
            in_address = st.selectbox("Home Address", ["U", "R"], format_func=lambda x: "Urban" if x == "U" else "Rural", index=0 if cur.get("address", "U") == "U" else 1)
            in_famsize = st.selectbox("Family Size", ["GT3", "LE3"], format_func=lambda x: "Greater than 3 (> 3)" if x == "GT3" else "Less or equal to 3 (≤ 3)", index=0 if cur.get("famsize", "GT3") == "GT3" else 1)
            in_pstatus = st.selectbox("Parents Cohabitation", ["T", "A"], format_func=lambda x: "Living Together (T)" if x == "T" else "Apart (A)", index=0 if cur.get("Pstatus", "T") == "T" else 1)
            in_medu = st.selectbox("Mother Education (Medu)", [0, 1, 2, 3, 4], format_func=lambda x: {0: "None", 1: "Primary (4th grade)", 2: "5th-9th grade", 3: "Secondary", 4: "Higher education"}[x], index=int(cur.get("Medu", 2)))
            in_fedu = st.selectbox("Father Education (Fedu)", [0, 1, 2, 3, 4], format_func=lambda x: {0: "None", 1: "Primary (4th grade)", 2: "5th-9th grade", 3: "Secondary", 4: "Higher education"}[x], index=int(cur.get("Fedu", 2)))
            in_mjob = st.selectbox("Mother Job", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(cur.get("Mjob", "other")))
            in_fjob = st.selectbox("Father Job", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(cur.get("Fjob", "other")))
            in_reason = st.selectbox("Reason to Choose School", ["home", "reputation", "course", "other"], index=["home", "reputation", "course", "other"].index(cur.get("reason", "course")))
            in_guardian = st.selectbox("Guardian", ["mother", "father", "other"], index=["mother", "father", "other"].index(cur.get("guardian", "mother")))

        with col_f3:
            st.markdown("##### 🌟 Lifestyle & Habits")
            in_traveltime = st.selectbox("Travel Time", [1, 2, 3, 4], format_func=lambda x: {1: "< 15 min", 2: "15-30 min", 3: "30-60 min", 4: "> 60 min"}[x], index=int(cur.get("traveltime", 1)) - 1)
            in_freetime = st.slider("Free Time After School (1-5)", 1, 5, int(cur.get("freetime", 3)))
            in_goout = st.slider("Going Out with Friends (1-5)", 1, 5, int(cur.get("goout", 3)))
            in_dalc = st.slider("Workday Alcohol Consumption (1-5)", 1, 5, int(cur.get("Dalc", 1)))
            in_walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, int(cur.get("Walc", 1)))
            in_health = st.slider("Current Health Status (1-5)", 1, 5, int(cur.get("health", 4)))
            in_absences = st.slider("School Absences (0-93)", 0, 75, int(cur.get("absences", 4)))
            in_higher = st.selectbox("Desires Higher Education", ["yes", "no"], index=0 if cur.get("higher", "yes") == "yes" else 1)
            in_internet = st.selectbox("Internet Access at Home", ["yes", "no"], index=0 if cur.get("internet", "yes") == "yes" else 1)
            in_romantic = st.selectbox("Romantic Relationship", ["no", "yes"], index=0 if cur.get("romantic", "no") == "no" else 1)
            in_activities = st.selectbox("Extracurricular Activities", ["yes", "no"], index=0 if cur.get("activities", "yes") == "yes" else 1)
            in_nursery = st.selectbox("Attended Nursery School", ["yes", "no"], index=0 if cur.get("nursery", "yes") == "yes" else 1)

        submitted = st.form_submit_button("🔮 Predict Student Performance", use_container_width=True)

    # When form is submitted or initialized
    input_dict = {
        "school": in_school, "sex": in_sex, "age": in_age, "address": in_address, "famsize": in_famsize,
        "Pstatus": in_pstatus, "Medu": in_medu, "Fedu": in_fedu, "Mjob": in_mjob, "Fjob": in_fjob,
        "reason": in_reason, "guardian": in_guardian, "traveltime": in_traveltime, "studytime": in_studytime,
        "failures": in_failures, "schoolsup": in_schoolsup, "famsup": in_famsup, "paid": in_paid,
        "activities": in_activities, "nursery": in_nursery, "higher": in_higher, "internet": in_internet,
        "romantic": in_romantic, "famrel": 4, "freetime": in_freetime, "goout": in_goout,
        "Dalc": in_dalc, "Walc": in_walc, "health": in_health, "absences": in_absences
    }
    if include_grades:
        input_dict["G1"] = in_g1
        input_dict["G2"] = in_g2

    input_df = pd.DataFrame([input_dict])
    X_single = preprocessor.transform(input_df)
    predictions = model_manager.predict_single(X_single)

    st.markdown("### 🏆 Prediction Results Dashboard")
    
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    # Decision Tree
    with p_col1:
        dt_res = predictions["Decision Tree"]
        badge_cls = "pass-badge" if dt_res["prediction"] == "Pass" else "fail-badge"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌲 Decision Tree</h3>
            <p><span class="{badge_cls}">{dt_res['prediction']}</span></p>
        </div>
        """, unsafe_allow_html=True)
        if dt_res["probabilities"]:
            st.caption(f"Pass Probability: **{dt_res['probabilities']['Pass']*100:.1f}%**")
            st.progress(float(dt_res["probabilities"]["Pass"]))

    # Naïve Bayes
    with p_col2:
        nb_res = predictions["Naïve Bayes"]
        badge_cls = "pass-badge" if nb_res["prediction"] == "Pass" else "fail-badge"
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Naïve Bayes</h3>
            <p><span class="{badge_cls}">{nb_res['prediction']}</span></p>
        </div>
        """, unsafe_allow_html=True)
        if nb_res["probabilities"]:
            st.caption(f"Pass Probability: **{nb_res['probabilities']['Pass']*100:.1f}%**")
            st.progress(float(nb_res["probabilities"]["Pass"]))

    # KNN
    with p_col3:
        knn_res = predictions["K-Nearest Neighbors"]
        badge_cls = "pass-badge" if knn_res["prediction"] == "Pass" else "fail-badge"
        st.markdown(f"""
        <div class="metric-card">
            <h3>📍 K-Nearest Neighbors</h3>
            <p><span class="{badge_cls}">{knn_res['prediction']}</span></p>
        </div>
        """, unsafe_allow_html=True)
        if knn_res["probabilities"]:
            st.caption(f"Pass Probability: **{knn_res['probabilities']['Pass']*100:.1f}%**")
            st.progress(float(knn_res["probabilities"]["Pass"]))

    # Majority Vote
    with p_col4:
        maj_res = predictions["Majority Vote"]
        badge_cls = "pass-badge" if maj_res["prediction"] == "Pass" else "fail-badge"
        st.markdown(f"""
        <div class="metric-card majority-card">
            <h3>🗳️ Majority Vote</h3>
            <p><span class="{badge_cls}">{maj_res['prediction']}</span></p>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Consensus: **{maj_res['consensus']}** ({maj_res['pass_votes']}/{maj_res['total_votes']} votes)")
        st.progress(float(maj_res["agreement_pct"] / 100.0))

    # Actionable Academic Advice
    st.markdown("#### 💡 Actionable Insights & Academic Advice")
    if maj_res["prediction"] == "Pass":
        st.success(
            f"✅ **High Likelihood of Academic Success!** "
            f"The student profile exhibits strong positive indicators. "
            f"Recommendations: Maintain consistent weekly study hours, continue extracurricular engagement, and pursue higher education goals."
        )
    else:
        st.warning(
            f"⚠️ **At-Risk Warning:** Majority consensus forecasts academic failure (< 10). "
            f"Key risk factors detected may include past failures ({in_failures}), high absenteeism ({in_absences} days), "
            f"or insufficient weekly study hours. "
            f"Recommended Interventions: Enroll in remedial tutoring, establish regular parent-teacher check-ins, and implement an attendance recovery plan."
        )


# -----------------------------------------------------------------------------
# TAB 4: Batch CSV Prediction
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📁 Batch Student Prediction from CSV")
    st.markdown("Upload a CSV file with student records to generate predictions across all 3 models + majority vote simultaneously.")

    # Option to download a sample template
    sample_download = labeled_df.drop(columns=["passed", "pass_label"], errors="ignore").head(5)
    csv_buffer = io.StringIO()
    sample_download.to_csv(csv_buffer, index=False, sep=";")
    
    st.download_button(
        label="📥 Download Sample Student Batch CSV Template",
        data=csv_buffer.getvalue(),
        file_name="sample_students_batch.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV file for Batch Prediction", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Check separator
            content = uploaded_file.getvalue().decode("utf-8")
            delimiter = ";" if ";" in content.split("\n")[0] else ","
            uploaded_file.seek(0)
            
            batch_df = pd.read_csv(uploaded_file, sep=delimiter)
            st.write(f"Loaded **{len(batch_df)} student records**.")
            
            # Preprocess and predict
            X_batch = preprocessor.transform(batch_df)
            
            dt_preds = model_manager.models["Decision Tree"].predict(X_batch)
            nb_preds = model_manager.models["Naïve Bayes"].predict(X_batch)
            knn_preds = model_manager.models["K-Nearest Neighbors"].predict(X_batch)
            
            preds_stacked = np.array([dt_preds, nb_preds, knn_preds])
            majority_batch = (np.sum(preds_stacked, axis=0) >= 2).astype(int)
            
            results_df = batch_df.copy()
            results_df["Decision Tree Pred"] = np.where(dt_preds == 1, "Pass", "Fail")
            results_df["Naïve Bayes Pred"] = np.where(nb_preds == 1, "Pass", "Fail")
            results_df["KNN Pred"] = np.where(knn_preds == 1, "Pass", "Fail")
            results_df["Majority Vote Pred"] = np.where(majority_batch == 1, "Pass", "Fail")
            
            st.markdown("#### 🎯 Batch Prediction Outputs")
            st.dataframe(results_df, use_container_width=True)
            
            # Export CSV
            out_buffer = io.StringIO()
            results_df.to_csv(out_buffer, index=False)
            st.download_button(
                "📥 Download Predictions CSV",
                data=out_buffer.getvalue(),
                file_name="student_predictions_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing uploaded batch file: {str(e)}")
    else:
        st.info("Upload a CSV file above or test the models with the sample holdout test set below:")
        
        test_preview = labeled_df.iloc[:10].copy()
        X_sub = preprocessor.transform(test_preview)
        p_dt = model_manager.models["Decision Tree"].predict(X_sub)
        p_nb = model_manager.models["Naïve Bayes"].predict(X_sub)
        p_knn = model_manager.models["K-Nearest Neighbors"].predict(X_sub)
        p_maj = (np.array([p_dt, p_nb, p_knn]).sum(axis=0) >= 2).astype(int)
        
        test_preview["DT_Pred"] = np.where(p_dt == 1, "Pass", "Fail")
        test_preview["NB_Pred"] = np.where(p_nb == 1, "Pass", "Fail")
        test_preview["KNN_Pred"] = np.where(p_knn == 1, "Pass", "Fail")
        test_preview["Majority_Pred"] = np.where(p_maj == 1, "Pass", "Fail")
        
        st.dataframe(
            test_preview[["school", "age", "studytime", "failures", "absences", "pass_label", "DT_Pred", "NB_Pred", "KNN_Pred", "Majority_Pred"]],
            use_container_width=True
        )


# -----------------------------------------------------------------------------
# TAB 5: Educational Guide & Theory for Beginners
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("📚 Machine Learning Concepts & Implementation Guide")
    st.markdown(
        "This project is crafted for **Machine Learning Foundations** students to demonstrate core concepts, "
        "data engineering, memory management, and classifier mechanics."
    )
    
    for model_name, info in MODEL_DESCRIPTIONS.items():
        with st.expander(f"📖 {info['title']}", expanded=(model_name == "Decision Tree")):
            st.markdown(f"**Concept:** {info['concept']}")
            st.markdown(f"**How It Operates:** {info['how_it_works']}")
            
            c_str, c_weak = st.columns(2)
            with c_str:
                st.success(f"**Strengths:** {info['strengths']}")
            with c_weak:
                st.error(f"**Weaknesses / Limitations:** {info['weaknesses']}")

    st.markdown("---")
    st.markdown("### 🛠️ Python & Software Engineering Best Practices Used")
    st.markdown("""
    1. **Modular Architecture:** Separated concerns into `data_loader.py`, `preprocessor.py`, `models.py`, and `utils.py`.
    2. **Memory Profiling & Optimization:** Measured memory via `df.memory_usage(deep=True)` and demonstrated downcasting of numeric columns and string-to-category conversions.
    3. **Function Signatures & Type Hints:** Explicit typing (`Tuple`, `List`, `Dict`, `Optional`) ensures code maintainability and self-documentation.
    4. **Defensive Preprocessing:** Handles missing values via median/mode imputation, aligns column names, and prevents data leakage using `ColumnTransformer`.
    5. **Model Generalization:** Applied stratified train/test splitting to preserve the natural Pass/Fail balance in academic data.
    """)
