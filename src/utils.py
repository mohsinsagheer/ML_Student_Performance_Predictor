"""
Utilities and Visualizations Module for Student Performance Prediction
----------------------------------------------------------------------
Demonstrates:
- Dynamic Plotly interactive chart creation supporting both Light and Dark modes
- Preset student personas for instant UI testing
- Comprehensive educational guides for ML models
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List


# ---------------------------------------------------------
# Educational Information for Beginners
# ---------------------------------------------------------
MODEL_DESCRIPTIONS = {
    "Decision Tree": {
        "title": "Decision Tree Classifier",
        "concept": "A tree-like model that recursively splits data by asking binary questions about feature values.",
        "how_it_works": (
            "Selects the feature and split point that maximizes Information Gain or minimizes Gini Impurity. "
            "Internal nodes represent decisions on attributes, branches represent outcomes, and leaf nodes assign class labels (Pass/Fail)."
        ),
        "strengths": "Highly interpretable, handles non-linear relationships, requires minimal data preprocessing.",
        "weaknesses": "Prone to overfitting if tree depth is unconstrained; sensitive to small data variations."
    },
    "Naïve Bayes": {
        "title": "Gaussian Naïve Bayes Classifier",
        "concept": "A probabilistic classifier applying Bayes' Theorem with a 'naïve' assumption of conditional feature independence.",
        "how_it_works": (
            "Calculates posterior probability P(Pass | Features) using prior probability P(Pass) "
            "and likelihoods P(Feature_i | Pass) modeled via Gaussian distribution: P(C|X) ∝ P(C) ∏ P(x_i|C)."
        ),
        "strengths": "Fast training, low computational overhead, works well even with smaller datasets.",
        "weaknesses": "Assumes all predictor features are independent given the class label, which is rarely fully true in practice."
    },
    "K-Nearest Neighbors": {
        "title": "K-Nearest Neighbors (KNN) Classifier",
        "concept": "An instance-based 'lazy learning' algorithm that classifies a query sample based on the labels of its closest k neighbors.",
        "how_it_works": (
            "Computes distance (e.g. Euclidean or Manhattan) between the student's normalized feature vector and all training points. "
            "Assigns the majority class label among the k closest training students."
        ),
        "strengths": "Simple and intuitive, no explicit training phase, naturally adapts as new data is added.",
        "weaknesses": "Computationally intensive at test time, sensitive to feature scaling and irrelevant features."
    },
    "Majority Vote": {
        "title": "Majority-Vote Ensemble",
        "concept": "Combines predictions from Decision Tree, Naïve Bayes, and KNN via democratic consensus.",
        "how_it_works": (
            "If at least 2 out of the 3 classifiers predict 'Pass', the ensemble outcome is 'Pass'; otherwise, 'Fail'. "
            "Leverages the diversity of distinct algorithms to reduce individual model bias and variance."
        ),
        "strengths": "More robust predictions, smoother decision boundary, reduced variance.",
        "weaknesses": "Slightly less interpretable than a single model alone."
    }
}


# ---------------------------------------------------------
# Pre-configured Student Personas for Live Testing
# ---------------------------------------------------------
SAMPLE_STUDENT_PROFILES = {
    "🌟 High Achieving Scholar": {
        "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "LE3", "Pstatus": "T",
        "Medu": 4, "Fedu": 4, "Mjob": "teacher", "Fjob": "services", "reason": "reputation",
        "guardian": "mother", "traveltime": 1, "studytime": 4, "failures": 0, "schoolsup": "no",
        "famsup": "yes", "paid": "yes", "activities": "yes", "nursery": "yes", "higher": "yes",
        "internet": "yes", "romantic": "no", "famrel": 5, "freetime": 3, "goout": 2,
        "Dalc": 1, "Walc": 1, "health": 5, "absences": 2, "G1": 17, "G2": 18
    },
    "⚠️ At-Risk / Struggling Student": {
        "school": "MS", "sex": "M", "age": 19, "address": "R", "famsize": "GT3", "Pstatus": "A",
        "Medu": 1, "Fedu": 1, "Mjob": "other", "Fjob": "other", "reason": "course",
        "guardian": "other", "traveltime": 3, "studytime": 1, "failures": 3, "schoolsup": "no",
        "famsup": "no", "paid": "no", "activities": "no", "nursery": "no", "higher": "no",
        "internet": "no", "romantic": "yes", "famrel": 2, "freetime": 5, "goout": 5,
        "Dalc": 3, "Walc": 4, "health": 2, "absences": 25, "G1": 6, "G2": 7
    },
    "📚 Average Hardworking Student": {
        "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 2, "Fedu": 2, "Mjob": "services", "Fjob": "services", "reason": "home",
        "guardian": "mother", "traveltime": 2, "studytime": 2, "failures": 0, "schoolsup": "no",
        "famsup": "yes", "paid": "no", "activities": "yes", "nursery": "yes", "higher": "yes",
        "internet": "yes", "romantic": "no", "famrel": 4, "freetime": 3, "goout": 3,
        "Dalc": 1, "Walc": 2, "health": 4, "absences": 6, "G1": 11, "G2": 12
    },
    "⚡ High Potential, High Absences": {
        "school": "GP", "sex": "M", "age": 18, "address": "U", "famsize": "LE3", "Pstatus": "T",
        "Medu": 3, "Fedu": 3, "Mjob": "health", "Fjob": "services", "reason": "reputation",
        "guardian": "father", "traveltime": 1, "studytime": 2, "failures": 0, "schoolsup": "no",
        "famsup": "yes", "paid": "yes", "activities": "no", "nursery": "yes", "higher": "yes",
        "internet": "yes", "romantic": "yes", "famrel": 3, "freetime": 4, "goout": 4,
        "Dalc": 2, "Walc": 3, "health": 3, "absences": 18, "G1": 14, "G2": 13
    }
}


# ---------------------------------------------------------
# Dynamic Plotly Visualization Builders (Light & Dark Support)
# ---------------------------------------------------------
def get_plotly_theme(is_dark: bool = False) -> Dict[str, Any]:
    """Helper to return consistent layout config based on theme."""
    return dict(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#F8FAFC" if is_dark else "#0F172A",
            family="Inter, -apple-system, sans-serif"
        )
    )


def plot_model_comparison(comparison_df: pd.DataFrame, is_dark: bool = False) -> go.Figure:
    """Creates a multi-metric comparative bar chart across models."""
    fig = go.Figure()
    
    if is_dark:
        colors = ["#6366F1", "#10B981", "#F59E0B", "#8B5CF6"]
        grid_color = "#334155"
    else:
        colors = ["#1E3A8A", "#059669", "#D97706", "#7C3AED"]
        grid_color = "#E2E8F0"
        
    metrics = [
        ("Raw Accuracy", "Accuracy", colors[0]),
        ("Raw Precision", "Precision", colors[1]),
        ("Raw Recall", "Recall", colors[2]),
        ("Raw F1", "F1-Score", colors[3])
    ]
    
    for metric_col, label, color in metrics:
        if metric_col in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=label,
                x=comparison_df["Model"],
                y=comparison_df[metric_col] * 100,
                text=(comparison_df[metric_col] * 100).round(1).astype(str) + "%",
                textposition="auto",
                marker_color=color
            ))

    fig.update_layout(
        title="<b>Classifier Performance Comparison (%)</b>",
        barmode="group",
        yaxis=dict(title="Score (%)", range=[0, 105], gridcolor=grid_color),
        xaxis=dict(title="Model", gridcolor=grid_color),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=40),
        height=400,
        **get_plotly_theme(is_dark)
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, model_name: str, is_dark: bool = False) -> go.Figure:
    """Creates an annotated confusion matrix heatmap."""
    total = np.sum(cm)
    percentages = (cm / total) * 100

    annot_text = [
        [f"<b>TN: {cm[0, 0]}</b><br>({percentages[0, 0]:.1f}%)", f"<b>FP: {cm[0, 1]}</b><br>({percentages[0, 1]:.1f}%)"],
        [f"<b>FN: {cm[1, 0]}</b><br>({percentages[1, 0]:.1f}%)", f"<b>TP: {cm[1, 1]}</b><br>({percentages[1, 1]:.1f}%)"]
    ]

    if is_dark:
        colorscale = [[0, "#1E293B"], [0.5, "#4338CA"], [1, "#6366F1"]]
        grid_color = "#334155"
    else:
        colorscale = [[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, "#1E3A8A"]]
        grid_color = "#E2E8F0"

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted Fail", "Predicted Pass"],
        y=["Actual Fail", "Actual Pass"],
        text=annot_text,
        texttemplate="%{text}",
        colorscale=colorscale,
        showscale=False
    ))

    fig.update_layout(
        title=f"<b>Confusion Matrix: {model_name}</b>",
        xaxis=dict(title="Predicted Outcome", gridcolor=grid_color),
        yaxis=dict(title="Actual Outcome", autorange="reversed", gridcolor=grid_color),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        **get_plotly_theme(is_dark)
    )
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 12, is_dark: bool = False) -> go.Figure:
    """Plots top feature importances for Decision Tree."""
    top_df = fi_df.head(top_n).sort_values(by="Importance", ascending=True)

    if is_dark:
        colorscale = [[0, "#4338CA"], [0.5, "#6366F1"], [1, "#06B6D4"]]
        grid_color = "#334155"
    else:
        colorscale = [[0, "#93C5FD"], [0.5, "#2563EB"], [1, "#1E3A8A"]]
        grid_color = "#E2E8F0"

    fig = go.Figure(go.Bar(
        x=top_df["Importance"],
        y=top_df["Feature"],
        orientation="h",
        marker=dict(
            color=top_df["Importance"],
            colorscale=colorscale,
            showscale=False
        ),
        text=(top_df["Importance"] * 100).round(2).astype(str) + "%",
        textposition="auto"
    ))

    fig.update_layout(
        title=f"<b>Decision Tree Top {top_n} Feature Importances</b>",
        xaxis=dict(title="Relative Importance Score", gridcolor=grid_color),
        yaxis=dict(title="Feature"),
        margin=dict(l=20, r=20, t=50, b=30),
        height=420,
        **get_plotly_theme(is_dark)
    )
    return fig


def plot_grade_distribution(df: pd.DataFrame, is_dark: bool = False) -> go.Figure:
    """Plots final grade G3 distribution broken down by Pass / Fail."""
    cutoff_color = "#F8FAFC" if is_dark else "#0F172A"
    grid_color = "#334155" if is_dark else "#E2E8F0"
    
    fig = px.histogram(
        df,
        x="G3",
        color="pass_label",
        color_discrete_map={"Pass": "#10B981", "Fail": "#EF4444"},
        nbins=21,
        barmode="overlay",
        title="<b>Distribution of Final Grades (G3) with Pass/Fail Threshold (10)</b>",
        labels={"G3": "Final Grade (0-20)", "pass_label": "Outcome"}
    )
    fig.add_vline(
        x=9.5, line_width=2, line_dash="dash", line_color=cutoff_color,
        annotation_text="Pass Cutoff (≥ 10)", annotation_position="top left",
        annotation_font_color=cutoff_color
    )
    fig.update_layout(
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color),
        margin=dict(l=20, r=20, t=50, b=30),
        height=360,
        **get_plotly_theme(is_dark)
    )
    return fig


def plot_study_vs_outcome(df: pd.DataFrame, is_dark: bool = False) -> go.Figure:
    """Plots study time vs pass rate breakdown."""
    study_map = {1: "<2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: ">10 hrs"}
    df_copy = df.copy()
    df_copy["study_label"] = df_copy["studytime"].map(study_map)

    summary = df_copy.groupby("study_label")["passed"].agg(
        Total="count",
        Passed="sum"
    ).reset_index()
    summary["Pass Rate (%)"] = (summary["Passed"] / summary["Total"]) * 100

    if is_dark:
        colorscale = [[0, "#4338CA"], [0.5, "#6366F1"], [1, "#10B981"]]
        grid_color = "#334155"
    else:
        colorscale = [[0, "#93C5FD"], [0.5, "#2563EB"], [1, "#059669"]]
        grid_color = "#E2E8F0"

    fig = px.bar(
        summary,
        x="study_label",
        y="Pass Rate (%)",
        text=summary["Pass Rate (%)"].round(1).astype(str) + "%",
        title="<b>Student Pass Rate by Weekly Study Time</b>",
        labels={"study_label": "Weekly Study Time"},
        color="Pass Rate (%)",
        color_continuous_scale=colorscale
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis=dict(range=[0, 110], gridcolor=grid_color),
        xaxis=dict(gridcolor=grid_color),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=30),
        height=350,
        **get_plotly_theme(is_dark)
    )
    return fig
