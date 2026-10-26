"""
Student Performance Prediction System - Source Package
"""
from .data_loader import load_dataset, handle_missing_values, create_target_variable, get_memory_usage, optimize_memory
from .preprocessor import StudentDataPreprocessor
from .models import StudentModelManager
from .utils import (
    MODEL_DESCRIPTIONS,
    SAMPLE_STUDENT_PROFILES,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_grade_distribution,
    plot_study_vs_outcome
)

__all__ = [
    "load_dataset",
    "handle_missing_values",
    "create_target_variable",
    "get_memory_usage",
    "optimize_memory",
    "StudentDataPreprocessor",
    "StudentModelManager",
    "MODEL_DESCRIPTIONS",
    "SAMPLE_STUDENT_PROFILES",
    "plot_model_comparison",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_grade_distribution",
    "plot_study_vs_outcome"
]
