"""
Data Loader Module for Student Performance Prediction System
------------------------------------------------------------
Demonstrates:
- File I/O with pandas
- Parameter passing and default arguments
- Memory profiling and optimization
- Missing value detection and handling
- Target variable creation
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


def load_dataset(dataset_type: str = "math", data_dir: str = "student-data") -> pd.DataFrame:
    """
    Load student dataset from CSV file.
    
    Parameters:
    -----------
    dataset_type : str, default='math'
        Choice of dataset: 'math' (student-mat.csv), 'portuguese' (student-por.csv),
        or 'combined' (merged on common student identifiers).
    data_dir : str, default='student-data'
        Directory where CSV files are stored.
        
    Returns:
    --------
    pd.DataFrame: Loaded student dataframe.
    """
    mat_path = os.path.join(data_dir, "student-mat.csv")
    por_path = os.path.join(data_dir, "student-por.csv")
    
    if dataset_type == "math":
        df = pd.read_csv(mat_path, sep=";")
        df["course"] = "Mathematics"
    elif dataset_type == "portuguese":
        df = pd.read_csv(por_path, sep=";")
        df["course"] = "Portuguese"
    elif dataset_type == "combined":
        # Load both and merge or concatenate
        df_mat = pd.read_csv(mat_path, sep=";")
        df_mat["course"] = "Mathematics"
        df_por = pd.read_csv(por_path, sep=";")
        df_por["course"] = "Portuguese"
        df = pd.concat([df_mat, df_por], ignore_index=True)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Choose 'math', 'portuguese', or 'combined'.")
        
    return df


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Inspects and handles any missing values in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input raw dataframe.
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, int]]:
        Cleaned dataframe and a dictionary counting missing values per column.
    """
    df_clean = df.copy()
    missing_summary = df_clean.isnull().sum().to_dict()
    
    # Impute missing values if any occur
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                # Impute numeric features with median (robust against outliers)
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
            else:
                # Impute categorical features with mode
                mode_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
                
    return df_clean, missing_summary


def create_target_variable(df: pd.DataFrame, threshold: int = 10, target_col_name: str = "passed") -> pd.DataFrame:
    """
    Creates binary classification target variable (Pass/Fail) from final grade G3.
    According to standard Portuguese secondary grading:
    G3 >= 10: Pass (1)
    G3 < 10: Fail (0)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with G3 column.
    threshold : int, default=10
        Grade threshold for passing.
    target_col_name : str, default='passed'
        Name of new target column.
        
    Returns:
    --------
    pd.DataFrame: Dataframe with added target column.
    """
    df_out = df.copy()
    if "G3" not in df_out.columns:
        raise KeyError("Column 'G3' not found in dataset.")
        
    df_out[target_col_name] = (df_out["G3"] >= threshold).astype(int)
    df_out["pass_label"] = df_out[target_col_name].map({1: "Pass", 0: "Fail"})
    return df_out


def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate detailed memory footprint of the DataFrame.
    Demonstrates Python data structures, memory calculation, and introspection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
        
    Returns:
    --------
    Dict[str, Any]:
        Total memory in KB, per-column memory in bytes, and breakdown by data type.
    """
    mem_series = df.memory_usage(deep=True)
    total_bytes = mem_series.sum()
    total_kb = total_bytes / 1024.0
    total_mb = total_kb / 1024.0
    
    dtype_counts = df.dtypes.value_counts().to_dict()
    # Convert dtype keys to strings for serialization
    dtype_breakdown = {str(k): v for k, v in dtype_counts.items()}
    
    return {
        "total_bytes": int(total_bytes),
        "total_kb": round(total_kb, 2),
        "total_mb": round(total_mb, 4),
        "per_column_kb": (mem_series / 1024.0).round(2).to_dict(),
        "dtype_breakdown": dtype_breakdown,
        "rows": len(df),
        "columns": len(df.columns)
    }


def optimize_memory(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Downcasts numeric columns and converts categorical strings to category types.
    Shows memory efficiency optimization techniques in Python & Pandas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, float]]:
        Optimized dataframe and savings comparison (before KB, after KB, reduction %).
    """
    before_mem = df.memory_usage(deep=True).sum() / 1024.0
    df_opt = df.copy()
    
    for col in df_opt.columns:
        col_type = df_opt[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            c_min = df_opt[col].min()
            c_max = df_opt[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df_opt[col] = df_opt[col].astype(np.uint8)
                elif c_max < 65535:
                    df_opt[col] = df_opt[col].astype(np.uint16)
                else:
                    df_opt[col] = df_opt[col].astype(np.uint32)
            else:
                if c_min > -128 and c_max < 127:
                    df_opt[col] = df_opt[col].astype(np.int8)
                elif c_min > -32768 and c_max < 32767:
                    df_opt[col] = df_opt[col].astype(np.int16)
                else:
                    df_opt[col] = df_opt[col].astype(np.int32)
        elif pd.api.types.is_float_dtype(col_type):
            df_opt[col] = df_opt[col].astype(np.float32)
        elif col_type == object or str(col_type).startswith("str"):
            df_opt[col] = df_opt[col].astype("category")
            
    after_mem = df_opt.memory_usage(deep=True).sum() / 1024.0
    pct_reduction = ((before_mem - after_mem) / before_mem) * 100.0 if before_mem > 0 else 0.0
    
    return df_opt, {
        "before_kb": round(before_mem, 2),
        "after_kb": round(after_mem, 2),
        "reduction_pct": round(pct_reduction, 2)
    }
