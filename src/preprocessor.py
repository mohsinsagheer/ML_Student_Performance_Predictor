"""
Data Preprocessing Module for Student Performance Prediction
-----------------------------------------------------------
Demonstrates:
- Feature selection and separation
- Train-test splitting with stratification
- Categorical encoding (One-Hot Encoding)
- Feature scaling with StandardScaler (essential for KNN & Naïve Bayes)
- Consistent transformation pipeline for live prediction
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class StudentDataPreprocessor:
    """
    End-to-end preprocessing pipeline for Student Performance datasets.
    Handles feature selection, categorical encoding, feature scaling,
    and single-instance transformation for user input prediction.
    """
    
    def __init__(self, include_period_grades: bool = True):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        include_period_grades : bool, default=True
            Whether to include G1 and G2 as predictor features.
            If False, models predict purely based on demographic & behavioral factors.
        """
        self.include_period_grades = include_period_grades
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_names_out: List[str] = []
        self.column_transformer: Optional[ColumnTransformer] = None
        self.is_fitted: bool = False
        
    def _identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Classifies columns into numeric and categorical features.
        Excludes target variable 'passed', label 'pass_label', and final grade 'G3'.
        """
        cols_to_exclude = {"G3", "passed", "pass_label", "course"}
        if not self.include_period_grades:
            cols_to_exclude.update({"G1", "G2"})
            
        numeric_cols = []
        categorical_cols = []
        
        for col in df.columns:
            if col in cols_to_exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
                
        return numeric_cols, categorical_cols

    def fit_transform(
        self, 
        df: pd.DataFrame, 
        target_col: str = "passed"
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Fits the encoders/scalers and transforms features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Cleaned dataframe containing features and target.
        target_col : str, default='passed'
            Name of binary target column.
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
            Transformed feature matrix X (numpy array), target array y,
            and feature dataframe with readable column names.
        """
        self.numeric_features, self.categorical_features = self._identify_feature_types(df)
        
        # Define transformers: StandardScaler for numerics, OneHotEncoder for categoricals
        # using sparse_output=False for dense matrices required by scikit-learn models
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), self.categorical_features),
            ],
            remainder="drop"
        )
        
        X_raw = df[self.numeric_features + self.categorical_features]
        y = df[target_col].to_numpy()
        
        X_transformed = preprocessor.fit_transform(X_raw)
        self.column_transformer = preprocessor
        self.is_fitted = True
        
        # Extract feature names after encoding
        cat_encoder = preprocessor.named_transformers_["cat"]
        encoded_cat_names = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
        self.feature_names_out = self.numeric_features + encoded_cat_names
        
        X_df = pd.DataFrame(X_transformed, columns=self.feature_names_out)
        return X_transformed, y, X_df

    def transform(self, df_input: pd.DataFrame) -> np.ndarray:
        """
        Transform new or single-student data using fitted transformers.
        
        Parameters:
        -----------
        df_input : pd.DataFrame
            Dataframe matching original feature schemas.
            
        Returns:
        --------
        np.ndarray: Transformed features array.
        """
        if not self.is_fitted or self.column_transformer is None:
            raise ValueError("Preprocessor has not been fitted yet! Call fit_transform first.")
            
        # Ensure all required columns exist in input
        required_cols = self.numeric_features + self.categorical_features
        for col in required_cols:
            if col not in df_input.columns:
                raise KeyError(f"Missing required feature '{col}' in input data.")
                
        X_subset = df_input[required_cols]
        return self.column_transformer.transform(X_subset)

    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2, 
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits data into train and test sets with stratification.
        
        Parameters:
        -----------
        X : np.ndarray
            Features matrix.
        y : np.ndarray
            Target labels array.
        test_size : float, default=0.2
            Proportion of dataset to include in test split (e.g., 0.2 = 20%).
        random_state : int, default=42
            Seed for reproducible random splitting.
        stratify : bool, default=True
            Whether to preserve target class distribution across splits.
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train, X_test, y_train, y_test
        """
        strat_param = y if stratify else None
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=strat_param
        )
