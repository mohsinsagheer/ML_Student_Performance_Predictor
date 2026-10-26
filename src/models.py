"""
Machine Learning Models Module for Student Performance Prediction
-----------------------------------------------------------------
Demonstrates:
- Decision Tree Classifier implementation & feature importances
- Naïve Bayes (GaussianNB) implementation & probabilistic reasoning
- K-Nearest Neighbors (KNN) Classifier implementation
- Ensemble Majority Voting logic
- Comprehensive evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Model memory consumption analysis
"""

import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class StudentModelManager:
    """
    Orchestrates training, evaluation, comparison, and prediction for:
    1. Decision Tree Classifier
    2. Naïve Bayes Classifier (GaussianNB)
    3. K-Nearest Neighbors (KNN) Classifier
    4. Majority-Vote Ensemble
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.feature_names: List[str] = []

    def train_decision_tree(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        criterion: str = "gini",
        max_depth: Optional[int] = 5,
        min_samples_split: int = 2,
        random_state: int = 42
    ) -> DecisionTreeClassifier:
        """
        Train Decision Tree Classifier.
        
        Parameters:
        -----------
        criterion : str, default='gini'
            Splitting quality criterion ('gini', 'entropy', 'log_loss').
        max_depth : int or None, default=5
            Maximum depth of tree (controls overfitting).
        min_samples_split : int, default=2
            Minimum samples needed to split an internal node.
        """
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        self.models["Decision Tree"] = model
        return model

    def train_naive_bayes(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        var_smoothing: float = 1e-9
    ) -> GaussianNB:
        """
        Train Gaussian Naïve Bayes Classifier.
        
        Parameters:
        -----------
        var_smoothing : float, default=1e-9
            Portion of largest variance of all features added to variances for calculation stability.
        """
        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(X_train, y_train)
        self.models["Naïve Bayes"] = model
        return model

    def train_knn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "minkowski"
    ) -> KNeighborsClassifier:
        """
        Train K-Nearest Neighbors Classifier.
        
        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors (k).
        weights : str, default='uniform'
            Weight function ('uniform' or 'distance').
        metric : str, default='minkowski'
            Distance metric ('minkowski', 'euclidean', 'manhattan').
        """
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
        model.fit(X_train, y_train)
        self.models["K-Nearest Neighbors"] = model
        return model

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        dt_params: Optional[Dict[str, Any]] = None,
        nb_params: Optional[Dict[str, Any]] = None,
        knn_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Trains all 3 classifiers with customizable or default hyperparameter settings.
        """
        dt_params = dt_params or {}
        nb_params = nb_params or {}
        knn_params = knn_params or {}

        if feature_names is not None:
            self.feature_names = feature_names

        self.train_decision_tree(X_train, y_train, **dt_params)
        self.train_naive_bayes(X_train, y_train, **nb_params)
        self.train_knn(X_train, y_train, **knn_params)

        return self.models

    def evaluate_model(
        self,
        name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculates standard classification performance metrics and confusion matrix.
        """
        y_pred = model.predict(X_test)
        
        # Calculate probabilities if supported
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Calculate memory footprint
        in_memory_bytes = sys.getsizeof(model)
        pickle_bytes = len(pickle.dumps(model))

        res = {
            "model_name": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "memory_bytes": in_memory_bytes,
            "serialized_bytes": pickle_bytes
        }
        return res

    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Evaluates all trained individual models as well as the Majority-Vote ensemble.
        
        Returns:
        --------
        Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
            Comparison summary table and full evaluation dictionaries.
        """
        self.evaluation_results = {}
        predictions_dict = {}

        # Evaluate individual models
        for name, model in self.models.items():
            res = self.evaluate_model(name, model, X_test, y_test)
            self.evaluation_results[name] = res
            predictions_dict[name] = res["y_pred"]

        # Evaluate Majority-Vote Ensemble
        if len(predictions_dict) == 3:
            preds_matrix = np.array([
                predictions_dict["Decision Tree"],
                predictions_dict["Naïve Bayes"],
                predictions_dict["K-Nearest Neighbors"]
            ])
            # Majority vote: 1 if sum >= 2 else 0
            majority_preds = (np.sum(preds_matrix, axis=0) >= 2).astype(int)
            
            maj_acc = accuracy_score(y_test, majority_preds)
            maj_prec = precision_score(y_test, majority_preds, zero_division=0)
            maj_rec = recall_score(y_test, majority_preds, zero_division=0)
            maj_f1 = f1_score(y_test, majority_preds, zero_division=0)
            maj_cm = confusion_matrix(y_test, majority_preds)

            self.evaluation_results["Majority Vote"] = {
                "model_name": "Majority Vote (Ensemble)",
                "accuracy": round(float(maj_acc), 4),
                "precision": round(float(maj_prec), 4),
                "recall": round(float(maj_rec), 4),
                "f1_score": round(float(maj_f1), 4),
                "confusion_matrix": maj_cm,
                "y_pred": majority_preds,
                "y_proba": np.mean(preds_matrix, axis=0),
                "memory_bytes": 0,
                "serialized_bytes": 0
            }

        # Build comparison DataFrame
        records = []
        for name, res in self.evaluation_results.items():
            records.append({
                "Model": name,
                "Accuracy": f"{res['accuracy'] * 100:.2f}%",
                "Precision": f"{res['precision'] * 100:.2f}%",
                "Recall": f"{res['recall'] * 100:.2f}%",
                "F1-Score": f"{res['f1_score'] * 100:.2f}%",
                "Raw Accuracy": res["accuracy"],
                "Raw Precision": res["precision"],
                "Raw Recall": res["recall"],
                "Raw F1": res["f1_score"],
                "Model Size (Bytes)": res["serialized_bytes"]
            })

        comparison_df = pd.DataFrame(records)
        return comparison_df, self.evaluation_results

    def get_feature_importances(self) -> Optional[pd.DataFrame]:
        """
        Extracts feature importances from trained Decision Tree model.
        """
        dt_model = self.models.get("Decision Tree")
        if dt_model is None or not hasattr(dt_model, "feature_importances_"):
            return None

        importances = dt_model.feature_importances_
        if len(self.feature_names) == len(importances):
            names = self.feature_names
        else:
            names = [f"Feature {i}" for i in range(len(importances))]

        fi_df = pd.DataFrame({
            "Feature": names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        return fi_df

    def predict_single(self, X_single: np.ndarray) -> Dict[str, Any]:
        """
        Runs prediction on a single student feature vector across all models.
        
        Parameters:
        -----------
        X_single : np.ndarray
            Single transformed feature array shape (1, num_features).
            
        Returns:
        --------
        Dict[str, Any]:
            Predictions, probabilities, majority vote, and consensus agreement.
        """
        results = {}
        votes = []

        for name in ["Decision Tree", "Naïve Bayes", "K-Nearest Neighbors"]:
            model = self.models.get(name)
            if model is None:
                continue

            pred = int(model.predict(X_single)[0])
            votes.append(pred)
            
            # Predict probability if available
            proba = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_single)[0]
                proba = {
                    "Fail": round(float(probs[0]), 4),
                    "Pass": round(float(probs[1]), 4)
                }

            results[name] = {
                "prediction": "Pass" if pred == 1 else "Fail",
                "pred_int": pred,
                "probabilities": proba
            }

        # Majority Vote
        if votes:
            pass_votes = sum(votes)
            total_votes = len(votes)
            majority_pred = 1 if pass_votes >= (total_votes / 2.0) else 0
            unanimous = (pass_votes == total_votes) or (pass_votes == 0)

            results["Majority Vote"] = {
                "prediction": "Pass" if majority_pred == 1 else "Fail",
                "pred_int": majority_pred,
                "pass_votes": pass_votes,
                "total_votes": total_votes,
                "consensus": "Unanimous" if unanimous else f"Split ({pass_votes}/{total_votes})",
                "agreement_pct": round((max(pass_votes, total_votes - pass_votes) / total_votes) * 100, 1)
            }

        return results
