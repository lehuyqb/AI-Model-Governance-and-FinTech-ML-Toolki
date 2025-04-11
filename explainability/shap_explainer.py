"""
SHAP-based model explainer for financial ML models.
Provides both global and local explanations for model predictions.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Union, List, Dict, Any, Optional
from sklearn.base import BaseEstimator


class ModelExplainer:
    """
    A class for generating explanations for ML model predictions using SHAP values.
    
    This explainer supports both global feature importance analysis and local
    explanations for individual predictions. It's particularly suited for
    financial models where interpretability and regulatory compliance are crucial.
    
    Attributes:
        model: The trained ML model to explain
        explainer: The SHAP explainer object
        feature_names: List of feature names
        background_data: Background dataset used for SHAP value computation
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        background_data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the ModelExplainer.
        
        Args:
            model: Trained ML model (must be compatible with SHAP)
            background_data: Background dataset for SHAP value computation
            feature_names: List of feature names (if None, will use indices)
        """
        self.model = model
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(background_data.shape[1])]
        self.background_data = background_data
        
        # Initialize SHAP explainer based on model type
        if hasattr(model, 'predict_proba'):
            self.explainer = shap.KernelExplainer(model.predict_proba, background_data)
        else:
            self.explainer = shap.KernelExplainer(model.predict, background_data)
    
    def get_feature_importance(
        self,
        n_samples: int = 100,
        plot: bool = True,
        plot_type: str = 'bar'
    ) -> Dict[str, float]:
        """
        Compute global feature importance using SHAP values.
        
        Args:
            n_samples: Number of samples to use for computing SHAP values
            plot: Whether to generate a visualization
            plot_type: Type of plot ('bar' or 'summary')
            
        Returns:
            Dictionary mapping feature names to their importance scores
        """
        # Compute SHAP values for a subset of the data
        shap_values = self.explainer.shap_values(
            self.background_data.iloc[:n_samples] if isinstance(self.background_data, pd.DataFrame)
            else self.background_data[:n_samples]
        )
        
        # If model outputs probabilities, take mean absolute value across classes
        if isinstance(shap_values, list):
            shap_values = np.abs(np.array(shap_values)).mean(0)
        
        # Calculate mean absolute SHAP values for each feature
        feature_importance = np.abs(shap_values).mean(0)
        
        # Create feature importance dictionary
        importance_dict = dict(zip(self.feature_names, feature_importance))
        
        if plot:
            if plot_type == 'bar':
                self._plot_feature_importance_bar(importance_dict)
            else:
                self._plot_feature_importance_summary(shap_values)
        
        return importance_dict
    
    def explain_prediction(
        self,
        sample: Union[pd.DataFrame, np.ndarray],
        plot: bool = True
    ) -> Dict[str, float]:
        """
        Generate local explanation for a single prediction.
        
        Args:
            sample: Input sample to explain
            plot: Whether to generate a visualization
            
        Returns:
            Dictionary mapping feature names to their SHAP values
        """
        # Ensure sample is in correct format
        if isinstance(sample, pd.DataFrame):
            sample = sample.values
        
        # Compute SHAP values for the sample
        shap_values = self.explainer.shap_values(sample)
        
        # If model outputs probabilities, take values for the predicted class
        if isinstance(shap_values, list):
            pred_class = np.argmax(self.model.predict_proba(sample))
            shap_values = shap_values[pred_class]
        
        # Create feature contribution dictionary
        contributions = dict(zip(self.feature_names, shap_values[0]))
        
        if plot:
            self._plot_local_explanation(contributions)
        
        return contributions
    
    def _plot_feature_importance_bar(self, importance_dict: Dict[str, float]) -> None:
        """Generate bar plot of feature importance."""
        plt.figure(figsize=(10, 6))
        features = list(importance_dict.keys())
        values = list(importance_dict.values())
        
        # Sort by absolute importance
        sorted_idx = np.argsort(np.abs(values))
        features = [features[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        plt.barh(range(len(features)), values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance (Global)')
        plt.tight_layout()
        plt.show()
    
    def _plot_feature_importance_summary(self, shap_values: np.ndarray) -> None:
        """Generate SHAP summary plot."""
        shap.summary_plot(
            shap_values,
            self.background_data,
            feature_names=self.feature_names,
            show=False
        )
        plt.title('Feature Importance Summary')
        plt.tight_layout()
        plt.show()
    
    def _plot_local_explanation(self, contributions: Dict[str, float]) -> None:
        """Generate waterfall plot for local explanation."""
        # Create waterfall plot using plotly
        features = list(contributions.keys())
        values = list(contributions.values())
        
        # Sort by absolute contribution
        sorted_idx = np.argsort(np.abs(values))
        features = [features[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # Calculate cumulative sum for waterfall
        cumulative = np.cumsum(values)
        
        fig = go.Figure(go.Waterfall(
            name="Feature Contributions",
            orientation="v",
            measure=["relative"] * len(features) + ["total"],
            x=features + ["Total"],
            y=values + [cumulative[-1]],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            text=[f"{v:.2f}" for v in values] + [f"{cumulative[-1]:.2f}"],
            textposition="outside",
        ))
        
        fig.update_layout(
            title="Local Feature Contributions",
            showlegend=False,
            height=600
        )
        
        fig.show() 