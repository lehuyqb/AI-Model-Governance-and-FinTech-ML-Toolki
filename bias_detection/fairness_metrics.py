"""
Bias detection module for financial ML models.
Provides comprehensive fairness metrics and analysis capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp
import warnings


class BiasDetector:
    """
    A class for detecting and measuring bias in ML models.
    
    This detector supports various fairness metrics and bias analysis techniques
    specifically designed for financial applications. It helps identify potential
    discrimination based on protected attributes like gender, age, or income.
    
    Attributes:
        model: The trained ML model to analyze
        protected_attributes: List of protected attribute names
        privileged_groups: Dictionary mapping protected attributes to their privileged values
    """
    
    def __init__(
        self,
        model: object,
        protected_attributes: List[str],
        privileged_groups: Dict[str, Union[str, int, float]]
    ):
        """
        Initialize the BiasDetector.
        
        Args:
            model: Trained ML model to analyze
            protected_attributes: List of protected attribute names (e.g., ['gender', 'age'])
            privileged_groups: Dictionary mapping protected attributes to their privileged values
                             (e.g., {'gender': 'male', 'age': 25})
        """
        if not protected_attributes:
            raise ValueError("Protected attributes list cannot be empty")
        
        if set(protected_attributes) != set(privileged_groups.keys()):
            raise ValueError("Protected attributes and privileged groups must match")
            
        self.model = model
        self.protected_attributes = protected_attributes
        self.privileged_groups = privileged_groups
    
    def compute_disparate_impact(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Compute the disparate impact ratio for each protected attribute.
        
        The disparate impact ratio is the ratio of the probability of a positive outcome
        for the unprivileged group to that of the privileged group. A value close to 1
        indicates fairness, while values far from 1 indicate potential bias.
        
        Args:
            X: Features DataFrame
            y_true: True labels (if None, uses model predictions)
            
        Returns:
            Dictionary mapping protected attributes to their disparate impact ratios
        """
        if y_true is None:
            y_pred = self.model.predict(X)
        else:
            y_pred = y_true
            
        ratios = {}
        
        for attr in self.protected_attributes:
            privileged_value = self.privileged_groups[attr]
            
            # Get predictions for privileged and unprivileged groups
            privileged_mask = X[attr] == privileged_value
            unprivileged_mask = ~privileged_mask
            
            privileged_prob = np.mean(y_pred[privileged_mask])
            unprivileged_prob = np.mean(y_pred[unprivileged_mask])
            
            # Compute ratio (avoid division by zero)
            ratio = unprivileged_prob / privileged_prob if privileged_prob > 0 else 0
            ratios[attr] = ratio
            
        return ratios
    
    def compute_demographic_parity(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Compute the demographic parity difference for each protected attribute.
        
        Demographic parity difference is the absolute difference in positive prediction
        rates between privileged and unprivileged groups. A value close to 0 indicates
        fairness.
        
        Args:
            X: Features DataFrame
            y_true: True labels (if None, uses model predictions)
            
        Returns:
            Dictionary mapping protected attributes to their demographic parity differences
        """
        if y_true is None:
            y_pred = self.model.predict(X)
        else:
            y_pred = y_true
            
        differences = {}
        
        for attr in self.protected_attributes:
            privileged_value = self.privileged_groups[attr]
            
            # Get predictions for privileged and unprivileged groups
            privileged_mask = X[attr] == privileged_value
            unprivileged_mask = ~privileged_mask
            
            privileged_prob = np.mean(y_pred[privileged_mask])
            unprivileged_prob = np.mean(y_pred[unprivileged_mask])
            
            # Compute absolute difference
            difference = abs(privileged_prob - unprivileged_prob)
            differences[attr] = difference
            
        return differences
    
    def compute_equal_opportunity(
        self,
        X: pd.DataFrame,
        y_true: pd.Series
    ) -> Dict[str, float]:
        """
        Compute the equal opportunity difference for each protected attribute.
        
        Equal opportunity difference is the absolute difference in true positive rates
        between privileged and unprivileged groups. A value close to 0 indicates fairness.
        
        Args:
            X: Features DataFrame
            y_true: True labels
            
        Returns:
            Dictionary mapping protected attributes to their equal opportunity differences
        """
        y_pred = self.model.predict(X)
        differences = {}
        
        for attr in self.protected_attributes:
            privileged_value = self.privileged_groups[attr]
            
            # Get masks for privileged and unprivileged groups
            privileged_mask = X[attr] == privileged_value
            unprivileged_mask = ~privileged_mask
            
            # Compute true positive rates
            privileged_tp = np.sum((y_pred == 1) & (y_true == 1) & privileged_mask)
            privileged_p = np.sum((y_true == 1) & privileged_mask)
            privileged_tpr = privileged_tp / privileged_p if privileged_p > 0 else 0
            
            unprivileged_tp = np.sum((y_pred == 1) & (y_true == 1) & unprivileged_mask)
            unprivileged_p = np.sum((y_true == 1) & unprivileged_mask)
            unprivileged_tpr = unprivileged_tp / unprivileged_p if unprivileged_p > 0 else 0
            
            # Compute absolute difference
            difference = abs(privileged_tpr - unprivileged_tpr)
            differences[attr] = difference
            
        return differences
    
    def compute_equalized_odds(
        self,
        X: pd.DataFrame,
        y_true: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute the equalized odds differences for each protected attribute.
        
        Equalized odds difference is the maximum of the absolute differences in true positive
        rates and false positive rates between privileged and unprivileged groups.
        A value close to 0 indicates fairness.
        
        Args:
            X: Features DataFrame
            y_true: True labels
            
        Returns:
            Dictionary mapping protected attributes to their equalized odds differences
        """
        y_pred = self.model.predict(X)
        differences = {}
        
        for attr in self.protected_attributes:
            privileged_value = self.privileged_groups[attr]
            
            # Get masks for privileged and unprivileged groups
            privileged_mask = X[attr] == privileged_value
            unprivileged_mask = ~privileged_mask
            
            # Compute true positive rates
            privileged_tp = np.sum((y_pred == 1) & (y_true == 1) & privileged_mask)
            privileged_p = np.sum((y_true == 1) & privileged_mask)
            privileged_tpr = privileged_tp / privileged_p if privileged_p > 0 else 0
            
            unprivileged_tp = np.sum((y_pred == 1) & (y_true == 1) & unprivileged_mask)
            unprivileged_p = np.sum((y_true == 1) & unprivileged_mask)
            unprivileged_tpr = unprivileged_tp / unprivileged_p if unprivileged_p > 0 else 0
            
            # Compute false positive rates
            privileged_fp = np.sum((y_pred == 1) & (y_true == 0) & privileged_mask)
            privileged_n = np.sum((y_true == 0) & privileged_mask)
            privileged_fpr = privileged_fp / privileged_n if privileged_n > 0 else 0
            
            unprivileged_fp = np.sum((y_pred == 1) & (y_true == 0) & unprivileged_mask)
            unprivileged_n = np.sum((y_true == 0) & unprivileged_mask)
            unprivileged_fpr = unprivileged_fp / unprivileged_n if unprivileged_n > 0 else 0
            
            # Compute absolute differences
            tpr_diff = abs(privileged_tpr - unprivileged_tpr)
            fpr_diff = abs(privileged_fpr - unprivileged_fpr)
            
            # Store both differences
            differences[attr] = {
                'tpr_difference': tpr_diff,
                'fpr_difference': fpr_diff,
                'max_difference': max(tpr_diff, fpr_diff)
            }
            
        return differences
    
    def compute_treatment_equality(
        self,
        X: pd.DataFrame,
        y_true: pd.Series
    ) -> Dict[str, float]:
        """
        Compute the treatment equality ratio for each protected attribute.
        
        Treatment equality ratio is the ratio of false negative rates to false positive rates
        for unprivileged groups compared to privileged groups. A value close to 1 indicates fairness.
        
        Args:
            X: Features DataFrame
            y_true: True labels
            
        Returns:
            Dictionary mapping protected attributes to their treatment equality ratios
        """
        y_pred = self.model.predict(X)
        ratios = {}
        
        for attr in self.protected_attributes:
            privileged_value = self.privileged_groups[attr]
            
            # Get masks for privileged and unprivileged groups
            privileged_mask = X[attr] == privileged_value
            unprivileged_mask = ~privileged_mask
            
            # Compute false negative rates
            privileged_fn = np.sum((y_pred == 0) & (y_true == 1) & privileged_mask)
            privileged_p = np.sum((y_true == 1) & privileged_mask)
            privileged_fnr = privileged_fn / privileged_p if privileged_p > 0 else 0
            
            unprivileged_fn = np.sum((y_pred == 0) & (y_true == 1) & unprivileged_mask)
            unprivileged_p = np.sum((y_true == 1) & unprivileged_mask)
            unprivileged_fnr = unprivileged_fn / unprivileged_p if unprivileged_p > 0 else 0
            
            # Compute false positive rates
            privileged_fp = np.sum((y_pred == 1) & (y_true == 0) & privileged_mask)
            privileged_n = np.sum((y_true == 0) & privileged_mask)
            privileged_fpr = privileged_fp / privileged_n if privileged_n > 0 else 0
            
            unprivileged_fp = np.sum((y_pred == 1) & (y_true == 0) & unprivileged_mask)
            unprivileged_n = np.sum((y_true == 0) & unprivileged_mask)
            unprivileged_fpr = unprivileged_fp / unprivileged_n if unprivileged_n > 0 else 0
            
            # Compute ratios
            fnr_ratio = unprivileged_fnr / privileged_fnr if privileged_fnr > 0 else 0
            fpr_ratio = unprivileged_fpr / privileged_fpr if privileged_fpr > 0 else 0
            
            # Store the ratio
            ratios[attr] = fnr_ratio / fpr_ratio if fpr_ratio > 0 else 0
            
        return ratios
    
    def compute_predictive_parity(
        self,
        X: pd.DataFrame,
        y_true: pd.Series
    ) -> Dict[str, float]:
        """
        Compute the predictive parity difference for each protected attribute.
        
        Predictive parity difference is the absolute difference in positive predictive values
        between privileged and unprivileged groups. A value close to 0 indicates fairness.
        
        Args:
            X: Features DataFrame
            y_true: True labels
            
        Returns:
            Dictionary mapping protected attributes to their predictive parity differences
        """
        y_pred = self.model.predict(X)
        differences = {}
        
        for attr in self.protected_attributes:
            privileged_value = self.privileged_groups[attr]
            
            # Get masks for privileged and unprivileged groups
            privileged_mask = X[attr] == privileged_value
            unprivileged_mask = ~privileged_mask
            
            # Compute positive predictive values
            privileged_tp = np.sum((y_pred == 1) & (y_true == 1) & privileged_mask)
            privileged_p = np.sum((y_pred == 1) & privileged_mask)
            privileged_ppv = privileged_tp / privileged_p if privileged_p > 0 else 0
            
            unprivileged_tp = np.sum((y_pred == 1) & (y_true == 1) & unprivileged_mask)
            unprivileged_p = np.sum((y_pred == 1) & unprivileged_mask)
            unprivileged_ppv = unprivileged_tp / unprivileged_p if unprivileged_p > 0 else 0
            
            # Compute absolute difference
            difference = abs(privileged_ppv - unprivileged_ppv)
            differences[attr] = difference
            
        return differences
    
    def compute_auc_difference(
        self,
        X: pd.DataFrame,
        y_true: pd.Series
    ) -> Dict[str, float]:
        """
        Compute the AUC difference for each protected attribute.
        
        AUC difference is the absolute difference in area under the ROC curve
        between privileged and unprivileged groups. A value close to 0 indicates fairness.
        
        Args:
            X: Features DataFrame
            y_true: True labels
            
        Returns:
            Dictionary mapping protected attributes to their AUC differences
        """
        # Get probability predictions
        try:
            y_prob = self.model.predict_proba(X)[:, 1]
        except (AttributeError, IndexError):
            warnings.warn("Model does not support predict_proba or has unexpected output format. "
                         "AUC difference cannot be computed.")
            return {attr: 0.0 for attr in self.protected_attributes}
        
        differences = {}
        
        for attr in self.protected_attributes:
            privileged_value = self.privileged_groups[attr]
            
            # Get masks for privileged and unprivileged groups
            privileged_mask = X[attr] == privileged_value
            unprivileged_mask = ~privileged_mask
            
            # Compute AUC for each group
            try:
                privileged_auc = roc_auc_score(y_true[privileged_mask], y_prob[privileged_mask])
                unprivileged_auc = roc_auc_score(y_true[unprivileged_mask], y_prob[unprivileged_mask])
                
                # Compute absolute difference
                difference = abs(privileged_auc - unprivileged_auc)
                differences[attr] = difference
            except ValueError:
                # Handle case where one group has all same labels
                differences[attr] = 0.0
                
        return differences
    
    def compute_score_distribution_difference(
        self,
        X: pd.DataFrame,
        metric: str = 'ks'
    ) -> Dict[str, float]:
        """
        Compute the difference in score distributions between privileged and unprivileged groups.
        
        Args:
            X: Features DataFrame
            metric: Metric to use for comparison ('ks' for Kolmogorov-Smirnov statistic)
            
        Returns:
            Dictionary mapping protected attributes to their distribution differences
        """
        # Get probability predictions
        try:
            y_prob = self.model.predict_proba(X)[:, 1]
        except (AttributeError, IndexError):
            warnings.warn("Model does not support predict_proba or has unexpected output format. "
                         "Score distribution difference cannot be computed.")
            return {attr: 0.0 for attr in self.protected_attributes}
        
        differences = {}
        
        for attr in self.protected_attributes:
            privileged_value = self.privileged_groups[attr]
            
            # Get masks for privileged and unprivileged groups
            privileged_mask = X[attr] == privileged_value
            unprivileged_mask = ~privileged_mask
            
            # Get scores for each group
            privileged_scores = y_prob[privileged_mask]
            unprivileged_scores = y_prob[unprivileged_mask]
            
            if len(privileged_scores) == 0 or len(unprivileged_scores) == 0:
                differences[attr] = 0.0
                continue
                
            if metric == 'ks':
                # Compute Kolmogorov-Smirnov statistic
                ks_stat, _ = ks_2samp(privileged_scores, unprivileged_scores)
                differences[attr] = ks_stat
            else:
                # Default to mean difference
                differences[attr] = abs(np.mean(privileged_scores) - np.mean(unprivileged_scores))
                
        return differences
    
    def compute_calibration_difference(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Compute the calibration difference between privileged and unprivileged groups.
        
        Calibration difference measures how well the model's probability predictions
        align with actual outcomes across different groups. A value close to 0 indicates fairness.
        
        Args:
            X: Features DataFrame
            y_true: True labels
            n_bins: Number of bins for calibration analysis
            
        Returns:
            Dictionary mapping protected attributes to their calibration differences
        """
        # Get probability predictions
        try:
            y_prob = self.model.predict_proba(X)[:, 1]
        except (AttributeError, IndexError):
            warnings.warn("Model does not support predict_proba or has unexpected output format. "
                         "Calibration difference cannot be computed.")
            return {attr: 0.0 for attr in self.protected_attributes}
        
        differences = {}
        
        for attr in self.protected_attributes:
            privileged_value = self.privileged_groups[attr]
            
            # Get masks for privileged and unprivileged groups
            privileged_mask = X[attr] == privileged_value
            unprivileged_mask = ~privileged_mask
            
            # Get scores and labels for each group
            privileged_scores = y_prob[privileged_mask]
            privileged_labels = y_true[privileged_mask]
            
            unprivileged_scores = y_prob[unprivileged_mask]
            unprivileged_labels = y_true[unprivileged_mask]
            
            if len(privileged_scores) == 0 or len(unprivileged_scores) == 0:
                differences[attr] = 0.0
                continue
                
            # Compute calibration curves
            privileged_precision, privileged_recall, _ = precision_recall_curve(privileged_labels, privileged_scores)
            unprivileged_precision, unprivileged_recall, _ = precision_recall_curve(unprivileged_labels, unprivileged_scores)
            
            # Compute AUC of precision-recall curves
            privileged_auc = auc(privileged_recall, privileged_precision)
            unprivileged_auc = auc(unprivileged_recall, unprivileged_precision)
            
            # Compute absolute difference
            difference = abs(privileged_auc - unprivileged_auc)
            differences[attr] = difference
                
        return differences
    
    def analyze_feature_correlation(
        self,
        X: pd.DataFrame,
        threshold: float = 0.1
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze correlations between protected attributes and other features.
        
        Args:
            X: Features DataFrame
            threshold: Correlation threshold to flag potential issues
            
        Returns:
            Dictionary mapping protected attributes to their correlations with other features
        """
        correlations = {}
        
        for attr in self.protected_attributes:
            # Compute correlations with all other features
            other_features = [col for col in X.columns if col != attr]
            corr = X[other_features].corrwith(X[attr])
            
            # Filter correlations above threshold
            significant_corr = corr[abs(corr) > threshold]
            correlations[attr] = significant_corr.to_dict()
            
        return correlations
    
    def analyze_feature_importance_bias(
        self,
        X: pd.DataFrame,
        feature_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Analyze potential bias in feature importance based on protected attributes.
        
        Args:
            X: Features DataFrame
            feature_importance: Dictionary mapping feature names to their importance scores
            
        Returns:
            Dictionary mapping protected attributes to their correlation with feature importance
        """
        bias_scores = {}
        
        for attr in self.protected_attributes:
            # Get feature importance values
            importance_values = np.array([feature_importance.get(f, 0) for f in X.columns])
            
            # Compute correlation with protected attribute
            if attr in X.columns:
                attr_values = X[attr].values
                if np.issubdtype(attr_values.dtype, np.number):
                    # For numerical attributes, compute correlation
                    correlation = np.corrcoef(attr_values, importance_values)[0, 1]
                else:
                    # For categorical attributes, compute correlation with one-hot encoding
                    one_hot = pd.get_dummies(X[attr]).values
                    correlations = [np.corrcoef(one_hot[:, i], importance_values)[0, 1] 
                                   for i in range(one_hot.shape[1])]
                    correlation = max(map(abs, correlations))
            else:
                correlation = 0.0
                
            bias_scores[attr] = abs(correlation)
            
        return bias_scores
    
    def analyze_subgroup_performance(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1']
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Analyze model performance across different subgroups defined by protected attributes.
        
        Args:
            X: Features DataFrame
            y_true: True labels
            metrics: List of metrics to compute
            
        Returns:
            Nested dictionary mapping protected attributes to their values to metric scores
        """
        y_pred = self.model.predict(X)
        subgroup_performance = {}
        
        for attr in self.protected_attributes:
            attr_values = X[attr].unique()
            attr_performance = {}
            
            for value in attr_values:
                # Get mask for this subgroup
                mask = X[attr] == value
                
                if np.sum(mask) == 0:
                    continue
                    
                # Get predictions and labels for this subgroup
                subgroup_pred = y_pred[mask]
                subgroup_true = y_true[mask]
                
                # Compute confusion matrix
                tn, fp, fn, tp = confusion_matrix(subgroup_true, subgroup_pred).ravel()
                
                # Compute metrics
                metrics_dict = {}
                
                if 'accuracy' in metrics:
                    metrics_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
                    
                if 'precision' in metrics:
                    metrics_dict['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                    
                if 'recall' in metrics:
                    metrics_dict['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                if 'f1' in metrics:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    metrics_dict['f1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                attr_performance[str(value)] = metrics_dict
                
            subgroup_performance[attr] = attr_performance
            
        return subgroup_performance
    
    def plot_bias_metrics(
        self,
        metrics: Dict[str, Dict[str, float]],
        metric_name: str
    ) -> None:
        """
        Plot bias metrics for visualization.
        
        Args:
            metrics: Dictionary of metrics to plot
            metric_name: Name of the metric for the plot title
        """
        plt.figure(figsize=(10, 6))
        
        # Prepare data for plotting
        attributes = []
        values = []
        
        for attr, attr_metrics in metrics.items():
            for group, value in attr_metrics.items():
                attributes.append(f"{attr}: {group}")
                values.append(value)
        
        # Create bar plot
        plt.bar(range(len(attributes)), values)
        plt.xticks(range(len(attributes)), attributes, rotation=45, ha='right')
        plt.ylabel(metric_name)
        plt.title(f'Bias Metrics: {metric_name}')
        plt.tight_layout()
        plt.show()
    
    def plot_subgroup_performance(
        self,
        subgroup_performance: Dict[str, Dict[str, Dict[str, float]]],
        metric: str = 'accuracy'
    ) -> None:
        """
        Plot model performance across different subgroups.
        
        Args:
            subgroup_performance: Dictionary from analyze_subgroup_performance
            metric: Metric to plot
        """
        plt.figure(figsize=(12, 6))
        
        for attr, attr_performance in subgroup_performance.items():
            values = []
            labels = []
            
            for value, metrics_dict in attr_performance.items():
                if metric in metrics_dict:
                    values.append(metrics_dict[metric])
                    labels.append(f"{attr}: {value}")
            
            plt.bar(range(len(values)), values)
            plt.xticks(range(len(values)), labels, rotation=45, ha='right')
            
        plt.ylabel(metric.capitalize())
        plt.title(f'Model Performance by Subgroup: {metric.capitalize()}')
        plt.tight_layout()
        plt.show()
    
    def generate_bias_report(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        correlation_threshold: float = 0.1,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive bias analysis report.
        
        Args:
            X: Features DataFrame
            y_true: True labels (if None, uses model predictions)
            correlation_threshold: Threshold for feature correlation analysis
            feature_importance: Dictionary mapping feature names to their importance scores
            
        Returns:
            Dictionary containing all bias metrics
        """
        report = {
            'disparate_impact': self.compute_disparate_impact(X, y_true),
            'demographic_parity': self.compute_demographic_parity(X, y_true),
            'feature_correlations': self.analyze_feature_correlation(X, correlation_threshold)
        }
        
        if y_true is not None:
            report['equal_opportunity'] = self.compute_equal_opportunity(X, y_true)
            report['equalized_odds'] = self.compute_equalized_odds(X, y_true)
            report['treatment_equality'] = self.compute_treatment_equality(X, y_true)
            report['predictive_parity'] = self.compute_predictive_parity(X, y_true)
            
            try:
                report['auc_difference'] = self.compute_auc_difference(X, y_true)
                report['score_distribution_difference'] = self.compute_score_distribution_difference(X)
                report['calibration_difference'] = self.compute_calibration_difference(X, y_true)
                report['subgroup_performance'] = self.analyze_subgroup_performance(X, y_true)
            except Exception as e:
                warnings.warn(f"Some metrics could not be computed: {str(e)}")
        
        if feature_importance is not None:
            report['feature_importance_bias'] = self.analyze_feature_importance_bias(X, feature_importance)
        
        return report
    
    def plot_bias_report(self, report: Dict[str, Any]) -> None:
        """
        Generate visualizations for all bias metrics in the report.
        
        Args:
            report: Bias analysis report from generate_bias_report
        """
        # Plot disparate impact ratios
        self.plot_bias_metrics(report['disparate_impact'], 'Disparate Impact Ratio')
        
        # Plot demographic parity differences
        self.plot_bias_metrics(report['demographic_parity'], 'Demographic Parity Difference')
        
        # Plot equal opportunity differences if available
        if 'equal_opportunity' in report:
            self.plot_bias_metrics(report['equal_opportunity'], 'Equal Opportunity Difference')
        
        # Plot equalized odds differences if available
        if 'equalized_odds' in report:
            for attr, metrics in report['equalized_odds'].items():
                plt.figure(figsize=(10, 6))
                plt.bar(['TPR Difference', 'FPR Difference', 'Max Difference'], 
                       [metrics['tpr_difference'], metrics['fpr_difference'], metrics['max_difference']])
                plt.title(f'Equalized Odds Differences: {attr}')
                plt.ylabel('Difference')
                plt.tight_layout()
                plt.show()
        
        # Plot treatment equality ratios if available
        if 'treatment_equality' in report:
            self.plot_bias_metrics(report['treatment_equality'], 'Treatment Equality Ratio')
        
        # Plot predictive parity differences if available
        if 'predictive_parity' in report:
            self.plot_bias_metrics(report['predictive_parity'], 'Predictive Parity Difference')
        
        # Plot AUC differences if available
        if 'auc_difference' in report:
            self.plot_bias_metrics(report['auc_difference'], 'AUC Difference')
        
        # Plot score distribution differences if available
        if 'score_distribution_difference' in report:
            self.plot_bias_metrics(report['score_distribution_difference'], 'Score Distribution Difference (KS)')
        
        # Plot calibration differences if available
        if 'calibration_difference' in report:
            self.plot_bias_metrics(report['calibration_difference'], 'Calibration Difference')
        
        # Plot feature importance bias if available
        if 'feature_importance_bias' in report:
            self.plot_bias_metrics(report['feature_importance_bias'], 'Feature Importance Bias')
        
        # Plot subgroup performance if available
        if 'subgroup_performance' in report:
            self.plot_subgroup_performance(report['subgroup_performance'], 'accuracy')
            self.plot_subgroup_performance(report['subgroup_performance'], 'f1')
        
        # Plot feature correlations
        plt.figure(figsize=(12, 6))
        for attr, correlations in report['feature_correlations'].items():
            if correlations:
                plt.plot(list(correlations.keys()), list(correlations.values()), 
                        marker='o', label=attr)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Correlation Coefficient')
        plt.title('Feature Correlations with Protected Attributes')
        plt.legend()
        plt.tight_layout()
        plt.show() 