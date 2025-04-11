"""
Report Generator module for AI Model Governance Toolkit.
Generates comprehensive reports combining explainability, bias analysis, and regulatory compliance information.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Union, Optional, Any, Tuple
from jinja2 import Environment, FileSystemLoader
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ReportGenerator:
    """
    A class for generating comprehensive reports for AI model governance.
    
    This generator combines explainability insights, bias analysis, and regulatory compliance
    information into a single, comprehensive report that can be exported in various formats.
    
    Attributes:
        model_name: Name of the model being analyzed
        model_version: Version of the model
        model_type: Type of model (e.g., 'classification', 'regression')
        report_dir: Directory to save reports
        template_dir: Directory containing report templates
    """
    
    def __init__(
        self,
        model_name: str,
        model_version: str,
        model_type: str,
        report_dir: str = "reports",
        template_dir: str = "report_generator/templates"
    ):
        """
        Initialize the ReportGenerator.
        
        Args:
            model_name: Name of the model being analyzed
            model_version: Version of the model
            model_type: Type of model (e.g., 'classification', 'regression')
            report_dir: Directory to save reports
            template_dir: Directory containing report templates
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model_type = model_type
        self.report_dir = report_dir
        self.template_dir = template_dir
        
        # Create report directory if it doesn't exist
        os.makedirs(report_dir, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(loader=FileSystemLoader(template_dir))
        
        # Initialize report data
        self.report_data = {
            "model_info": {
                "name": model_name,
                "version": model_version,
                "type": model_type,
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "explainability": {},
            "bias_analysis": {},
            "performance_metrics": {},
            "regulatory_compliance": {},
            "recommendations": []
        }
    
    def add_explainability_data(
        self,
        feature_importance: Dict[str, float],
        shap_values: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        sample_data: Optional[pd.DataFrame] = None,
        local_explanations: Optional[Dict[int, Dict[str, float]]] = None
    ) -> None:
        """
        Add explainability data to the report.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            shap_values: SHAP values for the dataset (optional)
            feature_names: List of feature names (optional)
            sample_data: Sample data used for explanations (optional)
            local_explanations: Dictionary mapping sample indices to local explanations (optional)
        """
        self.report_data["explainability"] = {
            "feature_importance": feature_importance,
            "has_shap_values": shap_values is not None,
            "feature_names": feature_names,
            "has_sample_data": sample_data is not None,
            "has_local_explanations": local_explanations is not None
        }
        
        # Generate feature importance plot
        if feature_importance:
            self._generate_feature_importance_plot(feature_importance)
    
    def add_bias_analysis_data(
        self,
        bias_report: Dict[str, Any],
        protected_attributes: List[str],
        privileged_groups: Dict[str, Union[str, int, float]]
    ) -> None:
        """
        Add bias analysis data to the report.
        
        Args:
            bias_report: Comprehensive bias report from BiasDetector
            protected_attributes: List of protected attributes analyzed
            privileged_groups: Dictionary mapping protected attributes to their privileged values
        """
        self.report_data["bias_analysis"] = {
            "bias_report": bias_report,
            "protected_attributes": protected_attributes,
            "privileged_groups": privileged_groups
        }
        
        # Generate bias metrics plots
        self._generate_bias_metrics_plots(bias_report)
    
    def add_performance_metrics(
        self,
        metrics: Dict[str, float],
        confusion_matrix: Optional[np.ndarray] = None,
        roc_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        precision_recall_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Add model performance metrics to the report.
        
        Args:
            metrics: Dictionary mapping metric names to values
            confusion_matrix: Confusion matrix (optional)
            roc_curve: Tuple of (fpr, tpr, thresholds) for ROC curve (optional)
            precision_recall_curve: Tuple of (precision, recall, thresholds) for precision-recall curve (optional)
        """
        self.report_data["performance_metrics"] = {
            "metrics": metrics,
            "has_confusion_matrix": confusion_matrix is not None,
            "has_roc_curve": roc_curve is not None,
            "has_precision_recall_curve": precision_recall_curve is not None
        }
        
        # Generate performance plots
        if confusion_matrix is not None:
            self._generate_confusion_matrix_plot(confusion_matrix)
        
        if roc_curve is not None:
            self._generate_roc_curve_plot(roc_curve)
        
        if precision_recall_curve is not None:
            self._generate_precision_recall_plot(precision_recall_curve)
    
    def add_regulatory_compliance(
        self,
        compliance_data: Dict[str, Dict[str, bool]],
        regulations: List[str]
    ) -> None:
        """
        Add regulatory compliance data to the report.
        
        Args:
            compliance_data: Dictionary mapping regulations to compliance requirements and their status
            regulations: List of regulations being checked
        """
        self.report_data["regulatory_compliance"] = {
            "compliance_data": compliance_data,
            "regulations": regulations
        }
        
        # Generate compliance summary
        self._generate_compliance_summary(compliance_data, regulations)
    
    def add_recommendations(
        self,
        recommendations: List[Dict[str, str]]
    ) -> None:
        """
        Add recommendations to the report.
        
        Args:
            recommendations: List of dictionaries with recommendation details
        """
        self.report_data["recommendations"] = recommendations
    
    def _generate_feature_importance_plot(self, feature_importance: Dict[str, float]) -> None:
        """
        Generate feature importance plot and save as base64 string.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
        """
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Create plot
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=[item[1] for item in sorted_features],
                y=[item[0] for item in sorted_features],
                orientation='h',
                marker_color='rgba(55, 83, 109, 0.7)'
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=max(400, len(sorted_features) * 20),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Save as base64
        self.report_data["explainability"]["feature_importance_plot"] = self._fig_to_base64(fig)
    
    def _generate_bias_metrics_plots(self, bias_report: Dict[str, Any]) -> None:
        """
        Generate plots for bias metrics and save as base64 strings.
        
        Args:
            bias_report: Comprehensive bias report from BiasDetector
        """
        plots = {}
        
        # Plot disparate impact ratios
        if "disparate_impact" in bias_report:
            fig = go.Figure()
            
            for attr, ratio in bias_report["disparate_impact"].items():
                fig.add_trace(
                    go.Bar(
                        name=attr,
                        x=[attr],
                        y=[ratio],
                        marker_color='rgba(55, 83, 109, 0.7)'
                    )
                )
            
            fig.update_layout(
                title="Disparate Impact Ratios",
                xaxis_title="Protected Attribute",
                yaxis_title="Ratio",
                height=400,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            plots["disparate_impact"] = self._fig_to_base64(fig)
        
        # Plot demographic parity differences
        if "demographic_parity" in bias_report:
            fig = go.Figure()
            
            for attr, diff in bias_report["demographic_parity"].items():
                fig.add_trace(
                    go.Bar(
                        name=attr,
                        x=[attr],
                        y=[diff],
                        marker_color='rgba(55, 83, 109, 0.7)'
                    )
                )
            
            fig.update_layout(
                title="Demographic Parity Differences",
                xaxis_title="Protected Attribute",
                yaxis_title="Difference",
                height=400,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            plots["demographic_parity"] = self._fig_to_base64(fig)
        
        # Plot equal opportunity differences
        if "equal_opportunity" in bias_report:
            fig = go.Figure()
            
            for attr, diff in bias_report["equal_opportunity"].items():
                fig.add_trace(
                    go.Bar(
                        name=attr,
                        x=[attr],
                        y=[diff],
                        marker_color='rgba(55, 83, 109, 0.7)'
                    )
                )
            
            fig.update_layout(
                title="Equal Opportunity Differences",
                xaxis_title="Protected Attribute",
                yaxis_title="Difference",
                height=400,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            plots["equal_opportunity"] = self._fig_to_base64(fig)
        
        # Plot feature correlations
        if "feature_correlations" in bias_report:
            fig = go.Figure()
            
            for attr, correlations in bias_report["feature_correlations"].items():
                for feature, corr in correlations.items():
                    fig.add_trace(
                        go.Bar(
                            name=f"{attr}: {feature}",
                            x=[f"{attr}: {feature}"],
                            y=[corr],
                            marker_color='rgba(55, 83, 109, 0.7)'
                        )
                    )
            
            fig.update_layout(
                title="Feature Correlations with Protected Attributes",
                xaxis_title="Feature",
                yaxis_title="Correlation Coefficient",
                height=max(400, len(plots) * 20),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            plots["feature_correlations"] = self._fig_to_base64(fig)
        
        # Add plots to report data
        self.report_data["bias_analysis"]["plots"] = plots
    
    def _generate_confusion_matrix_plot(self, confusion_matrix: np.ndarray) -> None:
        """
        Generate confusion matrix plot and save as base64 string.
        
        Args:
            confusion_matrix: Confusion matrix
        """
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        
        # Update layout
        fig.update_layout(
            title="Confusion Matrix",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Save as base64
        self.report_data["performance_metrics"]["confusion_matrix_plot"] = self._fig_to_base64(fig)
    
    def _generate_roc_curve_plot(self, roc_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Generate ROC curve plot and save as base64 string.
        
        Args:
            roc_curve: Tuple of (fpr, tpr, thresholds) for ROC curve
        """
        fpr, tpr, _ = roc_curve
        
        # Create plot
        fig = go.Figure()
        
        # Add ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name='ROC Curve',
                line=dict(color='rgba(55, 83, 109, 0.7)')
            )
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='rgba(200, 200, 200, 0.7)', dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Save as base64
        self.report_data["performance_metrics"]["roc_curve_plot"] = self._fig_to_base64(fig)
    
    def _generate_precision_recall_plot(self, precision_recall_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Generate precision-recall curve plot and save as base64 string.
        
        Args:
            precision_recall_curve: Tuple of (precision, recall, thresholds) for precision-recall curve
        """
        precision, recall, _ = precision_recall_curve
        
        # Create plot
        fig = go.Figure()
        
        # Add precision-recall curve
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode='lines',
                name='Precision-Recall Curve',
                line=dict(color='rgba(55, 83, 109, 0.7)')
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Save as base64
        self.report_data["performance_metrics"]["precision_recall_plot"] = self._fig_to_base64(fig)
    
    def _generate_compliance_summary(
        self,
        compliance_data: Dict[str, Dict[str, bool]],
        regulations: List[str]
    ) -> None:
        """
        Generate compliance summary and save as base64 string.
        
        Args:
            compliance_data: Dictionary mapping regulations to compliance requirements and their status
            regulations: List of regulations being checked
        """
        # Create summary data
        summary_data = []
        
        for regulation in regulations:
            if regulation in compliance_data:
                for requirement, status in compliance_data[regulation].items():
                    summary_data.append({
                        "regulation": regulation,
                        "requirement": requirement,
                        "status": "Compliant" if status else "Non-Compliant"
                    })
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Create plot
        fig = go.Figure(data=[
            go.Table(
                header=dict(
                    values=["Regulation", "Requirement", "Status"],
                    fill_color='rgba(55, 83, 109, 0.7)',
                    align='left'
                ),
                cells=dict(
                    values=[df["regulation"], df["requirement"], df["status"]],
                    fill_color=[
                        ['rgba(240, 240, 240, 0.7)'] * len(df),
                        ['rgba(240, 240, 240, 0.7)'] * len(df),
                        [['rgba(0, 255, 0, 0.2)' if s == "Compliant" else 'rgba(255, 0, 0, 0.2)' for s in df["status"]]]
                    ],
                    align='left'
                )
            )
        ])
        
        # Update layout
        fig.update_layout(
            title="Regulatory Compliance Summary",
            height=max(400, len(summary_data) * 30),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Save as base64
        self.report_data["regulatory_compliance"]["compliance_summary_plot"] = self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig: go.Figure) -> str:
        """
        Convert a Plotly figure to base64 string.
        
        Args:
            fig: Plotly figure
            
        Returns:
            Base64 string representation of the figure
        """
        # Convert to HTML
        html = fig.to_html(include_plotlyjs=False, full_html=False)
        
        # Encode as base64
        return base64.b64encode(html.encode()).decode()
    
    def generate_html_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate an HTML report.
        
        Args:
            output_file: Path to save the HTML report (optional)
            
        Returns:
            HTML report as a string
        """
        # Load template
        template = self.env.get_template("report_template.html")
        
        # Render template
        html = template.render(report_data=self.report_data)
        
        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(html)
        
        return html
    
    def generate_markdown_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a Markdown report.
        
        Args:
            output_file: Path to save the Markdown report (optional)
            
        Returns:
            Markdown report as a string
        """
        # Load template
        template = self.env.get_template("report_template.md")
        
        # Render template
        markdown = template.render(report_data=self.report_data)
        
        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(markdown)
        
        return markdown
    
    def generate_json_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a JSON report.
        
        Args:
            output_file: Path to save the JSON report (optional)
            
        Returns:
            JSON report as a string
        """
        # Convert report data to JSON
        json_report = json.dumps(self.report_data, indent=2)
        
        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(json_report)
        
        return json_report
    
    def generate_pdf_report(self, output_file: Optional[str] = None) -> bytes:
        """
        Generate a PDF report.
        
        Args:
            output_file: Path to save the PDF report (optional)
            
        Returns:
            PDF report as bytes
        """
        # This is a placeholder for PDF generation
        # In a real implementation, you would use a library like WeasyPrint or ReportLab
        # to convert HTML to PDF
        
        # For now, we'll just return an empty bytes object
        pdf_bytes = b""
        
        # Save to file if specified
        if output_file:
            with open(output_file, "wb") as f:
                f.write(pdf_bytes)
        
        return pdf_bytes 