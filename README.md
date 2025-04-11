# AI Model Governance and FinTech ML Toolkit

An open-source toolkit for ensuring transparency, fairness, and compliance in AI/ML models used in financial applications.

## Overview

This toolkit provides a comprehensive suite of tools for:
- Model explainability and interpretability
- Bias detection and fairness metrics
- Regulatory compliance reporting
- Performance monitoring and validation

## Components

### 1. Explainability Engine
- SHAP-based feature importance analysis
- Local and global explanations
- Interactive visualizations
- Support for various model types (classification, regression)

### 2. Bias Detection Module
- Disparate impact analysis
- Demographic parity metrics
- Equal opportunity measurements
- Feature correlation analysis
- Comprehensive bias reporting

### 3. Report Generator
- HTML, Markdown, and JSON report formats
- Interactive visualizations using Plotly
- Comprehensive model documentation
- Regulatory compliance tracking
- Customizable templates
- Performance metrics visualization
- Bias analysis reporting
- Recommendations generation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Explainability Analysis
```python
from explainability.shap_explainer import ShapExplainer

# Initialize explainer
explainer = ShapExplainer(model)

# Get feature importance
feature_importance = explainer.explain_feature_importance(X_train)

# Get SHAP values for a specific prediction
shap_values = explainer.explain_prediction(X_test.iloc[0:1])
```

### Bias Detection
```python
from bias_detection.fairness_metrics import BiasDetector

# Initialize bias detector
bias_detector = BiasDetector(
    model=model,
    protected_attributes=['gender', 'age'],
    privileged_groups={'gender': 'M', 'age': 25}
)

# Generate comprehensive bias report
bias_report = bias_detector.generate_bias_report(X_test, y_test)

# Visualize bias metrics
bias_detector.plot_bias_report(bias_report)
```

### Report Generation
```python
from report_generator.report_generator import ReportGenerator

# Initialize report generator
report_gen = ReportGenerator(
    model_info={
        'name': 'Credit Scoring Model',
        'version': '1.0.0',
        'type': 'Random Forest Classifier'
    }
)

# Add explainability data
report_gen.add_explainability_data(
    feature_importance=feature_importance,
    shap_values=shap_values,
    has_shap_values=True,
    has_local_explanations=True
)

# Add bias analysis data
report_gen.add_bias_analysis_data(
    bias_report=bias_report,
    protected_attributes=['gender', 'age'],
    privileged_groups={'gender': 'M', 'age': 25}
)

# Add performance metrics
report_gen.add_performance_metrics(
    metrics={
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.86,
        'f1_score': 0.84,
        'roc_auc': 0.92
    }
)

# Add regulatory compliance data
report_gen.add_regulatory_compliance(
    compliance_data={
        'EU AI Act': {
            'transparency': True,
            'fairness': True,
            'accuracy': True
        },
        'GDPR': {
            'explainability': True,
            'fairness': True,
            'accuracy': True
        }
    },
    regulations=['EU AI Act', 'GDPR']
)

# Add recommendations
report_gen.add_recommendations([
    {
        'title': 'Improve Model Performance',
        'description': 'Consider feature engineering to improve accuracy',
        'priority': 'High'
    }
])

# Generate reports
html_report = report_gen.generate_html_report()
markdown_report = report_gen.generate_markdown_report()
json_report = report_gen.generate_json_report()
```

## Demo Notebooks

1. `notebooks/demo_explainability.ipynb`: Demonstrates model explainability features
2. `notebooks/demo_bias_detection.ipynb`: Shows bias detection and analysis
3. `notebooks/demo_report_generator.ipynb`: Illustrates report generation capabilities

## Regulatory Compliance

This toolkit helps ensure compliance with:
- EU AI Act
- Fair Lending Laws
- GDPR
- CFPB Guidelines
- Local Financial Regulations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 