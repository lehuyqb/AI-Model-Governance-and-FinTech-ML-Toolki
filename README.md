# AI Model Governance Toolkit for FinTech

An open-source toolkit for ensuring transparency, fairness, and compliance in AI/ML models used in financial applications.

## Overview

This toolkit provides a comprehensive set of tools for:
- Model explainability and interpretability
- Bias detection and fairness metrics
- Regulatory compliance reporting
- Model monitoring and governance

## Components

### 1. Explainability Engine
- SHAP-based feature importance analysis
- LIME explanations for individual predictions
- Counterfactual explanations
- Feature interaction analysis
- Decision path visualization

### 2. Bias Detection Module
- Comprehensive fairness metrics:
  - Disparate Impact Analysis
  - Demographic Parity
  - Equal Opportunity
  - Equalized Odds
  - Treatment Equality
  - Predictive Parity
  - AUC Difference
  - Score Distribution Difference
  - Calibration Difference
- Feature correlation analysis
- Feature importance bias detection
- Subgroup performance analysis
- Interactive visualizations
- Regulatory compliance reporting

### 3. Report Generator
- Automated compliance documentation
- Model performance reports
- Bias analysis summaries
- Regulatory requirement mapping
- Audit trail generation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Explainability

```python
from explainability.shap_explainer import ShapExplainer

# Initialize explainer
explainer = ShapExplainer(model)

# Get global feature importance
importance = explainer.explain_global(X)

# Get local explanations
local_explanations = explainer.explain_local(X)
```

### Bias Detection

```python
from bias_detection.fairness_metrics import BiasDetector

# Initialize bias detector
detector = BiasDetector(
    model=model,
    protected_attributes=['gender', 'age'],
    privileged_groups={'gender': 'male', 'age': 25}
)

# Generate comprehensive bias report
report = detector.generate_bias_report(
    X=X_test,
    y_true=y_test,
    correlation_threshold=0.1,
    feature_importance=feature_importance
)

# Visualize bias metrics
detector.plot_bias_report(report)
```

## Regulatory Compliance

This toolkit helps ensure compliance with:
- EU AI Act
- Fair Lending Laws
- GDPR
- CFPB Guidelines
- Local Financial Regulations

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 