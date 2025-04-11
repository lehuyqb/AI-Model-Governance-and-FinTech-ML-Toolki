"""
Tests for the BiasDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bias_detection.fairness_metrics import BiasDetector


@pytest.fixture
def sample_data():
    """Generate sample data with protected attributes for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'gender': np.random.choice(['male', 'female'], size=n_samples),
        'age': np.random.normal(35, 10, n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples)
    }
    X = pd.DataFrame(data)
    
    # Generate target with some bias
    prob = 1 / (1 + np.exp(-(
        0.1 * X['feature1'] +
        0.1 * X['feature2'] +
        0.1 * (X['gender'] == 'male').astype(int)
    )))
    y = (prob > 0.5).astype(int)
    
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Train a simple model for testing."""
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def bias_detector(trained_model):
    """Create a BiasDetector instance for testing."""
    protected_attributes = ['gender', 'age']
    privileged_groups = {
        'gender': 'male',
        'age': 35
    }
    return BiasDetector(trained_model, protected_attributes, privileged_groups)


def test_bias_detector_initialization(bias_detector, trained_model):
    """Test BiasDetector initialization."""
    assert bias_detector.model == trained_model
    assert bias_detector.protected_attributes == ['gender', 'age']
    assert bias_detector.privileged_groups == {'gender': 'male', 'age': 35}


def test_compute_disparate_impact(bias_detector, sample_data):
    """Test disparate impact computation."""
    X, y = sample_data
    ratios = bias_detector.compute_disparate_impact(X, y)
    
    assert isinstance(ratios, dict)
    assert 'gender' in ratios
    assert 'age' in ratios
    assert all(isinstance(v, float) for v in ratios.values())
    assert all(v >= 0 for v in ratios.values())


def test_compute_demographic_parity(bias_detector, sample_data):
    """Test demographic parity computation."""
    X, y = sample_data
    differences = bias_detector.compute_demographic_parity(X, y)
    
    assert isinstance(differences, dict)
    assert 'gender' in differences
    assert 'age' in differences
    assert all(isinstance(v, float) for v in differences.values())
    assert all(v >= 0 for v in differences.values())


def test_compute_equal_opportunity(bias_detector, sample_data):
    """Test equal opportunity computation."""
    X, y = sample_data
    differences = bias_detector.compute_equal_opportunity(X, y)
    
    assert isinstance(differences, dict)
    assert 'gender' in differences
    assert 'age' in differences
    assert all(isinstance(v, float) for v in differences.values())
    assert all(v >= 0 for v in differences.values())


def test_analyze_feature_correlation(bias_detector, sample_data):
    """Test feature correlation analysis."""
    X, _ = sample_data
    correlations = bias_detector.analyze_feature_correlation(X)
    
    assert isinstance(correlations, dict)
    assert 'gender' in correlations
    assert 'age' in correlations
    assert all(isinstance(v, dict) for v in correlations.values())


def test_generate_bias_report(bias_detector, sample_data):
    """Test comprehensive bias report generation."""
    X, y = sample_data
    report = bias_detector.generate_bias_report(X, y)
    
    assert isinstance(report, dict)
    assert 'disparate_impact' in report
    assert 'demographic_parity' in report
    assert 'feature_correlations' in report
    assert 'equal_opportunity' in report
    
    # Check report structure
    assert isinstance(report['disparate_impact'], dict)
    assert isinstance(report['demographic_parity'], dict)
    assert isinstance(report['feature_correlations'], dict)
    assert isinstance(report['equal_opportunity'], dict)


def test_empty_protected_attributes():
    """Test BiasDetector with empty protected attributes."""
    model = RandomForestClassifier()
    with pytest.raises(ValueError):
        BiasDetector(model, [], {})


def test_mismatched_privileged_groups():
    """Test BiasDetector with mismatched privileged groups."""
    model = RandomForestClassifier()
    with pytest.raises(ValueError):
        BiasDetector(
            model,
            protected_attributes=['gender', 'age'],
            privileged_groups={'gender': 'male'}  # Missing age
        ) 