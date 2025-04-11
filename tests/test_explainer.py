"""
Tests for the ModelExplainer class.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from explainability.shap_explainer import ModelExplainer


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    }
    X = pd.DataFrame(data)
    y = (X['feature1'] + X['feature2'] > 0).astype(int)
    
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Train a simple model for testing."""
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


def test_model_explainer_initialization(trained_model, sample_data):
    """Test ModelExplainer initialization."""
    X, _ = sample_data
    explainer = ModelExplainer(trained_model, X)
    
    assert explainer.model == trained_model
    assert explainer.feature_names == ['feature1', 'feature2', 'feature3']
    assert explainer.background_data.shape == X.shape


def test_feature_importance(trained_model, sample_data):
    """Test feature importance computation."""
    X, _ = sample_data
    explainer = ModelExplainer(trained_model, X)
    
    importance = explainer.get_feature_importance(n_samples=50, plot=False)
    
    assert isinstance(importance, dict)
    assert len(importance) == 3
    assert all(isinstance(v, float) for v in importance.values())
    assert all(v >= 0 for v in importance.values())


def test_local_explanation(trained_model, sample_data):
    """Test local explanation generation."""
    X, _ = sample_data
    explainer = ModelExplainer(trained_model, X)
    
    sample = X.iloc[0:1]
    contributions = explainer.explain_prediction(sample, plot=False)
    
    assert isinstance(contributions, dict)
    assert len(contributions) == 3
    assert all(isinstance(v, float) for v in contributions.values())


def test_custom_feature_names(trained_model, sample_data):
    """Test custom feature names."""
    X, _ = sample_data
    custom_names = ['custom1', 'custom2', 'custom3']
    
    explainer = ModelExplainer(
        trained_model,
        X,
        feature_names=custom_names
    )
    
    assert explainer.feature_names == custom_names


def test_empty_feature_names(trained_model, sample_data):
    """Test automatic feature name generation."""
    X, _ = sample_data
    X.columns = [0, 1, 2]  # Numeric column names
    
    explainer = ModelExplainer(trained_model, X)
    
    assert explainer.feature_names == ['Feature_0', 'Feature_1', 'Feature_2'] 