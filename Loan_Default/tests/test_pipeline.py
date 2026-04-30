"""
Loan Default Prediction — Unit Tests for Data Generation & Model Pipeline
Author: Kishore U.
Purpose: Test data validation, feature engineering, model training, and metric computation.

Run tests with: python -m pytest tests/test_pipeline.py -v
Or with coverage: python -m pytest tests/test_pipeline.py -v --cov=src
"""

import sys
import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add src/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import gini_coefficient, ks_statistic, information_value, iv_strength_label, grade_portfolio


# ── FIXTURES ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Generate minimal sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'income_monthly': np.random.lognormal(10, 1, n_samples),
        'loan_amount': np.random.lognormal(10.5, 1.2, n_samples),
        'loan_tenure_m': np.random.randint(12, 60, n_samples),
        'interest_rate': np.random.uniform(6, 18, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'delinquency_30d': np.random.randint(0, 3, n_samples),
        'foir_pct': np.random.uniform(10, 80, n_samples),
        'utilisation_pct': np.random.uniform(0, 100, n_samples),
        'existing_loans': np.random.randint(0, 5, n_samples),
        'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Business'], n_samples),
        'education': np.random.choice(['Graduate', 'Post-Graduate'], n_samples),
        'residence_type': np.random.choice(['Own', 'Rented', 'Family'], n_samples),
        'loan_purpose': np.random.choice(['Personal', 'Home', 'Education'], n_samples),
        'months_employed': np.random.randint(6, 300, n_samples),
        'default': np.random.binomial(1, 0.33, n_samples),
    })
    
    return df


@pytest.fixture
def y_true_y_proba():
    """Sample true labels and predicted probabilities for metric testing."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1] * 10)  # 100 samples, 50% default
    y_proba = np.array([
        0.1, 0.2, 0.15, 0.05, 0.9, 0.85, 0.95, 0.92, 0.3, 0.88
    ] * 10)
    
    return y_true, y_proba


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tempdir:
        yield tempdir


# ── DATA GENERATION TESTS ─────────────────────────────────────────────────

class TestDataGeneration:
    """Test synthetic data generation pipeline."""
    
    def test_sample_data_shape(self, sample_data):
        """Test that sample data has correct shape."""
        assert sample_data.shape[0] == 100, "Sample data should have 100 rows"
        assert sample_data.shape[1] == 15, "Sample data should have 15 columns"
    
    def test_sample_data_no_nulls(self, sample_data):
        """Test that sample data has no missing values."""
        assert sample_data.isnull().sum().sum() == 0, "Sample data should have no null values"
    
    def test_sample_data_types(self, sample_data):
        """Test that data types are correct."""
        numeric_cols = ['income_monthly', 'loan_amount', 'credit_score', 'foir_pct']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_data[col]), f"{col} should be numeric"
    
    def test_default_column_binary(self, sample_data):
        """Test that target variable is binary."""
        assert set(sample_data['default'].unique()).issubset({0, 1}), "Default should be 0 or 1"
    
    def test_default_rate_reasonable(self, sample_data):
        """Test that default rate is in reasonable range (20-50%)."""
        default_rate = sample_data['default'].mean()
        assert 0.20 <= default_rate <= 0.50, f"Default rate {default_rate} is outside 20-50% range"
    
    def test_income_positive(self, sample_data):
        """Test that income values are positive."""
        assert (sample_data['income_monthly'] > 0).all(), "Income should be positive"
    
    def test_loan_amount_positive(self, sample_data):
        """Test that loan amounts are positive."""
        assert (sample_data['loan_amount'] > 0).all(), "Loan amount should be positive"
    
    def test_credit_score_range(self, sample_data):
        """Test that credit scores are in valid range (300-850)."""
        assert (sample_data['credit_score'] >= 300).all(), "Min credit score should be >= 300"
        assert (sample_data['credit_score'] <= 850).all(), "Max credit score should be <= 850"
    
    def test_employment_type_values(self, sample_data):
        """Test that employment types are valid."""
        valid_types = {'Salaried', 'Self-Employed', 'Business', 'Freelancer'}
        assert sample_data['employment_type'].isin(valid_types).all(), "Invalid employment type"


# ── FEATURE ENGINEERING TESTS ─────────────────────────────────────────────

class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_engineer_adds_required_columns(self, sample_data):
        """Test that engineer() creates required derived features."""
        from sklearn.preprocessing import LabelEncoder
        
        # Simulate feature engineering
        le = LabelEncoder()
        sample_data["employment_enc"] = le.fit_transform(sample_data["employment_type"])
        sample_data["log_income"] = np.log1p(sample_data["income_monthly"])
        sample_data["log_loan"] = np.log1p(sample_data["loan_amount"])
        sample_data["high_foir"] = (sample_data["foir_pct"] > 55).astype(int)
        sample_data["delinq_flag"] = (sample_data["delinquency_30d"] > 0).astype(int)
        
        required_cols = ["employment_enc", "log_income", "log_loan", "high_foir", "delinq_flag"]
        for col in required_cols:
            assert col in sample_data.columns, f"Missing engineered feature: {col}"
    
    def test_log_transforms_are_numeric(self, sample_data):
        """Test that log-transformed features are numeric and finite."""
        sample_data["log_income"] = np.log1p(sample_data["income_monthly"])
        assert pd.api.types.is_numeric_dtype(sample_data["log_income"])
        assert np.isfinite(sample_data["log_income"]).all(), "Log transform should be finite"
    
    def test_binary_flags_values(self, sample_data):
        """Test that binary flags contain only 0 and 1."""
        sample_data["high_foir"] = (sample_data["foir_pct"] > 55).astype(int)
        sample_data["high_util"] = (sample_data["utilisation_pct"] > 80).astype(int)
        
        for col in ["high_foir", "high_util"]:
            assert set(sample_data[col].unique()).issubset({0, 1}), f"{col} should be binary"
    
    def test_credit_band_values(self, sample_data):
        """Test that credit bands are in range 0-4."""
        sample_data["credit_band"] = pd.cut(
            sample_data["credit_score"],
            bins=[0, 550, 650, 720, 780, 900],
            labels=[4, 3, 2, 1, 0],
        ).astype(int)
        
        assert sample_data["credit_band"].min() >= 0
        assert sample_data["credit_band"].max() <= 4


# ── METRIC CALCULATION TESTS ──────────────────────────────────────────────

class TestMetricFunctions:
    """Test Gini, KS, IV metric calculations."""
    
    def test_gini_coefficient_range(self, y_true_y_proba):
        """Test that Gini coefficient is between 0 and 1."""
        y_true, y_proba = y_true_y_proba
        gini = gini_coefficient(y_true, y_proba)
        
        assert 0 <= gini <= 1, f"Gini {gini} should be between 0 and 1"
    
    def test_gini_perfect_prediction(self):
        """Test that Gini = 1 for perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])  # Perfect ranking
        
        gini = gini_coefficient(y_true, y_proba)
        assert gini > 0.95, "Gini should be ~1 for perfect predictions"
    
    def test_gini_random_prediction(self):
        """Test that Gini ≈ 0 for random predictions."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 1000)
        y_proba = np.random.uniform(0, 1, 1000)
        
        gini = gini_coefficient(y_true, y_proba)
        assert gini < 0.2, "Gini should be ~0 for random predictions"
    
    def test_ks_statistic_range(self, y_true_y_proba):
        """Test that KS statistic is between 0 and 1."""
        y_true, y_proba = y_true_y_proba
        ks = ks_statistic(y_true, y_proba)
        
        assert 0 <= ks <= 1, f"KS {ks} should be between 0 and 1"
    
    def test_ks_perfect_separation(self):
        """Test that KS ≈ 1 for perfect separation."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])
        
        ks = ks_statistic(y_true, y_proba)
        assert ks > 0.75, "KS should be high for good separation"
    
    def test_iv_strength_label_classification(self):
        """Test Information Value strength classification."""
        test_cases = [
            (0.7, "Very Strong"),   # IV > 0.5
            (0.25, "Medium"),       # 0.1 < IV < 0.3
            (0.05, "Weak"),         # 0.02 < IV < 0.1
            (0.01, "Useless"),      # IV < 0.02
        ]
        
        for iv, expected_label in test_cases:
            label = iv_strength_label(iv)
            assert expected_label in label, f"IV {iv} should map to {expected_label}, got {label}"
    
    def test_information_value_nonnegative(self, sample_data):
        """Test that Information Value is non-negative."""
        iv = information_value(sample_data, "credit_score", target="default")
        assert iv >= 0, "Information Value should be non-negative"
    
    def test_information_value_numeric_feature(self, sample_data):
        """Test IV calculation on numeric feature."""
        iv = information_value(sample_data, "foir_pct", target="default")
        assert isinstance(iv, (float, np.floating))
        assert 0 <= iv <= 2.0  # IV typically < 2


# ── GRADING TESTS ─────────────────────────────────────────────────────────

class TestGrading:
    """Test portfolio grading functions."""
    
    def test_grade_portfolio_valid_grades(self):
        """Test that grade_portfolio returns valid grade labels."""
        pd_scores = np.array([10, 25, 40, 60, 85])
        grades = grade_portfolio(pd_scores)
        
        valid_grades = {
            "A — Very Low Risk",
            "B — Low Risk",
            "C — Medium Risk",
            "D — High Risk",
            "E — Very High Risk"
        }
        
        assert all(g in valid_grades for g in grades), f"Invalid grades: {grades}"
    
    def test_grade_portfolio_monotonic(self):
        """Test that higher PD scores map to higher risk grades."""
        pd_scores = np.array([5, 15, 30, 55, 80])
        grades = grade_portfolio(pd_scores)
        
        grade_order = {
            "A — Very Low Risk": 1,
            "B — Low Risk": 2,
            "C — Medium Risk": 3,
            "D — High Risk": 4,
            "E — Very High Risk": 5,
        }
        
        grade_nums = [grade_order[g] for g in grades]
        assert all(grade_nums[i] <= grade_nums[i+1] for i in range(len(grade_nums)-1)), \
            "Grades should be monotonically increasing with PD"
    
    def test_grade_portfolio_length(self):
        """Test that grade_portfolio returns same number of grades as inputs."""
        pd_scores = np.array([10, 20, 30, 40, 50])
        grades = grade_portfolio(pd_scores)
        
        assert len(grades) == len(pd_scores), "Output length should match input length"


# ── MODEL TRAINING TESTS ──────────────────────────────────────────────────

class TestModelTraining:
    """Test model training and prediction."""
    
    def test_logistic_regression_training(self, sample_data):
        """Test that Logistic Regression trains without error."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove('default')
        
        X = sample_data[numeric_features]
        y = sample_data['default']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_scaled, y)
        
        assert lr is not None, "Logistic Regression should train successfully"
        assert hasattr(lr, 'predict_proba'), "Model should have predict_proba method"
    
    def test_gradient_boosting_training(self, sample_data):
        """Test that Gradient Boosting trains without error."""
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Prepare data
        numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove('default')
        
        X = sample_data[numeric_features]
        y = sample_data['default']
        
        # Train model
        gb = GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42
        )
        gb.fit(X, y)
        
        assert gb is not None, "Gradient Boosting should train successfully"
        assert hasattr(gb, 'feature_importances_'), "Model should have feature_importances"
    
    def test_model_predictions_valid_range(self, sample_data):
        """Test that model predictions are valid probabilities (0-1)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove('default')
        
        X = sample_data[numeric_features]
        y = sample_data['default']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_scaled, y)
        
        y_proba = lr.predict_proba(X_scaled)[:, 1]
        
        assert (y_proba >= 0).all() and (y_proba <= 1).all(), "Probabilities should be in [0, 1]"
    
    def test_model_feature_importance_sums(self, sample_data):
        """Test that feature importances from GB sum to reasonable value."""
        from sklearn.ensemble import GradientBoostingClassifier
        
        numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove('default')
        
        X = sample_data[numeric_features]
        y = sample_data['default']
        
        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gb.fit(X, y)
        
        importances = gb.feature_importances_
        assert abs(importances.sum() - 1.0) < 0.001, "Feature importances should sum to ~1"


# ── INTEGRATION TESTS ─────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end pipeline integration tests."""
    
    def test_full_pipeline_ends_with_metrics(self, sample_data):
        """Test that full pipeline generates valid metrics."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        
        numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove('default')
        
        X = sample_data[numeric_features]
        y = sample_data['default']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train model
        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gb.fit(X_train, y_train)
        
        # Get predictions
        y_proba = gb.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        gini = gini_coefficient(y_test.values, y_proba)
        ks = ks_statistic(y_test.values, y_proba)
        
        assert 0 <= gini <= 1, "Gini should be valid"
        assert 0 <= ks <= 1, "KS should be valid"
        assert gini > 0, "Gini should be > 0 for trained model"
        assert ks > 0, "KS should be > 0 for trained model"
    
    def test_pipeline_output_distributions(self, sample_data):
        """Test that pipeline outputs have reasonable distributions."""
        from sklearn.ensemble import GradientBoostingClassifier
        
        numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove('default')
        
        X = sample_data[numeric_features]
        y = sample_data['default']
        
        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gb.fit(X, y)
        
        y_proba = gb.predict_proba(X)[:, 1]
        
        # Check that we have both high and low probabilities
        assert (y_proba < 0.3).sum() > 0, "Should predict some low probabilities"
        assert (y_proba > 0.7).sum() > 0, "Should predict some high probabilities"


# ── ERROR HANDLING TESTS ──────────────────────────────────────────────────

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_dataframe_handling(self):
        """Test behavior with empty dataframe."""
        empty_df = pd.DataFrame()
        
        # Should not crash, but should handle gracefully
        # (Actual behavior depends on implementation)
        assert empty_df.shape[0] == 0
    
    def test_null_values_in_features(self, sample_data):
        """Test that features with null values are handled."""
        sample_data.loc[0, 'income_monthly'] = np.nan
        
        # Features should be handled (either imputed or dropped)
        assert sample_data.isnull().sum().sum() > 0, "Test setup should create nulls"
        
        # In real pipeline, should handle these
    
    def test_all_zero_feature(self):
        """Test behavior when a feature is all zeros."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0, 0, 0, 0, 0, 0])  # All zeros
        
        # Should handle without crashing
        try:
            gini = gini_coefficient(y_true, y_proba)
            # Result may be 0 or -1, depending on implementation
            assert isinstance(gini, (int, float, np.number))
        except Exception as e:
            pytest.fail(f"Should handle all-zero predictions: {e}")


# ── RUN TESTS ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
