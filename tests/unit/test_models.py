"""
Unit tests for scam classifier.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.scam_classifier import ScamClassifier
from src.preprocessing.feature_extraction import FeatureExtractor


class TestScamClassifier:
    """Test suite for ScamClassifier."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample training data."""
        X_train = [
            "Congratulations! You won $1000000. Click here to claim!",
            "Hey, are we still meeting for lunch?",
            "URGENT: Your account has been compromised. Verify now!",
            "Thanks for your help with the project",
            "You've been selected for a FREE iPhone. Pay shipping only!",
            "Meeting rescheduled to 3pm"
        ]
        y_train = ['scam', 'legitimate', 'scam', 'legitimate', 'scam', 'legitimate']
        
        return X_train, y_train
    
    def test_classifier_initialization(self):
        """Test classifier can be initialized."""
        classifier = ScamClassifier()
        assert classifier is not None
        assert not classifier.is_fitted
    
    def test_classifier_fit(self, sample_data):
        """Test classifier can be fitted."""
        X_train, y_train = sample_data
        classifier = ScamClassifier()
        classifier.fit(X_train, y_train)
        
        assert classifier.is_fitted
        assert classifier.classes_ == ['legitimate', 'scam']
    
    def test_classifier_predict(self, sample_data):
        """Test classifier predictions."""
        X_train, y_train = sample_data
        classifier = ScamClassifier()
        classifier.fit(X_train, y_train)
        
        test_messages = [
            "URGENT! Click this link now!",
            "Let's grab coffee tomorrow"
        ]
        
        predictions = classifier.predict(test_messages)
        assert len(predictions) == 2
        assert predictions[0] == 1  # Should predict scam
        assert predictions[1] == 0  # Should predict legitimate
    
    def test_predict_proba(self, sample_data):
        """Test probability predictions."""
        X_train, y_train = sample_data
        classifier = ScamClassifier()
        classifier.fit(X_train, y_train)
        
        probabilities = classifier.predict_proba(["URGENT! Win money now!"])
        
        assert probabilities.shape == (1, 2)
        assert 0 <= probabilities[0][0] <= 1
        assert 0 <= probabilities[0][1] <= 1
        assert abs(sum(probabilities[0]) - 1.0) < 0.001  # Probabilities sum to 1
    
    def test_predict_single(self, sample_data):
        """Test single message prediction."""
        X_train, y_train = sample_data
        classifier = ScamClassifier()
        classifier.fit(X_train, y_train)
        
        result = classifier.predict_single("FREE money! Click now!")
        
        assert 'prediction' in result
        assert 'is_scam' in result
        assert 'confidence' in result
        assert 'scam_probability' in result
        assert isinstance(result['is_scam'], bool)


if __name__ == '__main__':
    pytest.main([__file__])