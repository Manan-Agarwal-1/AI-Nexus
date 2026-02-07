"""
Anomaly detection model.
Detects unusual patterns that may indicate new scam techniques.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict
import joblib
from pathlib import Path

from src.preprocessing.feature_extraction import FeatureExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """Detect anomalous messages using unsupervised learning."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers
            random_state: Random seed
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.feature_extractor = FeatureExtractor()
        self.is_fitted = False
    
    def fit(self, X: List[str]) -> 'AnomalyDetector':
        """
        Fit the anomaly detector.
        
        Args:
            X: List of messages
            
        Returns:
            Self
        """
        logger.info(f"Fitting anomaly detector on {len(X)} samples")
        
        # Extract features
        X_features = self.feature_extractor.fit_transform(X)
        
        # Fit model
        self.model.fit(X_features)
        self.is_fitted = True
        
        logger.info("Anomaly detector fitted")
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: List of messages
            
        Returns:
            Array of predictions (1=normal, -1=anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_features = self.feature_extractor.transform(X)
        predictions = self.model.predict(X_features)
        
        return predictions
    
    def score_samples(self, X: List[str]) -> np.ndarray:
        """
        Get anomaly scores for messages.
        
        Args:
            X: List of messages
            
        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_features = self.feature_extractor.transform(X)
        scores = self.model.score_samples(X_features)
        
        return scores
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze a single message for anomalies.
        
        Args:
            text: Input message
            
        Returns:
            Dictionary with anomaly analysis
        """
        prediction = self.predict([text])[0]
        score = self.score_samples([text])[0]
        
        return {
            'is_anomaly': bool(prediction == -1),
            'anomaly_score': float(score),
            'confidence': float(abs(score))
        }
    
    def save(self, filepath: str):
        """Save the anomaly detector."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Anomaly detector saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AnomalyDetector':
        """Load anomaly detector from disk."""
        save_data = joblib.load(filepath)
        
        detector = cls()
        detector.model = save_data['model']
        detector.feature_extractor = save_data['feature_extractor']
        detector.is_fitted = save_data['is_fitted']
        
        logger.info(f"Anomaly detector loaded from {filepath}")
        return detector