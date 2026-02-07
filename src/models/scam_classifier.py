"""
Scam classifier model.
Main ML model for classifying messages as scam or legitimate.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from pathlib import Path

from src.preprocessing.feature_extraction import FeatureExtractor
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)
config = get_config()


class ScamClassifier:
    """Binary classifier for scam detection."""
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 feature_extractor: Optional[FeatureExtractor] = None,
                 **model_params):
        """
        Initialize scam classifier.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic')
            feature_extractor: Feature extractor instance
            **model_params: Additional model parameters
        """
        self.model_type = model_type
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.model = self._init_model(model_type, model_params)
        self.is_fitted = False
        self.classes_ = None
        self.feature_importance_ = None
    
    def _init_model(self, model_type: str, model_params: dict):
        """Initialize the ML model."""
        default_params = config.MODEL_PARAMS if hasattr(config, 'MODEL_PARAMS') else {}
        params = {**default_params, **model_params}
        
        if model_type == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 5),
                random_state=params.get('random_state', 42)
            )
        elif model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=params.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: List[str], y: List[str]) -> 'ScamClassifier':
        """
        Train the classifier.
        
        Args:
            X: List of text messages
            y: List of labels ('scam' or 'legitimate')
            
        Returns:
            Self
        """
        logger.info(f"Training {self.model_type} classifier on {len(X)} samples")
        
        # Convert labels to binary
        y_binary = np.array([1 if label.lower() == 'scam' else 0 for label in y])
        
        # Extract features
        if not self.feature_extractor.is_fitted:
            X_features = self.feature_extractor.fit_transform(X)
        else:
            X_features = self.feature_extractor.transform(X)
        
        # Train model
        self.model.fit(X_features, y_binary)
        
        self.is_fitted = True
        self.classes_ = ['legitimate', 'scam']
        
        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        
        logger.info("Training complete")
        
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Predict classes for messages.
        
        Args:
            X: List of text messages
            
        Returns:
            Array of predictions (0=legitimate, 1=scam)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_features = self.feature_extractor.transform(X)
        predictions = self.model.predict(X_features)
        
        return predictions
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Predict probabilities for messages.
        
        Args:
            X: List of text messages
            
        Returns:
            Array of probabilities [prob_legitimate, prob_scam]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_features = self.feature_extractor.transform(X)
        probabilities = self.model.predict_proba(X_features)
        
        return probabilities
    
    def predict_single(self, text: str) -> Dict:
        """
        Predict for a single message with detailed output.
        
        Args:
            text: Input message
            
        Returns:
            Dictionary with prediction results
        """
        predictions = self.predict([text])
        probabilities = self.predict_proba([text])[0]
        
        result = {
            'text': text,
            'prediction': 'scam' if predictions[0] == 1 else 'legitimate',
            'is_scam': bool(predictions[0]),
            'confidence': float(max(probabilities)),
            'scam_probability': float(probabilities[1]),
            'legitimate_probability': float(probabilities[0]),
        }
        
        return result
    
    def evaluate(self, X: List[str], y: List[str]) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: List of test messages
            y: List of true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model on {len(X)} samples")
        
        # Convert labels
        y_binary = np.array([1 if label.lower() == 'scam' else 0 for label in y])
        
        # Predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': float(accuracy_score(y_binary, y_pred)),
            'precision': float(precision_score(y_binary, y_pred)),
            'recall': float(recall_score(y_binary, y_pred)),
            'f1_score': float(f1_score(y_binary, y_pred)),
            'roc_auc': float(roc_auc_score(y_binary, y_proba)),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_binary, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_binary, y_pred, target_names=self.classes_)
        logger.info(f"\nClassification Report:\n{report}")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples
        """
        if self.feature_importance_ is None:
            logger.warning("Feature importance not available for this model")
            return []
        
        feature_names = self.feature_extractor.get_feature_names()
        importance_pairs = list(zip(feature_names, self.feature_importance_))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return importance_pairs[:top_n]
    
    def save(self, filepath: str):
        """
        Save the classifier to disk.
        
        Args:
            filepath: Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'classes_': self.classes_,
            'feature_importance_': self.feature_importance_
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ScamClassifier':
        """
        Load a classifier from disk.
        
        Args:
            filepath: Path to load from
            
        Returns:
            ScamClassifier instance
        """
        save_data = joblib.load(filepath)
        
        classifier = cls(
            model_type=save_data['model_type'],
            feature_extractor=save_data['feature_extractor']
        )
        
        classifier.model = save_data['model']
        classifier.is_fitted = save_data['is_fitted']
        classifier.classes_ = save_data['classes_']
        classifier.feature_importance_ = save_data['feature_importance_']
        
        logger.info(f"Model loaded from {filepath}")
        
        return classifier


def train_classifier(X_train: List[str], y_train: List[str], **kwargs) -> ScamClassifier:
    """
    Convenience function to train a classifier.
    
    Args:
        X_train: Training messages
        y_train: Training labels
        **kwargs: Additional arguments for ScamClassifier
        
    Returns:
        Trained classifier
    """
    classifier = ScamClassifier(**kwargs)
    classifier.fit(X_train, y_train)
    return classifier