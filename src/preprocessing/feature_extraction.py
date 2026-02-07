"""
Feature extraction module.
Extracts features from text for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from src.preprocessing.text_cleaning import TextCleaner
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)
config = get_config()


class FeatureExtractor:
    """Extract features from text data for ML models."""
    
    def __init__(self, 
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 use_tfidf: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range for text vectorization
            use_tfidf: Use TF-IDF (True) or Count vectorizer (False)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_tfidf = use_tfidf
        
        # Initialize vectorizer
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=config.MIN_DF,
                max_df=config.MAX_DF,
                stop_words='english'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=config.MIN_DF,
                max_df=config.MAX_DF,
                stop_words='english'
            )
        
        # Scaler for numerical features
        self.scaler = StandardScaler()
        
        # Text cleaner
        self.text_cleaner = TextCleaner(
            lowercase=True,
            remove_urls=False,  # Keep for feature extraction
            remove_emails=False,
            remove_phone_numbers=False,
            remove_special_chars=False,
            remove_emojis=False
        )
        
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'FeatureExtractor':
        """
        Fit the feature extractor on training data.
        
        Args:
            texts: List of text messages
            
        Returns:
            Self
        """
        logger.info("Fitting feature extractor...")
        
        # Clean texts
        cleaned_texts = self.text_cleaner.clean_batch(texts)
        
        # Fit vectorizer
        self.vectorizer.fit(cleaned_texts)
        
        # Extract metadata features for fitting scaler
        metadata_features = np.array([
            list(self.text_cleaner.extract_features_from_text(text).values())
            for text in texts
        ])
        
        self.scaler.fit(metadata_features)
        
        self.is_fitted = True
        logger.info(f"Feature extractor fitted with {len(self.vectorizer.get_feature_names_out())} text features")
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors.
        
        Args:
            texts: List of text messages
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        # Clean texts
        cleaned_texts = self.text_cleaner.clean_batch(texts)
        
        # Get text features (TF-IDF or Count)
        text_features = self.vectorizer.transform(cleaned_texts).toarray()
        
        # Get metadata features
        metadata_features = np.array([
            list(self.text_cleaner.extract_features_from_text(text).values())
            for text in texts
        ])
        
        # Scale metadata features
        metadata_features_scaled = self.scaler.transform(metadata_features)
        
        # Get urgency and money features
        urgency_features = np.array([
            [self.text_cleaner.detect_urgency_words(text),
             self.text_cleaner.detect_money_references(text)]
            for text in texts
        ])
        
        # Combine all features
        combined_features = np.hstack([
            text_features,
            metadata_features_scaled,
            urgency_features
        ])
        
        logger.debug(f"Transformed {len(texts)} texts to feature matrix of shape {combined_features.shape}")
        
        return combined_features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts in one step.
        
        Args:
            texts: List of text messages
            
        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted first")
        
        # Text feature names
        text_feature_names = list(self.vectorizer.get_feature_names_out())
        
        # Metadata feature names
        metadata_feature_names = [
            'length', 'num_words', 'num_chars', 'num_uppercase',
            'num_exclamation', 'num_question', 'has_url', 'has_email',
            'has_phone', 'num_digits', 'num_special_chars',
            'avg_word_length', 'uppercase_ratio'
        ]
        
        # Additional feature names
        additional_feature_names = ['urgency_words', 'money_references']
        
        return text_feature_names + metadata_feature_names + additional_feature_names
    
    def save(self, filepath: str):
        """
        Save the feature extractor to disk.
        
        Args:
            filepath: Path to save the feature extractor
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'use_tfidf': self.use_tfidf,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureExtractor':
        """
        Load a feature extractor from disk.
        
        Args:
            filepath: Path to load from
            
        Returns:
            FeatureExtractor instance
        """
        save_data = joblib.load(filepath)
        
        extractor = cls(
            max_features=save_data['max_features'],
            ngram_range=save_data['ngram_range'],
            use_tfidf=save_data['use_tfidf']
        )
        
        extractor.vectorizer = save_data['vectorizer']
        extractor.scaler = save_data['scaler']
        extractor.is_fitted = save_data['is_fitted']
        
        logger.info(f"Feature extractor loaded from {filepath}")
        
        return extractor


def extract_features(texts: List[str], 
                    extractor: Optional[FeatureExtractor] = None) -> np.ndarray:
    """
    Convenience function to extract features.
    
    Args:
        texts: List of text messages
        extractor: Optional pre-fitted extractor
        
    Returns:
        Feature matrix
    """
    if extractor is None:
        extractor = FeatureExtractor()
        return extractor.fit_transform(texts)
    else:
        return extractor.transform(texts)