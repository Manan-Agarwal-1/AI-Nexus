"""
Risk scoring model.
Calculates comprehensive risk scores for messages.
"""

import numpy as np
from typing import Dict, List
from src.models.scam_classifier import ScamClassifier
from src.nlp_engine.sentiment_analysis import SentimentAnalyzer
from src.nlp_engine.intent_detection import IntentDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskScoringModel:
    """Calculate risk scores for messages."""
    
    def __init__(self, 
                 classifier: ScamClassifier,
                 sentiment_analyzer: SentimentAnalyzer = None,
                 intent_detector: IntentDetector = None,
                 weights: Dict[str, float] = None):
        """
        Initialize risk scoring model.
        
        Args:
            classifier: Trained scam classifier
            sentiment_analyzer: Sentiment analyzer instance
            intent_detector: Intent detector instance
            weights: Weights for different scoring components
        """
        self.classifier = classifier
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.intent_detector = intent_detector or IntentDetector()
        
        # Default weights
        self.weights = weights or {
            'ml_prediction': 0.5,
            'intent_analysis': 0.25,
            'sentiment_analysis': 0.15,
            'pattern_matching': 0.1
        }
    
    def calculate_risk_score(self, text: str) -> Dict:
        """
        Calculate comprehensive risk score for a message.
        
        Args:
            text: Input message
            
        Returns:
            Dictionary with risk score and detailed breakdown
        """
        # ML Classifier score
        ml_result = self.classifier.predict_single(text)
        ml_score = ml_result['scam_probability']
        
        # Intent analysis score
        intent_analysis = self.intent_detector.analyze_intent(text)
        intent_score = min(intent_analysis['total_scam_indicators'] / 5.0, 1.0)
        
        # Sentiment analysis score
        sentiment_analysis = self.sentiment_analyzer.analyze(text)
        manipulation_features = self.sentiment_analyzer.detect_emotional_manipulation(text)
        sentiment_score = min(manipulation_features['total_manipulation_score'] / 5.0, 1.0)
        
        # Pattern matching score (basic)
        pattern_score = self._calculate_pattern_score(text)
        
        # Weighted final score
        final_risk_score = (
            self.weights['ml_prediction'] * ml_score +
            self.weights['intent_analysis'] * intent_score +
            self.weights['sentiment_analysis'] * sentiment_score +
            self.weights['pattern_matching'] * pattern_score
        )
        
        # Risk level classification
        risk_level = self._classify_risk_level(final_risk_score)
        
        return {
            'risk_score': float(final_risk_score),
            'risk_level': risk_level,
            'ml_score': float(ml_score),
            'intent_score': float(intent_score),
            'sentiment_score': float(sentiment_score),
            'pattern_score': float(pattern_score),
            'ml_prediction': ml_result['prediction'],
            'confidence': float(ml_result['confidence']),
            'risk_factors': self.intent_detector.get_risk_factors(text),
            'intent_analysis': intent_analysis,
            'sentiment_analysis': sentiment_analysis,
            'manipulation_indicators': manipulation_features
        }
    
    def _calculate_pattern_score(self, text: str) -> float:
        """
        Calculate pattern-based risk score.
        
        Args:
            text: Input text
            
        Returns:
            Pattern score between 0 and 1
        """
        import re
        
        score = 0.0
        
        # Multiple exclamation marks
        if len(re.findall(r'!{2,}', text)) > 0:
            score += 0.2
        
        # All caps words
        words = text.split()
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        if caps_words > len(words) * 0.3:
            score += 0.2
        
        # Excessive punctuation
        punct_ratio = sum(c in '!?.,' for c in text) / max(len(text), 1)
        if punct_ratio > 0.1:
            score += 0.2
        
        # Suspicious domains
        suspicious_domains = ['.xyz', '.tk', '.ml', 'bit.ly', 'tinyurl']
        if any(domain in text.lower() for domain in suspicious_domains):
            score += 0.3
        
        # Numbers with $ or monetary amounts
        if re.search(r'\$\d{4,}', text):
            score += 0.1
        
        return min(score, 1.0)
    
    def _classify_risk_level(self, score: float) -> str:
        """
        Classify risk level based on score.
        
        Args:
            score: Risk score between 0 and 1
            
        Returns:
            Risk level: 'low', 'medium', 'high', or 'critical'
        """
        if score < 0.3:
            return 'low'
        elif score < 0.5:
            return 'medium'
        elif score < 0.75:
            return 'high'
        else:
            return 'critical'
    
    def batch_score(self, texts: List[str]) -> List[Dict]:
        """
        Calculate risk scores for multiple messages.
        
        Args:
            texts: List of messages
            
        Returns:
            List of risk score dictionaries
        """
        logger.info(f"Calculating risk scores for {len(texts)} messages")
        return [self.calculate_risk_score(text) for text in texts]
    
    def get_summary_statistics(self, scores: List[Dict]) -> Dict:
        """
        Get summary statistics for a batch of scores.
        
        Args:
            scores: List of risk score dictionaries
            
        Returns:
            Summary statistics
        """
        risk_scores = [s['risk_score'] for s in scores]
        risk_levels = [s['risk_level'] for s in scores]
        
        return {
            'total_messages': len(scores),
            'avg_risk_score': float(np.mean(risk_scores)),
            'min_risk_score': float(np.min(risk_scores)),
            'max_risk_score': float(np.max(risk_scores)),
            'std_risk_score': float(np.std(risk_scores)),
            'risk_level_distribution': {
                level: risk_levels.count(level)
                for level in ['low', 'medium', 'high', 'critical']
            },
            'high_risk_count': sum(1 for s in scores if s['risk_level'] in ['high', 'critical'])
        }