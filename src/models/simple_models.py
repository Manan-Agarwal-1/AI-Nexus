"""
Lightweight fallback models for development and testing.
These provide simple heuristics and do not require heavy ML dependencies.
"""
from typing import List, Dict
from pathlib import Path
from src.utils.logger import get_logger
from src.nlp_engine.intent_detection import IntentDetector

logger = get_logger(__name__)


class SimpleScamClassifier:
    """Very small heuristic classifier.
    Returns scam probability based on keyword matching.
    """

    SCAM_KEYWORDS = [
        'win', 'winner', 'congratulations', 'prize', 'claim', 'free',
        'urgent', 'immediately', 'transfer', 'bank', 'account', 'password',
        'verify', 'ssn', 'social security', 'click', 'http', 'link'
    ]

    def __init__(self):
        self.is_fitted = True

    def predict(self, X: List[str]):
        return [1 if self._score_text(x) >= 0.5 else 0 for x in X]

    def predict_proba(self, X: List[str]):
        probs = []
        for x in X:
            p = float(self._score_text(x))
            probs.append([1.0 - p, p])
        return probs

    def predict_single(self, text: str) -> Dict:
        p = float(self._score_text(text))
        pred = 'scam' if p >= 0.5 else 'legitimate'
        return {
            'text': text,
            'prediction': pred,
            'is_scam': p >= 0.5,
            'confidence': p,
            'scam_probability': p,
            'legitimate_probability': 1.0 - p
        }

    def _score_text(self, text: str) -> float:
        text_l = text.lower()
        matches = sum(1 for kw in self.SCAM_KEYWORDS if kw in text_l)
        # score scaled between 0 and 1
        score = min(matches / 5.0, 1.0)
        # small boost for suspicious punctuation/links
        if 'http' in text_l or 'www.' in text_l:
            score = min(score + 0.3, 1.0)
        if '!' in text_l and text_l.count('!') > 1:
            score = min(score + 0.1, 1.0)
        return score


class SimpleRiskScoringModel:
    """Combines simple classifier, intent detector and heuristics to produce risk scores."""

    def __init__(self, classifier=None, intent_detector: IntentDetector = None, weights: Dict = None):
        self.classifier = classifier or SimpleScamClassifier()
        self.intent_detector = intent_detector or IntentDetector()
        self.weights = weights or {
            'ml_prediction': 0.6,
            'intent_analysis': 0.3,
            'pattern_matching': 0.1
        }

    def calculate_risk_score(self, text: str) -> Dict:
        ml = self.classifier.predict_single(text)
        intent = self.intent_detector.analyze_intent(text)

        ml_score = ml['scam_probability']
        intent_score = min(intent['total_scam_indicators'] / 5.0, 1.0)

        # simple pattern score
        pattern = 0.0
        if any(ext in text.lower() for ext in ['.xyz', '.tk', 'bit.ly', 'tinyurl']):
            pattern += 0.3
        if text.count('!') > 1:
            pattern += 0.1
        if any(sym in text for sym in ['$']) and any(ch.isdigit() for ch in text):
            pattern += 0.1
        pattern = min(pattern, 1.0)

        final = (
            self.weights['ml_prediction'] * ml_score +
            self.weights['intent_analysis'] * intent_score +
            self.weights['pattern_matching'] * pattern
        )

        risk_level = 'low'
        if final >= 0.75:
            risk_level = 'critical'
        elif final >= 0.5:
            risk_level = 'high'
        elif final >= 0.3:
            risk_level = 'medium'

        return {
            'risk_score': float(final),
            'risk_level': risk_level,
            'ml_score': float(ml_score),
            'intent_score': float(intent_score),
            'sentiment_score': 0.0,
            'pattern_score': float(pattern),
            'ml_prediction': ml['prediction'],
            'confidence': float(ml['confidence']),
            'risk_factors': self.intent_detector.get_risk_factors(text),
            'intent_analysis': intent
        }


class SimpleAnomalyDetector:
    """Tiny anomaly detector stub."""

    def analyze(self, text: str) -> Dict:
        # Very naive: flag messages with repeated characters or extremely long
        return {
            'is_anomalous': len(text) > 200 or ('!!!' in text) or (text.count('http') > 2),
            'reason': 'length_or_repetition'
        }
