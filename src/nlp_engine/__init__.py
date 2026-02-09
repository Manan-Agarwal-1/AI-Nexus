"""NLP engine package.

Avoid importing heavy NLP dependencies at package import time. Import
submodules directly when needed, for example:

    from src.nlp_engine.intent_detection import IntentDetector

This keeps startup lightweight in development environments.
"""

__all__ = [
    'IntentDetector',
    'detect_intent',
    'Tokenizer',
    'tokenize',
    'SentimentAnalyzer',
    'analyze_sentiment'
]