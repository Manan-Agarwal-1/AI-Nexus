"""
Sentiment analysis module.
Analyzes sentiment and emotional tone of messages.
"""

import nltk
from typing import Dict, List
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class SentimentAnalyzer:
    """Analyze sentiment of text messages."""
    
    def __init__(self, use_vader: bool = True, use_textblob: bool = True):
        """
        Initialize sentiment analyzer.
        
        Args:
            use_vader: Use VADER sentiment analyzer
            use_textblob: Use TextBlob sentiment analyzer
        """
        self.use_vader = use_vader
        self.use_textblob = use_textblob
        
        if use_vader:
            self.vader = SentimentIntensityAnalyzer()
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.use_vader:
            return {}
        
        scores = self.vader.polarity_scores(text)
        return {
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu'],
            'vader_positive': scores['pos'],
            'vader_compound': scores['compound']
        }
    
    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.use_textblob:
            return {}
        
        blob = TextBlob(text)
        return {
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Perform comprehensive sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with all sentiment scores
        """
        results = {}
        
        if self.use_vader:
            results.update(self.analyze_vader(text))
        
        if self.use_textblob:
            results.update(self.analyze_textblob(text))
        
        # Add derived features
        if 'vader_compound' in results:
            results['is_positive'] = 1 if results['vader_compound'] > 0.05 else 0
            results['is_negative'] = 1 if results['vader_compound'] < -0.05 else 0
            results['is_neutral'] = 1 if -0.05 <= results['vader_compound'] <= 0.05 else 0
        
        return results
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of sentiment dictionaries
        """
        logger.info(f"Analyzing sentiment for {len(texts)} texts")
        return [self.analyze(text) for text in texts]
    
    def get_sentiment_label(self, text: str) -> str:
        """
        Get simple sentiment label.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        if self.use_vader:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                return 'positive'
            elif compound <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        elif self.use_textblob:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0:
                return 'positive'
            elif polarity < 0:
                return 'negative'
            else:
                return 'neutral'
        else:
            return 'neutral'
    
    def detect_emotional_manipulation(self, text: str) -> Dict[str, any]:
        """
        Detect potential emotional manipulation tactics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with manipulation indicators
        """
        text_lower = text.lower()
        
        # Fear-based words
        fear_words = ['urgent', 'warning', 'alert', 'danger', 'risk', 'threat', 
                     'emergency', 'critical', 'immediately', 'suspend', 'block']
        fear_count = sum(1 for word in fear_words if word in text_lower)
        
        # Greed-based words
        greed_words = ['free', 'win', 'winner', 'prize', 'reward', 'claim',
                      'congratulations', 'selected', 'lucky', 'bonus', 'offer']
        greed_count = sum(1 for word in greed_words if word in text_lower)
        
        # Urgency words
        urgency_words = ['now', 'today', 'immediately', 'asap', 'hurry',
                        'limited', 'expires', 'deadline', 'act now', 'final']
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        
        # Authority/trust words
        authority_words = ['verify', 'confirm', 'update', 'secure', 'official',
                          'account', 'security', 'protected', 'authorized']
        authority_count = sum(1 for word in authority_words if word in text_lower)
        
        manipulation_score = fear_count + greed_count + urgency_count + authority_count
        
        return {
            'fear_words_count': fear_count,
            'greed_words_count': greed_count,
            'urgency_words_count': urgency_count,
            'authority_words_count': authority_count,
            'total_manipulation_score': manipulation_score,
            'high_manipulation': manipulation_score >= 3
        }


def analyze_sentiment(text: str, **kwargs) -> Dict[str, float]:
    """
    Convenience function for sentiment analysis.
    
    Args:
        text: Input text
        **kwargs: Additional arguments for SentimentAnalyzer
        
    Returns:
        Dictionary with sentiment scores
    """
    analyzer = SentimentAnalyzer(**kwargs)
    return analyzer.analyze(text)