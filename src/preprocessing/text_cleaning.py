"""
Text cleaning module for preprocessing messages.
Handles text normalization, cleaning, and standardization.
"""

import re
import string
from typing import List, Optional
import emoji
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """Text cleaning and normalization utilities."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_phone_numbers: bool = True,
                 remove_special_chars: bool = False,
                 remove_extra_whitespace: bool = True,
                 remove_emojis: bool = False,
                 expand_contractions: bool = True):
        """
        Initialize text cleaner.
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_phone_numbers: Remove phone numbers
            remove_special_chars: Remove special characters
            remove_extra_whitespace: Remove extra whitespace
            remove_emojis: Remove emoji characters
            expand_contractions: Expand contractions (e.g., don't -> do not)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_phone_numbers = remove_phone_numbers
        self.remove_special_chars = remove_special_chars
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_emojis = remove_emojis
        self.expand_contractions = expand_contractions
        
        # Regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        
        # Contractions mapping
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'t": " not",
            "'ve": " have",
            "'m": " am"
        }
    
    def clean(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Expand contractions
        if self.expand_contractions:
            text = self._expand_contractions(text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' URL ', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = self.email_pattern.sub(' EMAIL ', text)
        
        # Remove phone numbers
        if self.remove_phone_numbers:
            text = self.phone_pattern.sub(' PHONE ', text)
        
        # Remove emojis
        if self.remove_emojis:
            text = emoji.replace_emoji(text, replace='')
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove special characters (keeping alphanumeric and spaces)
        if self.remove_special_chars:
            text = re.sub(f'[^a-zA-Z0-9\\s]', ' ', text)
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = ' '.join(text.split())
        
        return text.strip()
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of text strings.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of cleaned texts
        """
        logger.info(f"Cleaning batch of {len(texts)} texts")
        return [self.clean(text) for text in texts]
    
    def _expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded contractions
        """
        for contraction, expansion in self.contractions.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        return text
    
    def extract_features_from_text(self, text: str) -> dict:
        """
        Extract metadata features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'length': len(text),
            'num_words': len(text.split()),
            'num_chars': len(text),
            'num_uppercase': sum(1 for c in text if c.isupper()),
            'num_exclamation': text.count('!'),
            'num_question': text.count('?'),
            'has_url': bool(self.url_pattern.search(text)),
            'has_email': bool(self.email_pattern.search(text)),
            'has_phone': bool(self.phone_pattern.search(text)),
            'num_digits': sum(c.isdigit() for c in text),
            'num_special_chars': sum(c in string.punctuation for c in text),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        }
        return features
    
    def detect_urgency_words(self, text: str) -> int:
        """
        Detect urgency-related words in text.
        
        Args:
            text: Input text
            
        Returns:
            Count of urgency words
        """
        urgency_words = [
            'urgent', 'immediate', 'now', 'hurry', 'limited', 'expires',
            'act now', 'final', 'last chance', 'today only', 'don\'t miss',
            'asap', 'emergency', 'critical', 'warning', 'alert'
        ]
        
        text_lower = text.lower()
        return sum(1 for word in urgency_words if word in text_lower)
    
    def detect_money_references(self, text: str) -> int:
        """
        Detect money-related references.
        
        Args:
            text: Input text
            
        Returns:
            Count of money references
        """
        money_patterns = [
            r'\$\d+',  # $100
            r'\d+\s*(?:dollars|bucks|USD|EUR|GBP)',
            r'(?:free|prize|won|win|claim|cash|money|reward)',
        ]
        
        count = 0
        for pattern in money_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count


def clean_text(text: str, **kwargs) -> str:
    """
    Convenience function to clean a single text.
    
    Args:
        text: Input text
        **kwargs: Cleaning options
        
    Returns:
        Cleaned text
    """
    cleaner = TextCleaner(**kwargs)
    return cleaner.clean(text)