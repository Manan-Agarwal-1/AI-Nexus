"""
Tokenizer module for text processing.
Provides tokenization utilities for NLP analysis.
"""

import nltk
from typing import List
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class Tokenizer:
    """Text tokenization utilities."""
    
    def __init__(self, remove_stopwords: bool = False, lowercase: bool = True):
        """
        Initialize tokenizer.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lowercase: Whether to convert to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.stopwords = set(stopwords.words('english')) if remove_stopwords else set()
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of word tokens
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lowercase
        if self.lowercase:
            tokens = [token.lower() for token in tokens]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stopwords]
        
        return tokens
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        return sent_tokenize(text)
    
    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[tuple]:
        """
        Generate n-grams from tokens.
        
        Args:
            tokens: List of tokens
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        
        return ngrams
    
    def tokenize_and_analyze(self, text: str) -> dict:
        """
        Tokenize and provide basic analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tokenization results and analysis
        """
        word_tokens = self.tokenize_words(text)
        sentence_tokens = self.tokenize_sentences(text)
        
        analysis = {
            'word_tokens': word_tokens,
            'sentence_tokens': sentence_tokens,
            'num_words': len(word_tokens),
            'num_sentences': len(sentence_tokens),
            'unique_words': len(set(word_tokens)),
            'avg_word_length': sum(len(w) for w in word_tokens) / max(len(word_tokens), 1),
            'avg_sentence_length': len(word_tokens) / max(len(sentence_tokens), 1),
        }
        
        return analysis


def tokenize(text: str, method: str = 'words', **kwargs) -> List:
    """
    Convenience function for tokenization.
    
    Args:
        text: Input text
        method: 'words' or 'sentences'
        **kwargs: Additional arguments for Tokenizer
        
    Returns:
        List of tokens
    """
    tokenizer = Tokenizer(**kwargs)
    
    if method == 'words':
        return tokenizer.tokenize_words(text)
    elif method == 'sentences':
        return tokenizer.tokenize_sentences(text)
    else:
        raise ValueError(f"Unknown method: {method}")