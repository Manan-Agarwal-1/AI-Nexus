"""
Configuration loader module.
Handles loading and managing application configuration.
"""

import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
import json


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, env_file: str = '.env'):
        """
        Initialize configuration.
        
        Args:
            env_file: Path to environment file
        """
        # Load environment variables
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
        
        # API Configuration
        self.API_HOST = os.getenv('API_HOST', '0.0.0.0')
        self.API_PORT = int(os.getenv('API_PORT', 5000))
        self.DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
        
        # Model Configuration
        self.MODEL_PATH = os.getenv('MODEL_PATH', 'models/saved_models/')
        self.SCAM_THRESHOLD = float(os.getenv('SCAM_THRESHOLD', 0.7))
        self.HIGH_RISK_THRESHOLD = float(os.getenv('HIGH_RISK_THRESHOLD', 0.85))
        
        # Database
        self.DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///scam_detection.db')
        
        # Logging
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('LOG_FILE', 'logs/scam_detection.log')
        
        # Feature Flags
        self.ENABLE_ANOMALY_DETECTION = os.getenv('ENABLE_ANOMALY_DETECTION', 'True').lower() == 'true'
        self.ENABLE_SENTIMENT_ANALYSIS = os.getenv('ENABLE_SENTIMENT_ANALYSIS', 'True').lower() == 'true'
        self.ENABLE_AUTO_RETRAIN = os.getenv('ENABLE_AUTO_RETRAIN', 'False').lower() == 'true'
        
        # Security
        self.SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
        self.JWT_SECRET = os.getenv('JWT_SECRET', 'dev-jwt-secret')
        
        # External Services
        self.REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')
        
        # Performance
        self.MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
        
        # Alerts
        self.EMAIL_ALERTS_ENABLED = os.getenv('EMAIL_ALERTS_ENABLED', 'False').lower() == 'true'
        self.SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
        self.EMAIL_FROM = os.getenv('EMAIL_FROM', 'alerts@scamdetection.com')
        self.EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
        
        # Data paths
        self.DATA_RAW_PATH = 'data/raw/'
        self.DATA_PROCESSED_PATH = 'data/processed/'
        self.DATA_EXTERNAL_PATH = 'data/external/'
        
        # Model hyperparameters
        self.MODEL_PARAMS = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # NLP Configuration
        self.MAX_FEATURES = 5000
        self.NGRAM_RANGE = (1, 2)
        self.MIN_DF = 2
        self.MAX_DF = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and not key.endswith('PASSWORD')
        }
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        # Check critical paths exist or can be created
        critical_paths = [
            self.MODEL_PATH,
            self.DATA_RAW_PATH,
            self.DATA_PROCESSED_PATH,
            Path(self.LOG_FILE).parent
        ]
        
        for path in critical_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"Error creating path {path}: {e}")
                    return False
        
        # Validate thresholds
        if not (0 <= self.SCAM_THRESHOLD <= 1):
            print(f"Invalid SCAM_THRESHOLD: {self.SCAM_THRESHOLD}")
            return False
        
        if not (0 <= self.HIGH_RISK_THRESHOLD <= 1):
            print(f"Invalid HIGH_RISK_THRESHOLD: {self.HIGH_RISK_THRESHOLD}")
            return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config({json.dumps(self.to_dict(), indent=2)})"


# Global configuration instance
config = Config()


def get_config() -> Config:
    """
    Get global configuration instance.
    
    Returns:
        Config instance
    """
    return config