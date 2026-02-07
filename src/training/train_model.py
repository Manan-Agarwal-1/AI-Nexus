"""
Model training script.
Trains and evaluates the scam detection model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.data_validation import DataValidator
from src.preprocessing.feature_extraction import FeatureExtractor
from src.models.scam_classifier import ScamClassifier
from src.models.anomaly_detection import AnomalyDetector
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)
config = get_config()


def train_model(data_path: str = None, 
                model_type: str = 'random_forest',
                save_path: str = None):
    """
    Train the scam detection model.
    
    Args:
        data_path: Path to training data CSV
        model_type: Type of model to train
        save_path: Path to save trained model
    """
    logger.info("=" * 50)
    logger.info("Starting Model Training")
    logger.info("=" * 50)
    
    # Default paths
    if data_path is None:
        data_path = Path(config.DATA_RAW_PATH) / 'scam_dataset.csv'
    
    if save_path is None:
        save_path = Path(config.MODEL_PATH) / 'scam_classifier.pkl'
    
    # Load and validate data
    logger.info(f"Loading data from {data_path}")
    validator = DataValidator()
    df = validator.load_and_validate(str(data_path))
    
    # Get statistics
    stats = validator.get_statistics(df)
    logger.info(f"Dataset statistics: {stats}")
    
    # Split data
    logger.info("Splitting data into train/val/test sets")
    train_df, val_df, test_df = validator.split_data(df, test_size=0.2, val_size=0.1)
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    # Prepare data
    X_train = train_df['message_text'].tolist()
    y_train = train_df['label'].tolist()
    
    X_val = val_df['message_text'].tolist()
    y_val = val_df['label'].tolist()
    
    X_test = test_df['message_text'].tolist()
    y_test = test_df['label'].tolist()
    
    # Train classifier
    logger.info(f"Training {model_type} classifier")
    classifier = ScamClassifier(model_type=model_type)
    classifier.fit(X_train, y_train)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set")
    val_metrics = classifier.evaluate(X_val, y_val)
    logger.info(f"Validation Metrics: {val_metrics}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_metrics = classifier.evaluate(X_test, y_test)
    logger.info(f"Test Metrics: {test_metrics}")
    
    # Feature importance
    logger.info("\nTop 20 Most Important Features:")
    top_features = classifier.get_feature_importance(top_n=20)
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"{i}. {feature}: {importance:.4f}")
    
    # Save model
    logger.info(f"\nSaving model to {save_path}")
    classifier.save(str(save_path))
    
    # Train anomaly detector (optional)
    if config.ENABLE_ANOMALY_DETECTION:
        logger.info("\nTraining anomaly detector")
        anomaly_detector = AnomalyDetector()
        anomaly_detector.fit(X_train)
        
        anomaly_save_path = Path(config.MODEL_PATH) / 'anomaly_detector.pkl'
        anomaly_detector.save(str(anomaly_save_path))
        logger.info(f"Anomaly detector saved to {anomaly_save_path}")
    
    # Save processed data
    processed_path = Path(config.DATA_PROCESSED_PATH) / 'cleaned_data.csv'
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")
    
    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info("=" * 50)
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    return classifier, test_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train scam detection model')
    parser.add_argument('--data', type=str, help='Path to training data CSV')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'logistic'],
                       help='Type of model to train')
    parser.add_argument('--output', type=str, help='Path to save trained model')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        model_type=args.model_type,
        save_path=args.output
    )