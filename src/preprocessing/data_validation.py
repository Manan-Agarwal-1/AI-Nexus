"""
Data validation module.
Validates and cleans datasets for training and inference.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate and clean datasets."""
    
    def __init__(self, 
                 required_columns: List[str] = None,
                 text_column: str = 'message_text',
                 label_column: str = 'label'):
        """
        Initialize data validator.
        
        Args:
            required_columns: List of required column names
            text_column: Name of the text column
            label_column: Name of the label column
        """
        self.required_columns = required_columns or ['message_text', 'label']
        self.text_column = text_column
        self.label_column = label_column
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check required columns exist
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for null values in critical columns
        for col in self.required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"Column '{col}' has {null_count} null values")
        
        # Check text column has string data
        if self.text_column in df.columns:
            non_string = (~df[self.text_column].apply(lambda x: isinstance(x, str))).sum()
            if non_string > 0:
                issues.append(f"Column '{self.text_column}' has {non_string} non-string values")
        
        # Check label distribution
        if self.label_column in df.columns:
            label_counts = df[self.label_column].value_counts()
            logger.info(f"Label distribution:\n{label_counts}")
            
            if len(label_counts) < 2:
                issues.append("Dataset has only one class")
            
            # Check for class imbalance
            min_class_count = label_counts.min()
            max_class_count = label_counts.max()
            imbalance_ratio = max_class_count / min_class_count
            
            if imbalance_ratio > 10:
                issues.append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning DataFrame with {len(df)} rows")
        
        df_cleaned = df.copy()
        
        # Remove rows with null values in critical columns
        for col in self.required_columns:
            if col in df_cleaned.columns:
                before = len(df_cleaned)
                df_cleaned = df_cleaned.dropna(subset=[col])
                removed = before - len(df_cleaned)
                if removed > 0:
                    logger.info(f"Removed {removed} rows with null values in '{col}'")
        
        # Convert text column to string
        if self.text_column in df_cleaned.columns:
            df_cleaned[self.text_column] = df_cleaned[self.text_column].astype(str)
        
        # Remove duplicates
        before = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=[self.text_column])
        removed = before - len(df_cleaned)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate messages")
        
        # Remove empty messages
        if self.text_column in df_cleaned.columns:
            before = len(df_cleaned)
            df_cleaned = df_cleaned[df_cleaned[self.text_column].str.strip().str.len() > 0]
            removed = before - len(df_cleaned)
            if removed > 0:
                logger.info(f"Removed {removed} empty messages")
        
        # Standardize labels
        if self.label_column in df_cleaned.columns:
            df_cleaned[self.label_column] = df_cleaned[self.label_column].str.lower().str.strip()
        
        logger.info(f"Cleaning complete. {len(df_cleaned)} rows remaining")
        
        return df_cleaned
    
    def load_and_validate(self, filepath: str) -> pd.DataFrame:
        """
        Load and validate a dataset file.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            Validated DataFrame
        """
        logger.info(f"Loading dataset from {filepath}")
        
        # Load data
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Validate
        is_valid, issues = self.validate_dataframe(df)
        
        if not is_valid:
            logger.warning(f"Validation issues found: {issues}")
        
        # Clean
        df_cleaned = self.clean_dataframe(df)
        
        return df_cleaned
    
    def split_data(self, 
                   df: pd.DataFrame,
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of test set
            val_size: Proportion of validation set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[self.label_column] if self.label_column in df.columns else None
        )
        
        # Second split: separate validation set from training
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df[self.label_column] if self.label_column in train_val_df.columns else None
        )
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get dataset statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_samples': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
        }
        
        if self.text_column in df.columns:
            stats['avg_text_length'] = df[self.text_column].str.len().mean()
            stats['min_text_length'] = df[self.text_column].str.len().min()
            stats['max_text_length'] = df[self.text_column].str.len().max()
        
        if self.label_column in df.columns:
            stats['label_distribution'] = df[self.label_column].value_counts().to_dict()
        
        return stats


def validate_dataset(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function to validate a dataset.
    
    Args:
        filepath: Path to dataset file
        **kwargs: Additional arguments for DataValidator
        
    Returns:
        Validated DataFrame
    """
    validator = DataValidator(**kwargs)
    return validator.load_and_validate(filepath)