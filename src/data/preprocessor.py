"""
Data preprocessing utilities for sensitive word filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from sklearn.model_selection import train_test_split

from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataProcessor:
    """Data preprocessing and management for sensitive word filtering."""
    
    def __init__(self):
        """Initialize data processor."""
        self.data_config = config.data_config
        self.random_seed = self.data_config["random_seed"]
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"Loaded {len(df)} samples from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
        
        # Validate data format
        self._validate_data(df)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate data format.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If data format is invalid
        """
        required_columns = ['text', 'label']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        if df['text'].isnull().any():
            raise ValueError("Text column contains null values")
        
        if not df['label'].isin([0, 1]).all():
            raise ValueError("Label column must contain only 0 (normal) or 1 (sensitive)")
        
        logger.info("Data validation passed")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        original_size = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        
        # Remove null values
        df = df.dropna()
        
        # Clean text
        df['text'] = df['text'].astype(str).str.strip()
        
        # Remove empty text
        df = df[df['text'].str.len() > 0]
        
        # Ensure labels are integers
        df['label'] = df['label'].astype(int)
        
        logger.info(f"Cleaned data: {original_size} -> {len(df)} samples")
        
        return df.reset_index(drop=True)
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data statistics
        """
        stats = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'text_length_stats': {
                'mean': df['text'].str.len().mean(),
                'median': df['text'].str.len().median(),
                'min': df['text'].str.len().min(),
                'max': df['text'].str.len().max(),
                'std': df['text'].str.len().std()
            },
            'class_balance': {
                'normal_ratio': (df['label'] == 0).mean(),
                'sensitive_ratio': (df['label'] == 1).mean()
            }
        }
        
        logger.info(f"Data analysis completed: {stats['total_samples']} total samples")
        
        return stats
    
    def split_data(
        self, 
        df: pd.DataFrame,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets.
        
        Args:
            df: Input DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio  
            test_ratio: Test set ratio
            stratify: Whether to stratify split by labels
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Use config defaults if not specified
        if train_ratio is None:
            train_ratio = self.data_config["train_ratio"]
        if val_ratio is None:
            val_ratio = self.data_config["val_ratio"]
        if test_ratio is None:
            test_ratio = self.data_config["test_ratio"]
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=self.random_seed,
            stratify=df['label'] if stratify else None
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=self.random_seed,
            stratify=train_val_df['label'] if stratify else None
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"  Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None
    ) -> None:
        """Save data splits to CSV files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            train_file: Training file path (optional)
            val_file: Validation file path (optional)
            test_file: Test file path (optional)
        """
        # Use config defaults if not specified
        if train_file is None:
            train_file = self.data_config["train_file"]
        if val_file is None:
            val_file = self.data_config["val_file"]
        if test_file is None:
            test_file = self.data_config["test_file"]
        
        # Ensure directory exists
        for file_path in [train_file, val_file, test_file]:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save files
        train_df.to_csv(train_file, index=False, encoding='utf-8')
        val_df.to_csv(val_file, index=False, encoding='utf-8')
        test_df.to_csv(test_file, index=False, encoding='utf-8')
        
        logger.info(f"Data splits saved:")
        logger.info(f"  Train: {train_file}")
        logger.info(f"  Validation: {val_file}")
        logger.info(f"  Test: {test_file}")
    
    def process_and_split(
        self,
        input_file: str,
        output_train: Optional[str] = None,
        output_val: Optional[str] = None,
        output_test: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Complete data processing pipeline.
        
        Args:
            input_file: Input CSV file path
            output_train: Output training file path
            output_val: Output validation file path
            output_test: Output test file path
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Starting data processing pipeline for {input_file}")
        
        # Load and clean data
        df = self.load_data(input_file)
        df = self.clean_data(df)
        
        # Analyze data
        stats = self.analyze_data(df)
        logger.info(f"Data statistics: {stats}")
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Save splits
        self.save_splits(
            train_df, val_df, test_df,
            output_train, output_val, output_test
        )
        
        logger.info("Data processing pipeline completed successfully")
        
        return train_df, val_df, test_df
