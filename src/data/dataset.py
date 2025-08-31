"""
PyTorch dataset classes for sensitive word filtering.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Union
from transformers import PreTrainedTokenizer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SensitiveTextDataset(Dataset):
    """PyTorch dataset for sensitive text classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        """Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels (0=normal, 1=sensitive)
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if len(texts) != len(labels):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of labels ({len(labels)})")
        
        logger.info(f"Created dataset with {len(self)} samples")
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with encoded text and label
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def get_sample_texts(self, num_samples: int = 5) -> List[str]:
        """Get sample texts for inspection.
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            List of sample texts
        """
        return self.texts[:min(num_samples, len(self.texts))]
    
    def get_label_distribution(self) -> dict:
        """Get label distribution in the dataset.
        
        Returns:
            Dictionary with label counts
        """
        from collections import Counter
        
        label_counts = Counter(self.labels)
        total = len(self.labels)
        
        return {
            'counts': dict(label_counts),
            'percentages': {
                label: count / total * 100 
                for label, count in label_counts.items()
            }
        }
