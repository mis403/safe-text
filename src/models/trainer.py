"""
Model training functionality for sensitive word filtering.
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import config
from src.data import SensitiveTextDataset, DataProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SensitiveWordTrainer:
    """Trainer for sensitive word classification models."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize trainer.
        
        Args:
            model_name: Pretrained model name or path
        """
        self.model_name = model_name or config.model_config["name"]
        self.device = self._get_device()
        self.model_config = config.model_config
        self.training_config = config.training_config
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        logger.info(f"Trainer initialized with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def _get_device(self) -> str:
        """Get the best available device for training.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model_and_tokenizer(self) -> None:
        """Load pretrained model and tokenizer."""
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.model_config.get("cache_dir")
            )
            
            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.model_config["num_labels"],
                problem_type="single_label_classification",
                cache_dir=self.model_config.get("cache_dir")
            )
            
            # Move model to device
            self.model.to(self.device)
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model and tokenizer: {e}")
            raise
    
    def prepare_datasets(
        self,
        train_file: str,
        val_file: str,
        test_file: Optional[str] = None
    ) -> Tuple[SensitiveTextDataset, SensitiveTextDataset, Optional[SensitiveTextDataset]]:
        """Prepare training, validation, and test datasets.
        
        Args:
            train_file: Training data file path
            val_file: Validation data file path
            test_file: Test data file path (optional)
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        processor = DataProcessor()
        max_length = self.model_config["max_length"]
        
        # Load training data
        logger.info(f"Loading training data from {train_file}")
        train_df = processor.load_data(train_file)
        train_dataset = SensitiveTextDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            max_length
        )
        
        # Load validation data
        logger.info(f"Loading validation data from {val_file}")
        val_df = processor.load_data(val_file)
        val_dataset = SensitiveTextDataset(
            val_df['text'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer,
            max_length
        )
        
        # Load test data if provided
        test_dataset = None
        if test_file and Path(test_file).exists():
            logger.info(f"Loading test data from {test_file}")
            test_df = processor.load_data(test_file)
            test_dataset = SensitiveTextDataset(
                test_df['text'].tolist(),
                test_df['label'].tolist(),
                self.tokenizer,
                max_length
            )
        
        logger.info(f"Datasets prepared: train={len(train_dataset)}, "
                   f"val={len(val_dataset)}, test={len(test_dataset) if test_dataset else 0}")
        
        return train_dataset, val_dataset, test_dataset
    
    def setup_training(
        self,
        train_dataset: SensitiveTextDataset,
        val_dataset: SensitiveTextDataset,
        output_dir: Optional[str] = None
    ) -> str:
        """Setup training configuration and trainer.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for training artifacts
            
        Returns:
            Output directory path
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        # Create output directory
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(Path(config.paths["output_dir"]) / f"training_{timestamp}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Adjust batch size based on device
        batch_size = self.training_config["batch_size"]
        if self.device == "cpu":
            batch_size = min(batch_size, 4)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=self.training_config["num_epochs"],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=self.training_config["warmup_steps"],
            weight_decay=self.training_config["weight_decay"],
            learning_rate=self.training_config["learning_rate"],
            logging_dir=str(output_path / "logs"),
            logging_steps=self.training_config["logging_steps"],
            eval_strategy=self.training_config["eval_strategy"],
            eval_steps=self.training_config["eval_steps"],
            save_strategy=self.training_config["save_strategy"],
            save_steps=self.training_config["save_steps"],
            load_best_model_at_end=self.training_config["load_best_model_at_end"],
            metric_for_best_model=self.training_config["metric_for_best_model"],
            greater_is_better=self.training_config["greater_is_better"],
            report_to=None,
            push_to_hub=False,
            dataloader_num_workers=0 if self.device == "mps" else 2,
            fp16=False,  # Disable for Apple Silicon compatibility
            remove_unused_columns=True,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info(f"Training setup completed. Output directory: {output_path}")
        return str(output_path)
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self) -> Dict[str, Any]:
        """Start model training.
        
        Returns:
            Training results dictionary
        """
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call setup_training() first.")
        
        logger.info("Starting model training...")
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            logger.info("Training completed successfully")
            
            # Get final evaluation results
            eval_results = self.trainer.evaluate()
            
            # Log results
            logger.info("Final evaluation results:")
            for key, value in eval_results.items():
                logger.info(f"  {key}: {value:.4f}")
            
            return {
                'train_result': train_result,
                'eval_results': eval_results
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_on_test_set(
        self, 
        test_dataset: SensitiveTextDataset,
        save_confusion_matrix: bool = True
    ) -> Dict[str, Any]:
        """Evaluate model on test set.
        
        Args:
            test_dataset: Test dataset
            save_confusion_matrix: Whether to save confusion matrix plot
            
        Returns:
            Test evaluation results
        """
        if not self.trainer:
            raise RuntimeError("Trainer not initialized.")
        
        logger.info("Evaluating on test set...")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = [test_dataset[i]['labels'].item() for i in range(len(test_dataset))]
        
        # Calculate detailed metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_metrics': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1': f1_per_class.tolist(),
                'support': support.tolist()
            }
        }
        
        logger.info(f"Test set results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Generate confusion matrix
        if save_confusion_matrix:
            self._save_confusion_matrix(y_true, y_pred)
        
        return results
    
    def _save_confusion_matrix(self, y_true, y_pred) -> None:
        """Save confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Normal', 'Sensitive'],
                yticklabels=['Normal', 'Sensitive']
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            output_file = Path(config.paths["output_dir"]) / 'confusion_matrix.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to: {output_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save confusion matrix: {e}")
    
    def save_model(self, output_path: Optional[str] = None) -> str:
        """Save trained model and tokenizer.
        
        Args:
            output_path: Output directory path
            
        Returns:
            Path where model was saved
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer not loaded.")
        
        if not output_path:
            output_path = str(config.paths["models_dir"] / "xlm_roberta_sensitive_filter")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to: {output_path}")
        
        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save training configuration
        config_dict = {
            'model_name': self.model_name,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'device': self.device,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path / 'training_config.json', 'w') as f:
            import json
            json.dump(config_dict, f, indent=2)
        
        logger.info("Model saved successfully")
        return str(output_path)
