"""
Model inference functionality for sensitive word filtering.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SensitiveWordInference:
    """Inference engine for sensitive word detection using AI model and rules."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_rules: bool = False,
        use_ai: bool = True
    ):
        """Initialize inference engine.
        
        Args:
            model_path: Path to trained model
            use_rules: Whether to use rule-based matching
            use_ai: Whether to use AI model
        """
        self.model_path = model_path or str(Path(config.paths["models_dir"]) / "xlm_roberta_sensitive_filter")
        self.use_rules = use_rules
        self.use_ai = use_ai
        self.device = self._get_device()
        
        self.tokenizer = None
        self.model = None
        self.rule_matcher = None
        
        # Load components
        if self.use_ai:
            self.load_ai_model()
        if self.use_rules:
            logger.warning("规则匹配功能已移除，只使用AI模型")
            self.use_rules = False
            self.rule_matcher = None
        
        logger.info(f"Inference engine initialized (AI: {use_ai}, Rules: {use_rules})")
    
    def _get_device(self) -> str:
        """Get the best available device for inference.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_ai_model(self) -> None:
        """Load AI model and tokenizer."""
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        try:
            logger.info(f"Loading AI model from: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("AI model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            raise
    
    def predict_single(
        self,
        text: str,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Predict sensitivity of a single text.
        
        Args:
            text: Input text
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction results dictionary
        """
        results = {
            'text': text,
            'is_sensitive': False,
            'confidence': 0.0,
            'ai_result': None,
            'rule_result': None,
            'final_decision': 'normal',
            'methods_used': []
        }
        
        # AI prediction
        if self.use_ai and self.model:
            ai_result = self._predict_with_ai(text, return_probabilities)
            results['ai_result'] = ai_result
            results['methods_used'].append('ai')
        
        # Rule-based prediction (已移除)
        # 只使用AI模型进行预测
        
        # Combine results
        results.update(self._combine_predictions(results))
        
        return results
    
    def _predict_with_ai(self, text: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """Make prediction using AI model.
        
        Args:
            text: Input text
            return_probabilities: Whether to return probabilities
            
        Returns:
            AI prediction results
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("AI model not loaded")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=config.model_config["max_length"],
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Calculate probabilities
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()
        
        labels = {0: "正常内容", 1: "敏感内容"}
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'label': labels[prediction]
        }
        
        if return_probabilities:
            result['probabilities'] = {
                labels[0]: float(probabilities[0][0].item()),
                labels[1]: float(probabilities[0][1].item())
            }
        
        return result
    
    def _predict_with_rules(self, text: str) -> Dict[str, Any]:
        """规则匹配功能已移除，只使用AI模型"""
        logger.warning("规则匹配功能已移除")
        return {
            'is_sensitive': False,
            'confidence': 0.0,
            'matches': [],
            'match_count': 0
        }
    
    def _combine_predictions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine AI and rule-based predictions.
        
        Args:
            results: Results dictionary with AI and rule predictions
            
        Returns:
            Combined prediction results
        """
        ai_result = results.get('ai_result')
        rule_result = results.get('rule_result')
        
        is_sensitive = False
        confidence = 0.0
        final_decision = 'normal'
        
        # 只使用AI预测结果
        if ai_result:
            ai_threshold = 0.5  # 默认置信度阈值
            
            # 获取敏感内容的概率（无论预测类别是什么）
            probabilities = ai_result.get('probabilities', {})
            sensitive_prob = probabilities.get('敏感内容', 0.0)
            
            if sensitive_prob >= ai_threshold:
                is_sensitive = True
                confidence = sensitive_prob
                final_decision = 'sensitive'
            else:
                is_sensitive = False
                confidence = 1 - sensitive_prob  # 正常内容的置信度
                final_decision = 'normal'
        else:
            confidence = 0.5  # Default neutral confidence
        
        return {
            'is_sensitive': is_sensitive,
            'confidence': confidence,
            'final_decision': final_decision
        }
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Predict sensitivity for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        if not batch_size:
            batch_size = 16  # 默认批次大小
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                result = self.predict_single(text)
                results.append(result)
            
            if len(texts) > batch_size:
                logger.debug(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        return results
    
    def filter_text(
        self,
        text: str,
        replacement: str = "[敏感内容已过滤]"
    ) -> Dict[str, Any]:
        """Filter sensitive content from text.
        
        Args:
            text: Input text
            replacement: Replacement string for sensitive content
            
        Returns:
            Filtering results with original and filtered text
        """
        prediction = self.predict_single(text)
        
        result = {
            'original_text': text,
            'filtered_text': text,
            'is_sensitive': prediction['is_sensitive'],
            'confidence': prediction['confidence'],
            'prediction_details': prediction
        }
        
        # Replace text if sensitive
        if prediction['is_sensitive']:
            result['filtered_text'] = replacement
        
        return result
    
    def get_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Get statistics for a list of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Statistics dictionary
        """
        results = self.predict_batch(texts)
        
        total = len(results)
        sensitive_count = sum(1 for r in results if r['is_sensitive'])
        normal_count = total - sensitive_count
        
        confidences = [r['confidence'] for r in results]
        
        return {
            'total_texts': total,
            'sensitive_count': sensitive_count,
            'normal_count': normal_count,
            'sensitive_ratio': sensitive_count / total if total > 0 else 0,
            'average_confidence': np.mean(confidences) if confidences else 0,
            'confidence_std': np.std(confidences) if confidences else 0
        }
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for AI predictions.
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        config.set("inference", "confidence_threshold", threshold)
        logger.info(f"Confidence threshold set to {threshold}")

class CombinedInference(SensitiveWordInference):
    """Combined inference using both AI and rules (deprecated - use SensitiveWordInference)."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("CombinedInference is deprecated. Use SensitiveWordInference instead.")
        super().__init__(*args, **kwargs)
