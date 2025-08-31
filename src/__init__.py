"""
Safe-Text - 智能文本安全检测系统

A comprehensive system for filtering sensitive content using AI models.
"""

__version__ = "1.0.0"
__author__ = "Safe-Text Team"
__email__ = "contact@example.com"

from .models import SensitiveWordTrainer, SensitiveWordInference
from .data import DataProcessor

__all__ = [
    "SensitiveWordTrainer",
    "SensitiveWordInference", 
    "DataProcessor"
]
