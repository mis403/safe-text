"""
Configuration settings for the sensitive word filtering system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "name": "xlm-roberta-base",
        "max_length": 512,
        "num_labels": 2,
        "cache_dir": str(PROJECT_ROOT / "models" / "cache")
    },
    
    "training": {
        "batch_size": 8,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "save_strategy": "steps",
        "save_steps": 100,
        "eval_strategy": "steps", 
        "eval_steps": 50,
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True
    },
    
    "data": {
        "train_file": str(PROJECT_ROOT / "data" / "train.csv"),
        "val_file": str(PROJECT_ROOT / "data" / "val.csv"),
        "test_file": str(PROJECT_ROOT / "data" / "test.csv"),
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "random_seed": 42
    },
    
    "inference": {
        "batch_size": 16,
        "confidence_threshold": 0.5,
        "use_rules": False,     # ❌ 禁用规则匹配
        "use_ai": True,         # ✅ 只使用AI
        "rule_priority": False  # ❌ 无规则优先级
    },
    
    "paths": {
        "models_dir": str(PROJECT_ROOT / "models"),
        "data_dir": str(PROJECT_ROOT / "data"),
        "logs_dir": str(PROJECT_ROOT / "logs"),
        "config_dir": str(PROJECT_ROOT / "config"),
        "output_dir": str(PROJECT_ROOT / "outputs")
    },
    
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": str(PROJECT_ROOT / "logs" / "app.log"),
        "max_size": "10MB",
        "backup_count": 5
    },
    
    "rules": {
        "config_file": str(PROJECT_ROOT / "config" / "multilingual_sensitive_words.yaml"),
        "case_sensitive": False,
        "use_regex": True,
        "min_match_length": 2
    }
}

class Config:
    """Configuration manager for the sensitive word filtering system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_file: Path to custom configuration file
        """
        self._config = DEFAULT_CONFIG.copy()
        if config_file:
            self.load_config(config_file)
        
        # Create necessary directories
        self._ensure_directories()
    
    def load_config(self, config_file: str):
        """Load configuration from YAML file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                custom_config = yaml.safe_load(f)
                self._merge_config(custom_config)
        except FileNotFoundError:
            print(f"Configuration file {config_file} not found, using defaults")
        except Exception as e:
            print(f"Error loading configuration: {e}, using defaults")
    
    def _merge_config(self, custom_config: Dict[str, Any]):
        """Merge custom configuration with defaults.
        
        Args:
            custom_config: Custom configuration dictionary
        """
        for section, values in custom_config.items():
            if section in self._config:
                if isinstance(values, dict):
                    self._config[section].update(values)
                else:
                    self._config[section] = values
            else:
                self._config[section] = values
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for key, path in self._config["paths"].items():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None):
        """Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key within section
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if key is None:
            return self._config.get(section, default)
        
        return self._config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def save_config(self, config_file: str):
        """Save current configuration to file.
        
        Args:
            config_file: Path to save configuration
        """
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config["model"]
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config["training"]
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config["data"]
    
    @property
    def inference_config(self) -> Dict[str, Any]:
        """Get inference configuration."""
        return self._config["inference"]
    
    @property
    def paths(self) -> Dict[str, str]:
        """Get paths configuration."""
        return self._config["paths"]
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config["logging"]
    
    @property
    def rules_config(self) -> Dict[str, Any]:
        """Get rules configuration."""
        return self._config["rules"]

# Global configuration instance
config = Config()

# Backward compatibility with old config.py
MODEL_NAME = config.get("model", "name")
MAX_LENGTH = config.get("model", "max_length")
BATCH_SIZE = config.get("training", "batch_size")
LEARNING_RATE = config.get("training", "learning_rate")
NUM_EPOCHS = config.get("training", "num_epochs")

# Paths
DATA_DIR = Path(config.paths["data_dir"])
MODEL_DIR = Path(config.paths["models_dir"]) 
OUTPUT_DIR = Path(config.paths["output_dir"])
LOGS_DIR = Path(config.paths["logs_dir"])

# Data files
TRAIN_DATA = DATA_DIR / "train.csv"
VAL_DATA = DATA_DIR / "val.csv"
TEST_DATA = DATA_DIR / "test.csv"

# Model path
FINE_TUNED_MODEL = MODEL_DIR / "xlm_roberta_sensitive_filter"

# Labels
LABELS = {0: "正常内容", 1: "敏感内容"}
