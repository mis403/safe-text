"""
简化的基础配置 - 只保留必要的路径和常量
主要训练配置请使用 config/training.yaml
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 基础路径配置
PATHS = {
    "models_dir": str(PROJECT_ROOT / "models"),
    "data_dir": str(PROJECT_ROOT / "data"),
    "logs_dir": str(PROJECT_ROOT / "logs"),
    "config_dir": str(PROJECT_ROOT / "config"),
    "output_dir": str(PROJECT_ROOT / "outputs")
}

# 数据文件路径
DATA_DIR = Path(PATHS["data_dir"])
MODEL_DIR = Path(PATHS["models_dir"]) 
OUTPUT_DIR = Path(PATHS["output_dir"])
LOGS_DIR = Path(PATHS["logs_dir"])

# 数据文件
TRAIN_DATA = DATA_DIR / "train.csv"
VAL_DATA = DATA_DIR / "val.csv"
TEST_DATA = DATA_DIR / "test.csv"

# 模型路径
FINE_TUNED_MODEL = MODEL_DIR / "xlm_roberta_sensitive_filter"

# 标签映射
LABELS = {0: "正常内容", 1: "敏感内容"}

# 创建必要目录
for path in PATHS.values():
    Path(path).mkdir(parents=True, exist_ok=True)

# 向后兼容的配置类（简化版）
class Config:
    """简化的配置类，主要用于向后兼容"""
    
    @property
    def paths(self):
        return PATHS
    
    @property
    def model_config(self):
        # 从 training.yaml 加载或使用默认值
        try:
            from config.simple_config import load_training_config
            config = load_training_config()
            return config['model']
        except:
            return {
                "name": "xlm-roberta-base",
                "max_length": 512,
                "num_labels": 2
            }
    
    @property
    def training_config(self):
        # 从 training.yaml 加载或使用默认值
        try:
            from config.simple_config import load_training_config
            config = load_training_config()
            return config['training']
        except:
            return {
                "batch_size": 8,
                "learning_rate": 1e-5,
                "num_epochs": 3
            }
    
    @property
    def logging_config(self):
        """日志配置"""
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": str(LOGS_DIR / "app.log"),
            "max_size": "10MB",
            "backup_count": 5
        }
    
    @property
    def data_config(self):
        """数据配置"""
        try:
            from config.simple_config import load_training_config
            config = load_training_config()
            data_config = config.get('data', {})
            # 添加文件路径
            data_config.update({
                'train_file': str(DATA_DIR / "train.csv"),
                'val_file': str(DATA_DIR / "val.csv"),
                'test_file': str(DATA_DIR / "test.csv")
            })
            return data_config
        except:
            return {
                'train_ratio': 0.7,
                'val_ratio': 0.1,
                'test_ratio': 0.2,
                'random_seed': 42,
                'train_file': str(DATA_DIR / "train.csv"),
                'val_file': str(DATA_DIR / "val.csv"),
                'test_file': str(DATA_DIR / "test.csv")
            }

# 全局配置实例（向后兼容）
config = Config()

# 向后兼容的常量
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3