"""
模型查找工具 - 自动检测最新训练的模型
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import logging

from config.settings import config

logger = logging.getLogger(__name__)

def find_latest_model() -> Optional[str]:
    """查找最新训练的模型
    
    Returns:
        最新模型的路径，如果未找到则返回None
    """
    valid_models = []
    
    # 1. 查找outputs目录下的训练结果 (优先级最高)
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        training_dirs = sorted(outputs_dir.glob("training_*"))
        for training_dir in training_dirs:
            # 检查final_model目录
            final_model = training_dir / "final_model"
            if final_model.exists() and (final_model / "config.json").exists():
                config_file = final_model / "config.json"
                model_time = datetime.fromtimestamp(config_file.stat().st_mtime)
                valid_models.append((final_model, model_time, f"训练结果: {training_dir.name}"))
            
            # 检查最新的检查点
            checkpoints = sorted(training_dir.glob("checkpoint-*"))
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                if (latest_checkpoint / "config.json").exists():
                    config_file = latest_checkpoint / "config.json"
                    model_time = datetime.fromtimestamp(config_file.stat().st_mtime)
                    valid_models.append((latest_checkpoint, model_time, f"检查点: {training_dir.name}/{latest_checkpoint.name}"))
    
    # 2. 查找models目录下的当前模型 (备用)
    possible_paths = [
        (Path(config.paths["models_dir"]) / "xlm_roberta_sensitive_filter", "当前使用模型"),
        (Path("models") / "xlm_roberta_sensitive_filter", "备用模型路径"),
    ]
    
    for path, desc in possible_paths:
        if path.exists() and (path / "config.json").exists():
            config_file = path / "config.json"
            model_time = datetime.fromtimestamp(config_file.stat().st_mtime)
            valid_models.append((path, model_time, desc))
    
    if not valid_models:
        logger.warning("未找到训练好的模型")
        return None
    
    # 按修改时间降序排序，返回最新的模型
    valid_models.sort(key=lambda x: x[1], reverse=True)
    latest_path, latest_time, desc = valid_models[0]
    
    logger.info(f"找到最新模型: {latest_path}")
    logger.info(f"  类型: {desc}")
    logger.info(f"  训练时间: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
    return str(latest_path)

def find_all_models() -> List[Tuple[str, bool, Optional[datetime]]]:
    """查找所有可能的模型路径
    
    Returns:
        模型路径列表，每个元素包含 (路径, 是否存在, 修改时间)
    """
    possible_paths = [
        (str(Path(config.paths["models_dir"]) / "xlm_roberta_sensitive_filter"), "默认模型"),
        ("models/xlm_roberta_sensitive_filter", "备用模型"),
        ("ultimate_xlm_roberta_model/checkpoint-100", "旧检查点模型"),
        ("ultimate_xlm_roberta_model", "旧根目录模型"),
    ]
    
    results = []
    for path_str, description in possible_paths:
        path = Path(path_str)
        exists = path.exists() and (path / "config.json").exists()
        
        mod_time = None
        if exists:
            config_file = path / "config.json"
            mod_time = datetime.fromtimestamp(config_file.stat().st_mtime)
        
        results.append((path_str, exists, mod_time, description))
    
    return results

def get_model_info(model_path: str) -> dict:
    """获取模型详细信息
    
    Args:
        model_path: 模型路径
        
    Returns:
        模型信息字典
    """
    path = Path(model_path)
    
    if not path.exists():
        return {"exists": False, "error": "模型路径不存在"}
    
    config_file = path / "config.json"
    if not config_file.exists():
        return {"exists": False, "error": "模型配置文件不存在"}
    
    try:
        import json
        
        # 读取模型配置
        with open(config_file, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        
        # 获取文件信息
        mod_time = datetime.fromtimestamp(config_file.stat().st_mtime)
        
        # 检查必要文件
        required_files = [
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "tokenizer.json"
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (path / file_name).exists():
                missing_files.append(file_name)
        
        return {
            "exists": True,
            "path": str(path),
            "modified_time": mod_time.isoformat(),
            "modified_time_str": mod_time.strftime('%Y-%m-%d %H:%M:%S'),
            "model_type": model_config.get("model_type", "unknown"),
            "architectures": model_config.get("architectures", []),
            "vocab_size": model_config.get("vocab_size", 0),
            "hidden_size": model_config.get("hidden_size", 0),
            "num_hidden_layers": model_config.get("num_hidden_layers", 0),
            "missing_files": missing_files,
            "is_complete": len(missing_files) == 0
        }
        
    except Exception as e:
        return {"exists": True, "error": f"读取模型信息失败: {e}"}

def validate_model_path(model_path: str) -> bool:
    """验证模型路径是否有效
    
    Args:
        model_path: 模型路径
        
    Returns:
        是否为有效的模型路径
    """
    path = Path(model_path)
    
    # 检查路径是否存在
    if not path.exists():
        return False
    
    # 检查必要文件
    required_files = ["config.json", "model.safetensors"]
    for file_name in required_files:
        if not (path / file_name).exists():
            return False
    
    return True
