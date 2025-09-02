"""
简化的配置加载器
直接从 YAML 文件读取配置，无需复杂的类结构
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_training_config(config_file: str = None) -> Dict[str, Any]:
    """加载训练配置
    
    Args:
        config_file: 配置文件路径，默认为 config/training.yaml
        
    Returns:
        配置字典
    """
    if config_file is None:
        config_file = Path(__file__).parent / "training.yaml"
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 应用预设配置（如果有的话）
        if 'presets' in config and config['presets']:
            for preset_name, preset_config in config['presets'].items():
                if preset_config and isinstance(preset_config, dict):  # 如果预设不为空且是字典，应用它
                    print(f"🎯 应用预设配置: {preset_name}")
                    config['training'].update(preset_config)
                    break  # 只应用第一个非空预设
        
        return config
        
    except FileNotFoundError:
        print(f"❌ 配置文件未找到: {config_file}")
        print("使用默认配置...")
        return get_default_config()
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        print("使用默认配置...")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'model': {
            'name': 'xlm-roberta-base',
            'max_length': 512,
            'num_labels': 2
        },
        'training': {
            'batch_size': 8,
            'learning_rate': 1e-5,
            'num_epochs': 3,
            'weight_decay': 0.1,
            'max_grad_norm': 1.0,
            'label_smoothing_factor': 0.1,
            'early_stopping_patience': 2,
            'early_stopping_threshold': 0.0005,
            'warmup_steps': 100,
            'warmup_ratio': 0.1,
            'lr_scheduler_type': 'cosine_with_restarts',
            'gradient_accumulation_steps': 2,
            'gradient_checkpointing': True,
            'eval_strategy': 'steps',
            'eval_steps': 50,
            'save_strategy': 'steps',
            'save_steps': 100,
            'logging_steps': 25,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'dataloader_pin_memory': True,
            'dataloader_drop_last': True,
            'remove_unused_columns': True,
            'seed': 42
        },
        'data': {
            'train_ratio': 0.7,
            'val_ratio': 0.1,
            'test_ratio': 0.2,
            'random_seed': 42
        }
    }

def show_config(config: Dict[str, Any] = None):
    """显示当前配置"""
    if config is None:
        config = load_training_config()
    
    print("🎯 当前训练配置:")
    print("="*50)
    
    model_config = config['model']
    training_config = config['training']
    
    print(f"📚 模型配置:")
    print(f"   • 模型名称: {model_config['name']}")
    print(f"   • 最大长度: {model_config['max_length']}")
    print(f"   • 标签数量: {model_config['num_labels']}")
    
    print(f"\n🎯 训练配置:")
    key_params = [
        ('batch_size', '批次大小'),
        ('learning_rate', '学习率'),
        ('num_epochs', '训练轮数'),
        ('weight_decay', '权重衰减'),
        ('early_stopping_patience', '早停耐心值'),
        ('early_stopping_threshold', '早停阈值'),
        ('label_smoothing_factor', '标签平滑'),
        ('max_grad_norm', '梯度裁剪'),
        ('lr_scheduler_type', '学习率调度器')
    ]
    
    for key, desc in key_params:
        value = training_config.get(key, '未设置')
        print(f"   • {desc}: {value}")
    
    print("="*50)

if __name__ == '__main__':
    # 测试配置加载
    config = load_training_config()
    show_config(config)
