#!/usr/bin/env python3
"""
简化的模型训练脚本
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.trainer import SensitiveWordTrainer
from src.data import DataProcessor
from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="训练敏感词检测模型")
    parser.add_argument('--input-data', type=str, help='输入训练数据文件路径')
    parser.add_argument('--model-name', type=str, default='xlm-roberta-base', help='预训练模型名称')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    try:
        logger.info("开始训练模型...")
        
        # 初始化数据处理器
        processor = DataProcessor()
        
        # 处理数据（如果提供了输入文件）
        if args.input_data:
            if not Path(args.input_data).exists():
                raise FileNotFoundError(f"训练数据文件不存在: {args.input_data}")
            
            logger.info(f"处理输入数据: {args.input_data}")
            train_file = config.data_config["train_file"]
            val_file = config.data_config["val_file"]
            test_file = config.data_config["test_file"]
            
            processor.process_and_split(args.input_data, train_file, val_file, test_file)
        
        # 检查训练文件是否存在
        train_file = config.data_config["train_file"]
        val_file = config.data_config["val_file"]
        test_file = config.data_config["test_file"]
        
        if not Path(train_file).exists():
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        if not Path(val_file).exists():
            raise FileNotFoundError(f"验证文件不存在: {val_file}")
        
        # 初始化训练器
        trainer = SensitiveWordTrainer(args.model_name)
        
        # 加载模型和分词器
        trainer.load_model_and_tokenizer()
        
        # 准备数据集
        train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(
            train_file, val_file, test_file if Path(test_file).exists() else None
        )
        
        # 设置训练
        output_dir = trainer.setup_training(train_dataset, val_dataset, args.output_dir)
        
        # 训练模型
        results = trainer.train()
        
        # 在测试集上评估
        if test_dataset:
            test_results = trainer.evaluate_on_test_set(test_dataset)
            logger.info(f"测试结果: {test_results}")
        
        # 保存模型
        model_path = trainer.save_model()
        
        logger.info("模型训练完成!")
        logger.info(f"模型保存到: {model_path}")
        logger.info(f"训练日志保存到: {output_dir}")
        
        print(f"\n✅ 训练完成!")
        print(f"📁 模型保存到: {model_path}")
        print(f"📊 训练日志: {output_dir}")
        
        if test_dataset:
            test_accuracy = test_results.get('accuracy', 0)
            test_f1 = test_results.get('f1', 0)
            print(f"🎯 测试准确率: {test_accuracy:.4f}")
            print(f"📈 测试F1分数: {test_f1:.4f}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        print(f"❌ 训练失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
