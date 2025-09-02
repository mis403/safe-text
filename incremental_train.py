#!/usr/bin/env python3
"""
增量训练脚本 - 在已有模型基础上继续训练
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.trainer import SensitiveWordTrainer
from src.data import DataProcessor
from config.settings import config
from src.utils.logger import setup_logger
from src.utils.model_finder import find_latest_model

logger = setup_logger(__name__)

def merge_training_data(original_file: str, additional_file: str, output_file: str):
    """
    合并原始训练数据和新增数据
    
    Args:
        original_file: 原始训练数据文件
        additional_file: 新增训练数据文件
        output_file: 合并后的输出文件
    """
    logger.info(f"合并训练数据: {original_file} + {additional_file} -> {output_file}")
    
    # 读取原始数据
    df_original = pd.read_csv(original_file)
    logger.info(f"原始数据: {len(df_original)} 条")
    
    # 读取新增数据
    df_additional = pd.read_csv(additional_file)
    logger.info(f"新增数据: {len(df_additional)} 条")
    
    # 合并数据
    df_merged = pd.concat([df_original, df_additional], ignore_index=True)
    
    # 去重（基于text列）
    initial_count = len(df_merged)
    df_merged.drop_duplicates(subset=['text'], inplace=True)
    final_count = len(df_merged)
    
    logger.info(f"合并后数据: {initial_count} 条")
    logger.info(f"去重后数据: {final_count} 条")
    logger.info(f"去除重复: {initial_count - final_count} 条")
    
    # 保存合并后的数据
    df_merged.to_csv(output_file, index=False)
    logger.info(f"合并数据保存到: {output_file}")
    
    return output_file

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="增量训练敏感词检测模型")
    parser.add_argument('--additional-data', type=str, required=True, help='新增训练数据文件路径')
    parser.add_argument('--base-model', type=str, help='基础模型路径（不指定则自动查找最新模型）')
    parser.add_argument('--original-data', type=str, default='data/formatted_sensitive_data.csv', help='原始训练数据文件路径')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='学习率（增量训练建议使用较小的学习率）')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数（增量训练建议使用较少轮数）')
    
    args = parser.parse_args()
    
    try:
        logger.info("开始增量训练...")
        
        # 检查新增数据文件
        if not Path(args.additional_data).exists():
            raise FileNotFoundError(f"新增训练数据文件不存在: {args.additional_data}")
        
        # 检查原始数据文件
        if not Path(args.original_data).exists():
            raise FileNotFoundError(f"原始训练数据文件不存在: {args.original_data}")
        
        # 合并训练数据
        merged_data_file = "data/merged_training_data.csv"
        merge_training_data(args.original_data, args.additional_data, merged_data_file)
        
        # 初始化数据处理器
        processor = DataProcessor()
        
        # 处理合并后的数据
        logger.info(f"处理合并后的数据: {merged_data_file}")
        train_file = config.data_config["train_file"]
        val_file = config.data_config["val_file"]
        test_file = config.data_config["test_file"]
        
        processor.process_and_split(merged_data_file, train_file, val_file, test_file)
        
        # 查找基础模型
        if args.base_model:
            base_model_path = args.base_model
        else:
            base_model_path = find_latest_model()
            if not base_model_path:
                logger.warning("未找到已训练的模型，将从预训练模型开始训练")
                base_model_path = "xlm-roberta-base"
        
        logger.info(f"使用基础模型: {base_model_path}")
        
        # 初始化训练器
        trainer = SensitiveWordTrainer(base_model_path)
        
        # 设置较小的学习率用于增量训练
        trainer.training_config.update({
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.epochs,
            "warmup_steps": 100,  # 减少warmup步数
        })
        
        logger.info(f"增量训练配置: 学习率={args.learning_rate}, 轮数={args.epochs}")
        
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
        
        logger.info("增量训练完成!")
        logger.info(f"模型保存到: {model_path}")
        logger.info(f"训练日志保存到: {output_dir}")
        
        print(f"\n✅ 增量训练完成!")
        print(f"📁 模型保存到: {model_path}")
        print(f"📊 训练日志: {output_dir}")
        
        if test_dataset:
            test_accuracy = test_results.get('accuracy', 0)
            test_f1 = test_results.get('f1', 0)
            print(f"🎯 测试准确率: {test_accuracy:.4f}")
            print(f"📈 测试F1分数: {test_f1:.4f}")
        
        # 清理临时文件
        if Path(merged_data_file).exists():
            Path(merged_data_file).unlink()
            logger.info(f"清理临时文件: {merged_data_file}")
        
    except Exception as e:
        logger.error(f"增量训练失败: {e}")
        print(f"❌ 增量训练失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
