#!/usr/bin/env python3
"""
敏感词检测模型训练脚本 - 集成防过拟合功能
支持从零开始训练和增量训练
"""

import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config.simple_config import load_training_config, show_config
from src.models.trainer import SensitiveWordTrainer
from src.data import DataProcessor
from config.settings import config
from src.utils.logger import setup_logger
from src.utils.overfitting_detector import OverfittingDetector
import torch
import numpy as np
import random

logger = setup_logger(__name__)

def set_deterministic_training(seed=42):
    """设置确定性训练环境"""
    # Python随机种子
    random.seed(seed)
    
    # NumPy随机种子
    np.random.seed(seed)
    
    # PyTorch随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 设置确定性算法（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"✅ 设置确定性训练环境，种子: {seed}")

def clear_previous_models(keep_cache=True):
    """清理之前的模型文件，但保留缓存"""
    models_dir = Path("models")
    
    # 清理xlm_roberta_sensitive_filter目录
    model_filter_dir = models_dir / "xlm_roberta_sensitive_filter"
    if model_filter_dir.exists():
        logger.info(f"清理旧模型: {model_filter_dir}")
        shutil.rmtree(model_filter_dir)
    
    # 清理ultimate_xlm_roberta_model目录  
    ultimate_model_dir = Path("ultimate_xlm_roberta_model")
    if ultimate_model_dir.exists():
        logger.info(f"清理旧模型: {ultimate_model_dir}")
        shutil.rmtree(ultimate_model_dir)
    
    # 清理输出目录（保留最新的用于分析）
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        training_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith('training_')]
        if len(training_dirs) > 2:  # 保留最近2个训练目录
            training_dirs.sort(key=lambda x: x.stat().st_mtime)
            for old_dir in training_dirs[:-2]:
                logger.info(f"清理旧训练目录: {old_dir}")
                shutil.rmtree(old_dir)
    
    logger.info("✅ 模型清理完成")

def prepare_training_data(input_data=None, force_resplit=False, use_random_split=False):
    """准备和验证训练数据
    
    Args:
        input_data: 输入数据文件路径
        force_resplit: 是否强制重新分割数据
        use_random_split: 是否使用随机分割（每次不同的分割结果）
    """
    # 根据是否使用随机分割来初始化处理器
    processor = DataProcessor(use_random_seed=use_random_split)
    
    # 如果提供了输入数据，先处理它
    if input_data:
        if not Path(input_data).exists():
            raise FileNotFoundError(f"训练数据文件不存在: {input_data}")
        
        logger.info(f"处理输入数据: {input_data}")
        if use_random_split:
            logger.info("🎲 使用随机数据分割 - 每次训练将使用不同的数据分布")
        else:
            logger.info("🔒 使用固定数据分割 - 确保可重现性")
            
        train_file = "data/train.csv"
        val_file = "data/val.csv"
        test_file = "data/test.csv"
        
        processor.process_and_split(input_data, train_file, val_file, test_file)
        logger.info("✅ 数据处理完成")
    
    # 检查格式化数据是否存在（用于从零训练）
    formatted_data = Path("data/formatted_sensitive_data.csv")
    train_file = Path("data/train.csv")
    val_file = Path("data/val.csv") 
    test_file = Path("data/test.csv")
    
    # 如果强制重新分割或分割文件不存在，尝试从格式化数据分割
    if force_resplit or not all([train_file.exists(), val_file.exists(), test_file.exists()]):
        if formatted_data.exists():
            if force_resplit:
                logger.info("强制重新分割训练数据...")
            else:
                logger.info("从格式化数据分割训练数据...")
            processor.process_and_split(
                str(formatted_data),
                str(train_file),
                str(val_file), 
                str(test_file)
            )
            logger.info("✅ 数据分割完成")
        else:
            raise FileNotFoundError("未找到训练数据，请提供 --input-data 或确保 data/formatted_sensitive_data.csv 存在")
    else:
        logger.info("✅ 使用现有的数据分割文件（确保一致性）")
    
    # 验证数据文件
    for file_path in [train_file, val_file, test_file]:
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 检查文件大小
        size = file_path.stat().st_size
        if size < 1000:  # 小于1KB可能有问题
            logger.warning(f"数据文件可能过小: {file_path} ({size} bytes)")
    
    logger.info(f"✅ 训练数据验证完成")
    logger.info(f"   📁 训练集: {train_file}")
    logger.info(f"   📁 验证集: {val_file}")
    logger.info(f"   📁 测试集: {test_file}")
    
    return str(train_file), str(val_file), str(test_file)

def print_training_config(config):
    """打印训练配置信息"""
    print("\n🔧 训练配置信息:")
    show_config(config)

def monitor_training_progress(output_dir):
    """监控训练进度和过拟合"""
    detector = OverfittingDetector()
    
    try:
        # 分析训练日志
        analysis = detector.analyze_training_logs(output_dir)
        
        if analysis.get("overfitting"):
            logger.warning("⚠️  检测到过拟合信号:")
            for signal in analysis.get("signals", []):
                logger.warning(f"   • {signal}")
            
            if analysis.get("recommendation"):
                logger.info(f"💡 建议: {analysis['recommendation']}")
        else:
            logger.info("✅ 未检测到明显过拟合")
        
        # 保存分析图表
        if 'data' in analysis:
            save_path = Path("outputs") / "training_analysis_latest.png"
            detector.visualize_training(analysis, str(save_path))
            logger.info(f"📊 训练分析图保存到: {save_path}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"训练监控失败: {e}")
        return {"error": str(e)}

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="敏感词检测模型训练 - 支持防过拟合")
    parser.add_argument('--input-data', type=str, help='输入训练数据文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径 (默认: config/training.yaml)')
    parser.add_argument('--model-name', type=str, help='预训练模型名称 (覆盖配置文件)')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--clear-models', action='store_true', help='清理之前的模型文件')
    parser.add_argument('--skip-monitoring', action='store_true', help='跳过过拟合监控')
    parser.add_argument('--simple-mode', action='store_true', help='简化模式，不显示详细配置')
    parser.add_argument('--force-resplit', action='store_true', help='强制重新分割数据（可能导致不同结果）')
    parser.add_argument('--random-split', action='store_true', help='使用随机数据分割（每次训练不同的数据分布）')
    parser.add_argument('--deterministic', action='store_true', help='启用完全确定性训练（可能影响性能）')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_training_config(args.config)
        model_name = args.model_name or config['model']['name']
        
        # 设置确定性训练（如果启用）
        if args.deterministic:
            training_seed = config['training'].get('seed', 42)
            set_deterministic_training(training_seed)
        
        if not args.simple_mode:
            print("🚀 开始模型训练 - 防过拟合版本")
            print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print_training_config(config)
        else:
            logger.info("开始训练模型...")
        
        # 1. 清理旧模型（如果需要）
        if args.clear_models:
            clear_previous_models()
        
        # 2. 准备训练数据
        logger.info("📊 准备训练数据...")
        train_file, val_file, test_file = prepare_training_data(
            args.input_data, 
            args.force_resplit, 
            args.random_split
        )
        
        # 3. 初始化训练器（传入配置）
        logger.info(f"🤖 初始化训练器，模型: {model_name}")
        trainer = SensitiveWordTrainer(model_name, config)
        
        # 4. 加载模型和分词器
        logger.info("📥 加载预训练模型和分词器...")
        trainer.load_model_and_tokenizer()
        
        # 5. 准备数据集
        logger.info("🗃️  准备数据集...")
        train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(
            train_file, val_file, test_file
        )
        
        logger.info(f"📈 数据集统计:")
        logger.info(f"   • 训练集: {len(train_dataset)} 样本")
        logger.info(f"   • 验证集: {len(val_dataset)} 样本")
        logger.info(f"   • 测试集: {len(test_dataset)} 样本")
        
        # 6. 设置训练
        logger.info("⚙️  设置训练配置...")
        output_dir = trainer.setup_training(train_dataset, val_dataset, args.output_dir)
        
        # 7. 开始训练
        logger.info("🎯 开始模型训练...")
        if not args.simple_mode:
            print("\n" + "="*60)
            print("🔥 训练进行中... 请等待")
            print("💡 提示: 训练将自动应用早停机制防止过拟合")
            print("="*60)
        
        results = trainer.train()
        logger.info("✅ 训练完成!")
        
        # 8. 直接在训练目录中保存最终模型
        final_model_dir = Path(output_dir) / "final_model"
        final_model_dir.mkdir(exist_ok=True)
        
        # 直接保存到训练目录
        logger.info(f"💾 保存最终模型到训练目录: {final_model_dir}")
        trainer.model.save_pretrained(final_model_dir)
        trainer.tokenizer.save_pretrained(final_model_dir)
        
        # 保存训练配置
        training_config_dict = {
            'model_name': args.model_name,
            'model_config': trainer.model_config,
            'training_config': trainer.training_config,
            'device': trainer.device,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(final_model_dir / 'training_config.json', 'w') as f:
            json.dump(training_config_dict, f, indent=2)
        
        # 9. 测试集评估（使用刚保存的模型）
        test_results = None
        if test_dataset:
            logger.info("📊 在测试集上评估模型...")
            try:
                # 使用推理接口进行测试集评估
                from src.models.inference import SensitiveWordInference
                import pandas as pd
                from sklearn.metrics import accuracy_score
                
                # 加载刚保存的模型
                inference_model = SensitiveWordInference(str(final_model_dir))
                
                # 加载测试数据
                test_df = pd.read_csv(test_file)
                sample_size = min(200, len(test_df))  # 测试前200个样本
                test_sample = test_df.head(sample_size)
                
                predictions = []
                true_labels = test_sample['label'].tolist()
                
                for _, row in test_sample.iterrows():
                    try:
                        result = inference_model.predict_single(row['text'])
                        pred_label = 1 if result['is_sensitive'] else 0
                        predictions.append(pred_label)
                    except:
                        predictions.append(0)  # 默认为正常
                
                # 计算指标
                accuracy = accuracy_score(true_labels, predictions)
                test_results = {
                    'accuracy': accuracy,
                    'sample_size': sample_size,
                    'total_size': len(test_df)
                }
                
                logger.info(f"测试集评估完成 (样本: {sample_size}/{len(test_df)})")
                logger.info(f"测试准确率: {accuracy:.4f}")
                
            except Exception as e:
                logger.warning(f"测试集评估失败: {e}")
        
        # 保存训练摘要
        training_summary = {
            'training_date': datetime.now().isoformat(),
            'model_name': args.model_name,
            'final_model_path': str(final_model_dir),
            'training_output_dir': output_dir,
            'test_results': test_results
        }
        
        with open(Path(output_dir) / "training_summary.json", 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False)
        
        # 10. 更新models目录的符号链接
        try:
            models_link = Path("models/xlm_roberta_sensitive_filter")
            if models_link.exists() or models_link.is_symlink():
                models_link.unlink()  # 删除现有的文件或符号链接
            
            # 创建新的符号链接指向最新训练结果
            relative_path = Path("..") / "outputs" / Path(output_dir).name / "final_model"
            models_link.symlink_to(relative_path)
            
            logger.info(f"✅ 更新符号链接: models/xlm_roberta_sensitive_filter -> {relative_path}")
            model_path = str(models_link)  # 返回符号链接路径以保持兼容性
            
        except Exception as e:
            logger.warning(f"符号链接更新失败: {e}")
            model_path = str(final_model_dir)  # 返回实际路径
        
        # 9. 过拟合监控和分析
        if not args.skip_monitoring:
            logger.info("🔍 分析训练过程和过拟合...")
            analysis = monitor_training_progress(output_dir)
            
            if not args.simple_mode:
                if analysis.get("overfitting"):
                    print(f"\n⚠️  过拟合检测结果: 检测到过拟合信号")
                    print(f"💡 建议: {analysis.get('recommendation', '无')}")
                else:
                    print(f"\n✅ 过拟合检测结果: 训练良好")
        
        # 最终结果汇总
        if not args.simple_mode:
            print("\n" + "="*60)
            print("🎉 训练完成! 结果汇总:")
            print("="*60)
            print(f"📁 模型保存路径: {model_path}")
            print(f"📊 训练日志目录: {output_dir}")
            
            if test_results:
                print(f"🎯 测试准确率: {test_results['accuracy']:.4f} (样本: {test_results['sample_size']}/{test_results['total_size']})")
            
            analysis_plot = Path("outputs") / "training_analysis_latest.png"
            if analysis_plot.exists():
                print(f"📊 训练分析图: {analysis_plot}")
            
            print(f"⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
        else:
            print(f"\n✅ 训练完成!")
            print(f"📁 模型保存到: {model_path}")
            print(f"📊 训练日志: {output_dir}")
            
            if test_results:
                print(f"🎯 测试准确率: {test_results['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        print(f"\n❌ 训练失败: {e}")
        
        # 如果有部分结果，尝试保存
        try:
            if 'trainer' in locals() and trainer.model:
                emergency_path = Path("emergency_model_save")
                trainer.save_model(str(emergency_path))
                print(f"🚑 紧急保存模型到: {emergency_path}")
        except:
            pass
        
        sys.exit(1)

if __name__ == '__main__':
    main()