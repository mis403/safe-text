#!/usr/bin/env python3
"""
交互式测试脚本 - 自动检测并使用最新训练好的模型
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.inference import SensitiveWordInference
from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class InteractiveModelTester:
    """交互式模型测试器"""
    
    def __init__(self):
        self.inference_engine = None
        self.model_path = None
        
    def find_latest_model(self):
        """自动查找最新的训练模型"""
        print("🔍 正在查找最新训练的模型...")
        
        # 可能的模型路径列表（按优先级排序）
        possible_paths = [
            Path("ultimate_xlm_roberta_model"),  # 最新训练的模型
            Path(config.paths["models_dir"]) / "xlm_roberta_sensitive_filter",  # 默认模型路径
            Path("models") / "xlm_roberta_sensitive_filter",  # 备用路径
        ]
        
        # 查找存在的模型路径
        for path in possible_paths:
            if path.exists() and (path / "config.json").exists():
                # 检查模型文件的修改时间
                config_file = path / "config.json"
                model_time = datetime.fromtimestamp(config_file.stat().st_mtime)
                
                print(f"✅ 找到模型: {path}")
                print(f"   训练时间: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                self.model_path = str(path)
                return True
        
        print("❌ 未找到训练好的模型")
        print("   请先运行训练脚本: python train.py")
        return False
    
    def load_model(self):
        """加载模型"""
        if not self.model_path:
            if not self.find_latest_model():
                return False
        
        try:
            print(f"🚀 正在加载模型: {self.model_path}")
            
            self.inference_engine = SensitiveWordInference(
                model_path=self.model_path,
                use_rules=False,  # 只使用AI模型
                use_ai=True
            )
            
            print("✅ 模型加载成功!")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def show_model_info(self):
        """显示模型信息"""
        if not self.inference_engine:
            return
        
        print("\n" + "="*50)
        print("📊 模型信息")
        print("="*50)
        print(f"模型路径: {self.model_path}")
        print(f"设备: {self.inference_engine.device}")
        print(f"最大长度: {config.model_config['max_length']}")
        print(f"置信度阈值: {config.inference_config.get('confidence_threshold', 0.5)}")
        print("="*50)
    
    def predict_text(self, text):
        """预测单个文本"""
        if not self.inference_engine:
            print("❌ 模型未加载")
            return None
        
        try:
            result = self.inference_engine.predict_single(text, return_probabilities=True)
            return result
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def format_result(self, result, text):
        """格式化显示结果"""
        if not result:
            return
        
        print(f"\n📝 输入文本: {text}")
        print("-" * 50)
        
        # 基本结果
        is_sensitive = result.get('is_sensitive', False)
        confidence = result.get('confidence', 0)
        final_decision = result.get('final_decision', 'unknown')
        
        # 状态显示
        status_icon = "🚨" if is_sensitive else "✅"
        status_text = "敏感内容" if is_sensitive else "正常内容"
        
        print(f"{status_icon} 检测结果: {status_text}")
        print(f"🎯 置信度: {confidence:.4f}")
        print(f"📊 最终决策: {final_decision}")
        
        # AI详细结果
        ai_result = result.get('ai_result')
        if ai_result:
            print(f"\n🤖 AI模型详情:")
            print(f"   预测类别: {ai_result.get('prediction', 'N/A')}")
            print(f"   标签: {ai_result.get('label', 'N/A')}")
            print(f"   置信度: {ai_result.get('confidence', 0):.4f}")
            
            # 概率分布
            probabilities = ai_result.get('probabilities', {})
            if probabilities:
                print(f"   概率分布:")
                for label, prob in probabilities.items():
                    print(f"     {label}: {prob:.4f}")
        
        print("-" * 50)
    
    def run_batch_test(self):
        """运行批量测试"""
        print("\n🧪 批量测试模式")
        print("请输入多个文本，每行一个，输入空行结束:")
        
        texts = []
        while True:
            text = input("文本: ").strip()
            if not text:
                break
            texts.append(text)
        
        if not texts:
            print("❌ 没有输入文本")
            return
        
        print(f"\n🚀 正在处理 {len(texts)} 个文本...")
        
        try:
            results = self.inference_engine.predict_batch(texts)
            
            print(f"\n📊 批量测试结果:")
            print("="*60)
            
            sensitive_count = 0
            for i, result in enumerate(results, 1):
                text = result['text']
                is_sensitive = result['is_sensitive']
                confidence = result['confidence']
                
                if is_sensitive:
                    sensitive_count += 1
                
                status_icon = "🚨" if is_sensitive else "✅"
                status_text = "敏感" if is_sensitive else "正常"
                
                print(f"{i:2d}. {status_icon} {text[:30]}{'...' if len(text) > 30 else ''}")
                print(f"     结果: {status_text} (置信度: {confidence:.4f})")
            
            print("="*60)
            print(f"📈 统计信息:")
            print(f"   总数: {len(results)}")
            print(f"   敏感: {sensitive_count}")
            print(f"   正常: {len(results) - sensitive_count}")
            print(f"   敏感比例: {sensitive_count/len(results)*100:.1f}%")
            
        except Exception as e:
            print(f"❌ 批量测试失败: {e}")
    
    def run_interactive_mode(self):
        """运行交互式模式"""
        print("\n💬 交互式测试模式")
        print("输入文本进行检测，输入 'quit' 或 'exit' 退出")
        print("输入 'batch' 进入批量测试模式")
        print("输入 'info' 查看模型信息")
        print("-" * 50)
        
        while True:
            try:
                text = input("\n请输入文本: ").strip()
                
                if not text:
                    continue
                
                # 特殊命令
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见!")
                    break
                elif text.lower() == 'batch':
                    self.run_batch_test()
                    continue
                elif text.lower() == 'info':
                    self.show_model_info()
                    continue
                
                # 预测文本
                result = self.predict_text(text)
                self.format_result(result, text)
                
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，再见!")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def run_preset_tests(self):
        """运行预设测试用例"""
        print("\n🧪 运行预设测试用例...")
        
        test_cases = [
            "这是一段正常的文本内容",
            "今天天气很好，适合出门散步",
            "我喜欢学习人工智能技术",
            "Python是一门很棒的编程语言",
            "机器学习让世界变得更美好",
            "深度学习模型的准确率很高",
            "自然语言处理技术发展迅速",
            "这个项目的代码写得很好",
            "数据科学是一个有趣的领域",
            "算法优化可以提升性能"
        ]
        
        print(f"正在测试 {len(test_cases)} 个预设用例...")
        
        try:
            results = self.inference_engine.predict_batch(test_cases)
            
            print(f"\n📊 预设测试结果:")
            print("="*70)
            
            for i, result in enumerate(results, 1):
                text = result['text']
                is_sensitive = result['is_sensitive']
                confidence = result['confidence']
                
                status_icon = "🚨" if is_sensitive else "✅"
                status_text = "敏感" if is_sensitive else "正常"
                
                print(f"{i:2d}. {text}")
                print(f"    {status_icon} {status_text} (置信度: {confidence:.4f})")
            
            # 统计信息
            sensitive_count = sum(1 for r in results if r['is_sensitive'])
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            print("="*70)
            print(f"📈 统计信息:")
            print(f"   敏感文本: {sensitive_count}/{len(results)}")
            print(f"   平均置信度: {avg_confidence:.4f}")
            
        except Exception as e:
            print(f"❌ 预设测试失败: {e}")
    
    def show_menu(self):
        """显示主菜单"""
        print("\n" + "="*50)
        print("🎯 敏感词检测 - 交互式测试")
        print("="*50)
        print("1. 交互式测试 (推荐)")
        print("2. 批量测试")
        print("3. 预设测试用例")
        print("4. 查看模型信息")
        print("5. 退出")
        print("="*50)
    
    def run(self):
        """运行测试器"""
        print("🚀 启动交互式模型测试器...")
        
        # 加载模型
        if not self.load_model():
            return
        
        # 显示模型信息
        self.show_model_info()
        
        # 主循环
        while True:
            self.show_menu()
            
            try:
                choice = input("请选择操作 (1-5): ").strip()
                
                if choice == '1':
                    self.run_interactive_mode()
                elif choice == '2':
                    self.run_batch_test()
                elif choice == '3':
                    self.run_preset_tests()
                elif choice == '4':
                    self.show_model_info()
                elif choice == '5':
                    print("👋 再见!")
                    break
                else:
                    print("❌ 无效选择，请输入 1-5")
                    
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，再见!")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")

def main():
    """主函数"""
    tester = InteractiveModelTester()
    tester.run()

if __name__ == "__main__":
    main()
