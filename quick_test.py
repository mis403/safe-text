#!/usr/bin/env python3
"""
快速测试脚本 - 简化版交互式测试
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.inference import SensitiveWordInference
from config.settings import config

def find_latest_model():
    """查找最新模型"""
    paths = [
        Path("ultimate_xlm_roberta_model"),
        Path(config.paths["models_dir"]) / "xlm_roberta_sensitive_filter",
    ]
    
    for path in paths:
        if path.exists() and (path / "config.json").exists():
            return str(path)
    return None

def main():
    """主函数"""
    print("🚀 快速测试模式")
    
    # 查找并加载模型
    model_path = find_latest_model()
    if not model_path:
        print("❌ 未找到训练好的模型，请先运行: python train.py")
        return
    
    print(f"✅ 使用模型: {model_path}")
    
    try:
        # 初始化推理引擎
        engine = SensitiveWordInference(model_path=model_path, use_rules=False, use_ai=True)
        print("✅ 模型加载成功!")
        
        # 交互式测试
        print("\n💬 输入文本进行检测 (输入 'quit' 退出):")
        print("-" * 40)
        
        while True:
            text = input("\n文本: ").strip()
            
            if not text or text.lower() in ['quit', 'exit', 'q']:
                print("👋 再见!")
                break
            
            # 预测
            result = engine.predict_single(text)
            
            # 显示结果
            is_sensitive = result['is_sensitive']
            confidence = result['confidence']
            
            status_icon = "🚨" if is_sensitive else "✅"
            status_text = "敏感内容" if is_sensitive else "正常内容"
            
            print(f"{status_icon} 结果: {status_text} (置信度: {confidence:.4f})")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
