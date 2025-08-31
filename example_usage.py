#!/usr/bin/env python3
"""
使用示例：如何调用敏感词检测API
"""

import requests
import json
import time

# API服务器地址
API_BASE = "http://localhost:8080"

def test_health():
    """测试健康检查接口"""
    print("🔍 测试健康检查...")
    response = requests.get(f"{API_BASE}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_model_status():
    """测试模型状态接口"""
    print("📊 检查模型状态...")
    response = requests.get(f"{API_BASE}/model/status")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_single_prediction():
    """测试单文本预测"""
    print("🤖 测试单文本预测...")
    
    test_texts = [
        "这是一段正常的文本内容",
        "今天天气很好",
        "我喜欢学习人工智能",
    ]
    
    for text in test_texts:
        response = requests.post(f"{API_BASE}/predict", 
                               json={"text": text, "include_details": True})
        if response.status_code == 200:
            result = response.json()
            print(f"文本: {result['text']}")
            print(f"结果: {result['label']} (置信度: {result['confidence']:.4f})")
            print()
        else:
            print(f"错误: {response.status_code} - {response.text}")
            print()

def test_batch_prediction():
    """测试批量预测"""
    print("📦 测试批量预测...")
    
    texts = [
        "这是第一段文本",
        "这是第二段文本", 
        "这是第三段文本",
        "今天是个好日子",
        "学习使我快乐"
    ]
    
    response = requests.post(f"{API_BASE}/predict/batch", 
                           json={"texts": texts})
    
    if response.status_code == 200:
        result = response.json()
        print(f"总数: {result['total']}")
        for item in result['results']:
            print(f"  {item['text']} -> {item['label']} ({item['confidence']:.4f})")
        print()
    else:
        print(f"错误: {response.status_code} - {response.text}")
        print()

def test_training():
    """测试训练接口（如果有训练数据）"""
    print("🎓 测试训练接口...")
    
    # 检查是否有训练数据
    import os
    training_data_path = "data/final_enhanced_training_data.csv"
    
    if os.path.exists(training_data_path):
        print(f"使用训练数据: {training_data_path}")
        response = requests.post(f"{API_BASE}/train", 
                               json={"input_data": training_data_path})
        
        if response.status_code == 200:
            result = response.json()
            print(f"训练结果: {result['message']}")
            print(f"模型路径: {result['model_path']}")
        else:
            print(f"训练失败: {response.status_code} - {response.text}")
    else:
        print(f"训练数据文件不存在: {training_data_path}")
        print("跳过训练测试")
    print()

def main():
    """主函数"""
    print("🚀 敏感词检测API测试")
    print("=" * 50)
    
    try:
        # 测试各个接口
        test_health()
        test_model_status()
        test_single_prediction()
        test_batch_prediction()
        
        # 可选：测试训练（需要较长时间）
        # test_training()
        
        print("✅ 所有测试完成!")
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务器")
        print("请确保API服务器正在运行: python api_server.py")
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")

if __name__ == "__main__":
    main()
