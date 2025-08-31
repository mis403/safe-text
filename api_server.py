#!/usr/bin/env python3
"""
Safe-Text API服务器
智能文本安全检测系统 - 支持训练和推理功能
"""

from flask import Flask, request, jsonify
import sys
import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.inference import SensitiveWordInference
from src.models.trainer import SensitiveWordTrainer
from src.data import DataProcessor
from config.settings import config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局推理引擎
inference_engine: Optional[SensitiveWordInference] = None

def init_inference_engine():
    """初始化推理引擎"""
    global inference_engine
    if inference_engine is None:
        try:
            # 检查模型是否存在
            model_path = Path(config.paths["models_dir"]) / "xlm_roberta_sensitive_filter"
            ultimate_model_path = Path("ultimate_xlm_roberta_model")
            
            if ultimate_model_path.exists():
                model_path = str(ultimate_model_path)
            elif not model_path.exists():
                logger.warning("训练好的模型不存在，请先训练模型")
                return False
            
            inference_engine = SensitiveWordInference(
                model_path=str(model_path),
                use_rules=False,  # 只使用AI模型
                use_ai=True
            )
            logger.info("推理引擎初始化成功")
            return True
        except Exception as e:
            logger.error(f"推理引擎初始化失败: {e}")
            return False
    return True

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    model_status = "loaded" if inference_engine else "not_loaded"
    return jsonify({
        "status": "healthy",
        "service": "safe-text",
        "model_status": model_status
    })

@app.route('/predict', methods=['POST'])
def predict_text():
    """敏感词检测接口"""
    if not inference_engine:
        return jsonify({
            "error": "模型未加载，请先训练模型或检查模型路径"
        }), 503
    
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                "error": "缺少必需的参数 'text'"
            }), 400
        
        text = data['text']
        if not isinstance(text, str) or not text.strip():
            return jsonify({
                "error": "文本内容不能为空"
            }), 400
        
        # 进行预测
        result = inference_engine.predict_single(text)
        
        # 格式化返回结果
        response = {
            "text": text,
            "is_sensitive": result['is_sensitive'],
            "confidence": round(result['confidence'], 4),
            "label": "敏感内容" if result['is_sensitive'] else "正常内容"
        }
        
        # 可选：返回详细的AI预测结果
        if data.get('include_details', False) and result.get('ai_result'):
            response['ai_details'] = result['ai_result']
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"预测过程中出现错误: {e}")
        return jsonify({
            "error": "服务器内部错误"
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """批量敏感词检测接口"""
    if not inference_engine:
        return jsonify({
            "error": "模型未加载，请先训练模型"
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                "error": "缺少必需的参数 'texts'"
            }), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                "error": "texts 必须是非空数组"
            }), 400
        
        # 限制批量大小
        max_batch_size = 100
        if len(texts) > max_batch_size:
            return jsonify({
                "error": f"批量大小不能超过 {max_batch_size}"
            }), 400
        
        # 批量预测
        results = inference_engine.predict_batch(texts)
        
        # 格式化返回结果
        response = {
            "total": len(results),
            "results": []
        }
        
        for result in results:
            response["results"].append({
                "text": result['text'],
                "is_sensitive": result['is_sensitive'],
                "confidence": round(result['confidence'], 4),
                "label": "敏感内容" if result['is_sensitive'] else "正常内容"
            })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"批量预测过程中出现错误: {e}")
        return jsonify({
            "error": "服务器内部错误"
        }), 500

@app.route('/train', methods=['POST'])
def train_model():
    """训练模型接口"""
    try:
        data = request.get_json() or {}
        
        # 获取训练参数
        input_data = data.get('input_data')
        model_name = data.get('model_name', 'xlm-roberta-base')
        
        # 检查训练数据
        if input_data:
            if not Path(input_data).exists():
                return jsonify({
                    "error": f"训练数据文件不存在: {input_data}"
                }), 400
        else:
            # 检查默认训练文件
            train_file = config.data_config["train_file"]
            if not Path(train_file).exists():
                return jsonify({
                    "error": "没有找到训练数据，请提供 input_data 参数或确保默认训练文件存在"
                }), 400
        
        logger.info("开始训练模型...")
        
        # 初始化数据处理器
        processor = DataProcessor()
        
        # 处理数据（如果提供了输入文件）
        if input_data:
            train_file = config.data_config["train_file"]
            val_file = config.data_config["val_file"]
            test_file = config.data_config["test_file"]
            
            processor.process_and_split(input_data, train_file, val_file, test_file)
        
        # 初始化训练器
        trainer = SensitiveWordTrainer(model_name)
        
        # 加载模型和分词器
        trainer.load_model_and_tokenizer()
        
        # 准备数据集
        train_file = config.data_config["train_file"]
        val_file = config.data_config["val_file"]
        test_file = config.data_config["test_file"]
        
        train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(
            train_file, val_file, test_file if Path(test_file).exists() else None
        )
        
        # 设置训练
        output_dir = trainer.setup_training(train_dataset, val_dataset)
        
        # 训练模型
        results = trainer.train()
        
        # 在测试集上评估
        test_results = {}
        if test_dataset:
            test_results = trainer.evaluate_on_test_set(test_dataset)
        
        # 保存模型
        model_path = trainer.save_model()
        
        logger.info("模型训练完成")
        
        # 重新初始化推理引擎
        global inference_engine
        inference_engine = None
        init_inference_engine()
        
        return jsonify({
            "message": "模型训练完成",
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "training_results": {
                "train_loss": results.get('train_loss', 0),
                "eval_loss": results.get('eval_loss', 0)
            },
            "test_results": test_results
        })
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        return jsonify({
            "error": f"训练失败: {str(e)}"
        }), 500

@app.route('/model/status', methods=['GET'])
def model_status():
    """获取模型状态"""
    model_path = Path(config.paths["models_dir"]) / "xlm_roberta_sensitive_filter"
    ultimate_model_path = Path("ultimate_xlm_roberta_model")
    
    status = {
        "inference_engine_loaded": inference_engine is not None,
        "model_paths": {
            "default_model": {
                "path": str(model_path),
                "exists": model_path.exists()
            },
            "ultimate_model": {
                "path": str(ultimate_model_path),
                "exists": ultimate_model_path.exists()
            }
        },
        "training_data": {
            "train_file": {
                "path": config.data_config["train_file"],
                "exists": Path(config.data_config["train_file"]).exists()
            },
            "val_file": {
                "path": config.data_config["val_file"],
                "exists": Path(config.data_config["val_file"]).exists()
            },
            "test_file": {
                "path": config.data_config["test_file"],
                "exists": Path(config.data_config["test_file"]).exists()
            }
        }
    }
    
    return jsonify(status)

if __name__ == '__main__':
    print("🚀 启动敏感词检测API服务器...")
    
    # 尝试初始化推理引擎
    if init_inference_engine():
        print("✅ 推理引擎加载成功")
    else:
        print("⚠️  推理引擎未加载，请先训练模型")
    
    print("📡 API服务器运行在: http://localhost:8080")
    print("\n可用接口:")
    print("  GET  /health          - 健康检查")
    print("  GET  /model/status    - 模型状态")
    print("  POST /predict         - 单文本预测")
    print("  POST /predict/batch   - 批量文本预测")
    print("  POST /train           - 训练模型")
    
    # 启动服务
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=False
    )
