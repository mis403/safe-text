#!/usr/bin/env python3
"""
Safe-Text API服务器
智能文本安全检测系统 - 支持训练和推理功能
"""

from flask import Flask, request, jsonify
import sys
import os
import argparse
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.inference import SensitiveWordInference
from src.models.trainer import SensitiveWordTrainer
from src.data import DataProcessor
from src.utils.model_finder import find_latest_model, find_all_models, get_model_info
from config.settings import config
from config.simple_config import load_training_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局推理引擎 (每个工作进程独立实例)
inference_engine: Optional[SensitiveWordInference] = None

# 进程初始化标记
_process_initialized = False

# 加载安全配置
try:
    training_config = load_training_config()
    security_config = training_config.get('security', {})
    API_TOKEN = security_config.get('api_token', 'safe-text-api-2025-default')
    TOKEN_REQUIRED = security_config.get('token_required', True)
except Exception as e:
    logger.warning(f"无法加载安全配置，使用默认值: {e}")
    API_TOKEN = 'safe-text-api-2025-default'
    TOKEN_REQUIRED = True

# 无需认证的端点列表
EXEMPT_ENDPOINTS = ['/auth/info']

@app.before_request
def authenticate_request():
    """全局认证中间件 - 在每个请求前执行"""
    # 如果认证被禁用，直接通过
    if not TOKEN_REQUIRED:
        return None
    
    # 检查是否是免认证端点
    if request.endpoint and any(request.path.startswith(exempt) for exempt in EXEMPT_ENDPOINTS):
        return None
    
    # 获取Authorization头
    auth_header = request.headers.get('Authorization')
    
    if not auth_header:
        return jsonify({
            "error": "缺少Authorization头",
            "message": "请在请求头中包含: Authorization: Bearer <token>"
        }), 401
    
    # 检查Bearer格式
    if not auth_header.startswith('Bearer '):
        return jsonify({
            "error": "无效的Authorization格式",
            "message": "格式应为: Authorization: Bearer <token>"
        }), 401
    
    # 提取token
    token = auth_header[7:]  # 移除 "Bearer " 前缀
    
    # 验证token
    if token != API_TOKEN:
        return jsonify({
            "error": "无效的访问令牌",
            "message": "请提供有效的API Token"
        }), 403
    
    # 认证成功，继续处理请求
    return None

# 模型查找功能已移至 src.utils.model_finder 模块

def init_inference_engine():
    """初始化推理引擎 - 支持多进程独立初始化"""
    global inference_engine, _process_initialized
    
    # 防止同一进程重复初始化
    if _process_initialized and inference_engine is not None:
        return True
    
    try:
        # 获取当前进程ID
        import os
        process_id = os.getpid()
        logger.info(f"🔍 进程 {process_id}: 查找最新训练的模型...")
        
        # 查找最新模型
        model_path = find_latest_model()
        if not model_path:
            logger.warning(f"进程 {process_id}: 训练好的模型不存在，请先训练模型")
            return False
        
        # 初始化推理引擎
        logger.info(f"🤖 进程 {process_id}: 加载模型 {model_path}")
        inference_engine = SensitiveWordInference(
            model_path=model_path,
            use_rules=False,  # 只使用AI模型
            use_ai=True
        )
        _process_initialized = True
        
        logger.info(f"✅ 进程 {process_id}: 推理引擎初始化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 进程 {process_id}: 推理引擎初始化失败: {e}")
        return False

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
    # 确保推理引擎已初始化（多进程安全）
    if not init_inference_engine():
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
    # 确保推理引擎已初始化（多进程安全）
    if not init_inference_engine():
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
        
        # 重新初始化推理引擎以使用最新训练的模型
        global inference_engine
        inference_engine = None
        if init_inference_engine():
            logger.info("推理引擎已更新为最新训练的模型")
        else:
            logger.warning("推理引擎更新失败")
        
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
    current_model_path = find_latest_model()
    
    # 检查所有可能的模型路径
    model_paths = {
        "current_model": {
            "path": current_model_path or "未找到",
            "exists": current_model_path is not None,
            "is_active": True
        },
        "checkpoint_model": {
            "path": "ultimate_xlm_roberta_model/checkpoint-100",
            "exists": Path("ultimate_xlm_roberta_model/checkpoint-100").exists()
        },
        "ultimate_model": {
            "path": "ultimate_xlm_roberta_model",
            "exists": Path("ultimate_xlm_roberta_model").exists()
        },
        "default_model": {
            "path": str(Path(config.paths["models_dir"]) / "xlm_roberta_sensitive_filter"),
            "exists": (Path(config.paths["models_dir"]) / "xlm_roberta_sensitive_filter").exists()
        }
    }
    
    status = {
        "inference_engine_loaded": inference_engine is not None,
        "model_paths": model_paths,
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

@app.route('/auth/info', methods=['GET'])
def auth_info():
    """获取认证信息（无需Token）"""
    return jsonify({
        "token_required": TOKEN_REQUIRED,
        "auth_method": "Bearer Token",
        "header_format": "Authorization: Bearer <token>",
        "message": "请联系管理员获取API Token" if TOKEN_REQUIRED else "当前无需认证"
    })

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Safe-Text API服务器')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=9900, help='服务器端口号 (默认: 9900)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()
    
    # 使用命令行参数
    host = args.host
    port = args.port
    debug = args.debug
    
    print("🚀 启动敏感词检测API服务器...")
    
    # 显示安全配置
    if TOKEN_REQUIRED:
        print("🔐 API安全认证已启用")
        print(f"🔑 API Token: {API_TOKEN}")
        print("📋 请求头格式: Authorization: Bearer <token>")
    else:
        print("⚠️  API安全认证已禁用")
    
    # 尝试初始化推理引擎
    if init_inference_engine():
        print("✅ 推理引擎加载成功")
    else:
        print("⚠️  推理引擎未加载，请先训练模型")
    
    print(f"📡 API服务器运行在: http://{host}:{port}")
    print("\n可用接口:")
    print("  GET  /auth/info       - 认证信息 (无需Token)")
    print("  GET  /health          - 健康检查")
    print("  GET  /model/status    - 模型状态")
    print("  POST /predict         - 单文本预测")
    print("  POST /predict/batch   - 批量文本预测")
    print("  POST /train           - 训练模型")
    
    # 启动服务
    app.run(
        host=host,
        port=port,
        debug=debug
    )
