#!/bin/bash

# 敏感词检测API服务器启动脚本

echo "🚀 启动敏感词检测API服务器..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python3 -c "import flask, torch, transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  警告: 缺少依赖，正在安装..."
    pip3 install -r requirements.txt
fi

# 检查模型文件
if [ -d "ultimate_xlm_roberta_model" ]; then
    echo "✅ 找到训练好的模型"
elif [ -d "models/xlm_roberta_sensitive_filter" ]; then
    echo "✅ 找到训练好的模型"
else
    echo "⚠️  警告: 未找到训练好的模型，请先训练模型"
    echo "   运行: python3 train.py --input-data your_data.csv"
fi

# 启动服务器
echo "🌐 启动API服务器..."
python3 api_server.py
