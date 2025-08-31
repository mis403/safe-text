# 🛡️ Safe-Text - 智能文本安全检测系统

一个简洁高效的文本安全检测系统，集成了AI模型训练和API服务功能。基于XLM-RoBERTa模型，专为中英文敏感内容检测优化。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

## 🌟 特性

- **🤖 AI驱动**: 使用XLM-RoBERTa进行语义理解的敏感内容检测
- **🚀 简洁高效**: 集成训练和推理功能，一体化解决方案
- **📡 API服务**: 提供RESTful API接口，支持单文本和批量检测
- **⚙️ 易于配置**: YAML配置文件，轻松自定义参数
- **🌐 多语言支持**: 针对中文和英文内容优化
- **🍎 Apple Silicon优化**: 原生支持Mac M1/M2的MPS加速

## 🏗️ 项目结构

```
safe-text/
├── README.md                    # 项目说明
├── api_server.py               # API服务器（主要入口）
├── train.py                    # 训练脚本
├── requirements.txt            # 依赖包
├── config/
│   └── settings.py             # 配置管理
├── src/
│   ├── models/                 # AI模型组件
│   │   ├── trainer.py          # 模型训练
│   │   └── inference.py        # 模型推理
│   ├── data/                   # 数据处理
│   │   ├── preprocessor.py     # 数据预处理
│   │   └── dataset.py          # PyTorch数据集
│   └── utils/                  # 工具函数
│       └── logger.py           # 日志工具
├── data/                       # 数据目录
├── models/                     # 模型存储
├── outputs/                    # 输出文件
└── logs/                       # 日志文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd safe-text

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备训练数据

训练数据应为CSV格式，包含两列：
- `text`: 文本内容
- `label`: 标签 (0=正常, 1=敏感)

示例：
```csv
text,label
这是正常的文本内容,0
这是敏感的内容,1
```

### 3. 训练模型

```bash
# 使用自定义数据训练
python train.py --input-data your_training_data.csv

# 使用默认数据训练（如果已有data/train.csv等文件）
python train.py
```

### 4. 启动API服务

```bash
python api_server.py
```

服务启动后，访问 http://localhost:8080

### 5. 交互式测试模型

```bash
# 完整交互式测试（推荐）
python interactive_test.py

# 快速测试模式
python quick_test.py
```

## 📡 API接口

### 健康检查
```bash
GET /health
```

### 模型状态
```bash
GET /model/status
```

### 单文本检测
```bash
POST /predict
Content-Type: application/json

{
    "text": "要检测的文本内容",
    "include_details": false  # 可选，是否返回详细信息
}
```

响应：
```json
{
    "text": "要检测的文本内容",
    "is_sensitive": false,
    "confidence": 0.8234,
    "label": "正常内容"
}
```

### 批量文本检测
```bash
POST /predict/batch
Content-Type: application/json

{
    "texts": ["文本1", "文本2", "文本3"]
}
```

### 训练模型
```bash
POST /train
Content-Type: application/json

{
    "input_data": "path/to/training_data.csv",  # 可选
    "model_name": "xlm-roberta-base"            # 可选
}
```

## 🧪 交互式测试

### 完整交互式测试 (`interactive_test.py`)

这是推荐的测试方式，提供丰富的功能：

```bash
python interactive_test.py
```

**功能特性：**
- 🔍 **自动模型检测**: 自动查找最新训练的模型
- 💬 **交互式测试**: 逐个输入文本进行实时检测
- 📦 **批量测试**: 一次输入多个文本进行批量检测
- 🧪 **预设测试**: 运行预定义的测试用例
- 📊 **详细结果**: 显示置信度、概率分布等详细信息
- 📈 **统计分析**: 提供测试结果的统计信息

**使用流程：**
1. 脚本自动查找最新训练的模型
2. 选择测试模式（交互式/批量/预设）
3. 输入文本或选择测试用例
4. 查看详细的检测结果和统计信息

### 快速测试 (`quick_test.py`)

简化版本，适合快速验证：

```bash
python quick_test.py
```

**特点：**
- ⚡ **快速启动**: 最小化的界面，快速加载
- 🎯 **专注测试**: 只提供基本的文本检测功能
- 📝 **简洁输出**: 显示核心结果信息

### 模型自动检测逻辑

两个测试脚本都会按以下优先级自动查找模型：

1. `ultimate_xlm_roberta_model/` - 最新训练的模型
2. `models/xlm_roberta_sensitive_filter/` - 默认模型路径

**模型信息显示：**
- 模型路径和训练时间
- 使用的设备（CPU/GPU/MPS）
- 模型配置参数
- 置信度阈值设置

## 💡 使用示例

### 交互式测试示例

```bash
$ python interactive_test.py

🚀 启动交互式模型测试器...
🔍 正在查找最新训练的模型...
✅ 找到模型: ultimate_xlm_roberta_model
   训练时间: 2024-01-15 14:30:25
🚀 正在加载模型: ultimate_xlm_roberta_model
✅ 模型加载成功!

==================================================
📊 模型信息
==================================================
模型路径: ultimate_xlm_roberta_model
设备: mps
最大长度: 512
置信度阈值: 0.5
==================================================

🎯 敏感词检测 - 交互式测试
==================================================
1. 交互式测试 (推荐)
2. 批量测试
3. 预设测试用例
4. 查看模型信息
5. 退出
==================================================
请选择操作 (1-5): 1

💬 交互式测试模式
输入文本进行检测，输入 'quit' 或 'exit' 退出
输入 'batch' 进入批量测试模式
输入 'info' 查看模型信息
--------------------------------------------------

请输入文本: 这是一段正常的文本内容

📝 输入文本: 这是一段正常的文本内容
--------------------------------------------------
✅ 检测结果: 正常内容
🎯 置信度: 0.8234
📊 最终决策: normal

🤖 AI模型详情:
   预测类别: 0
   标签: 正常内容
   置信度: 0.8234
   概率分布:
     正常内容: 0.8234
     敏感内容: 0.1766
--------------------------------------------------
```

### Python客户端示例

```python
import requests

# 单文本检测
response = requests.post('http://localhost:8080/predict', 
                        json={'text': '这是要检测的文本'})
result = response.json()
print(f"是否敏感: {result['is_sensitive']}")
print(f"置信度: {result['confidence']}")

# 批量检测
response = requests.post('http://localhost:8080/predict/batch',
                        json={'texts': ['文本1', '文本2', '文本3']})
results = response.json()
for item in results['results']:
    print(f"文本: {item['text']} -> {item['label']}")
```

### curl示例

```bash
# 单文本检测
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "这是要检测的文本内容"}'

# 批量检测
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["文本1", "文本2", "文本3"]}'

# 训练模型
curl -X POST http://localhost:8080/train \
  -H "Content-Type: application/json" \
  -d '{"input_data": "data/training_data.csv"}'
```

## ⚙️ 配置说明

主要配置在 `config/settings.py` 中：

```python
# 模型配置
model:
  name: "xlm-roberta-base"           # 预训练模型
  max_length: 512                   # 最大输入长度
  num_labels: 2                     # 分类数量

# 训练配置
training:
  batch_size: 8                     # 批次大小
  learning_rate: 2e-5               # 学习率
  num_epochs: 3                     # 训练轮数

# 推理配置
inference:
  batch_size: 16                    # 推理批次大小
  confidence_threshold: 0.5         # 置信度阈值
```

## 📋 系统要求

- **Python**: 3.8 或更高版本
- **操作系统**: macOS, Linux, Windows
- **内存**: 8GB+ RAM (推荐16GB)
- **存储**: 5GB+ 可用空间
- **GPU**: 可选 (支持CUDA/MPS)

## 🔧 开发说明

### 项目特点

1. **简化架构**: 移除了复杂的CLI系统和规则匹配，专注于AI模型
2. **一体化**: 训练和推理功能集成在同一个项目中
3. **API优先**: 提供标准的RESTful API接口
4. **易于部署**: 单个Python文件即可启动完整服务

### 核心组件

- `api_server.py`: 主要的API服务器，包含所有接口
- `train.py`: 独立的训练脚本
- `src/models/`: 模型相关代码（训练器和推理引擎）
- `src/data/`: 数据处理相关代码
- `config/`: 配置管理

## 🚨 注意事项

1. 首次运行需要下载预训练模型，可能需要一些时间
2. 训练过程会消耗较多内存和计算资源
3. 模型文件较大（约1.1GB），确保有足够存储空间
4. 建议在GPU环境下训练以提高速度

## 📈 性能指标

- **推理速度**: ~100 文本/秒 (CPU), ~500 文本/秒 (GPU)
- **内存使用**: ~2GB (模型加载), ~4GB (训练时)
- **模型大小**: 1.1GB (XLM-RoBERTa base)
- **启动时间**: ~5-10秒 (模型加载)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件