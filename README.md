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
├── train.py                    # 初始训练脚本
├── incremental_train.py        # 增量训练脚本
├── interactive_test.py         # 交互式测试
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
│       ├── logger.py           # 日志工具
│       └── model_finder.py     # 模型查找工具
├── data/                       # 数据目录
│   ├── formatted_sensitive_data.csv  # 主训练数据
│   ├── additional_training_data.csv  # 增量训练示例
│   ├── train.csv               # 训练集
│   ├── val.csv                 # 验证集
│   └── test.csv                # 测试集
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

#### 初始训练

```bash
# 使用自定义数据训练（完整模式，包含防过拟合监控）
python3 train.py --input-data data/data.csv --deterministic --simple-mode

--deterministic
作用: 启用完全确定性训练
含义: 设置所有随机种子，确保训练结果可重现
 --simple-mode
作用: 简化输出模式
含义: 减少详细的配置信息显示，只显示关键信息
好处: 输出更简洁，专注于训练过程

# 使用默认数据训练（如果已有data/train.csv等文件）
python train.py

# 简化模式（不显示详细配置）
python train.py --simple-mode

# 清理旧模型重新训练
python train.py --clear-models

# 跳过过拟合监控（快速训练）
python train.py --skip-monitoring
```

#### 增量训练（追加训练）

当你需要在已有模型基础上添加新的训练数据时：

```bash
# 基本增量训练（自动找到最新模型）
python incremental_train.py --additional-data data/new_training_data.csv

# 指定基础模型进行增量训练
python incremental_train.py --additional-data data/new_training_data.csv --base-model models/xlm_roberta_sensitive_filter

# 自定义学习率和训练轮数
python incremental_train.py --additional-data data/new_training_data.csv --learning-rate 5e-6 --epochs 1
```

### 4. 启动API服务

```bash
python api_server.py
```

服务启动后，访问 http://localhost:8080

### 5. 查看训练历史和测试模型

```bash
# 查看所有训练历史
python show_training_history.py

# 交互式测试当前模型
python interactive_test.py
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

### 交互式测试 (`interactive_test.py`)

这是推荐的测试方式，提供丰富的功能：

```bash
python3 interactive_test.py
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

### 模型自动检测逻辑

测试脚本会按以下优先级自动查找模型：

1. `ultimate_xlm_roberta_model/` - 最新训练的模型
2. `models/xlm_roberta_sensitive_filter/` - 默认模型路径

**模型信息显示：**
- 模型路径和训练时间
- 使用的设备（CPU/GPU/MPS）
- 模型配置参数
- 置信度阈值设置

## 🔄 增量训练详细指南

### 什么是增量训练？

增量训练（Incremental Training）是在已有训练好的模型基础上，添加新的训练数据进行进一步训练的过程。这种方式可以：

- 📈 **提升模型性能**: 在特定领域或新数据上提高准确率
- ⚡ **节省训练时间**: 无需从头开始训练
- 💾 **保留原有知识**: 在新数据基础上保持原有模型的能力
- 🎯 **针对性优化**: 针对特定类型的敏感内容进行优化

### 📋 增量训练完整流程

#### 步骤1: 准备新的训练数据

创建CSV格式的新训练数据文件：

```csv
text,label
"新的敏感内容示例",1
"新的正常内容示例",0
"更多训练数据...",0
```

**数据准备建议：**
- 确保数据质量高，标注准确
- 保持敏感(1)和正常(0)数据的平衡
- 每个类别至少准备50-100条数据
- 避免与原有数据重复

#### 步骤2: 选择合适的训练参数

**学习率设置：**
```bash
# 推荐学习率范围
--learning-rate 1e-5    # 标准增量训练学习率
--learning-rate 5e-6    # 更保守的学习率，适合微调
--learning-rate 2e-5    # 较大的学习率，适合大量新数据
```

**训练轮数设置：**
```bash
--epochs 1    # 少量新数据，避免过拟合
--epochs 2    # 标准设置，适合大多数情况
--epochs 3    # 大量新数据或需要显著改进时
```

#### 步骤3: 执行增量训练

**基本用法：**
```bash
# 最简单的增量训练
python incremental_train.py --additional-data data/new_data.csv
```

**完整参数示例：**
```bash
# 完整的增量训练命令
python incremental_train.py \
  --additional-data data/new_sensitive_words.csv \
  --base-model models/xlm_roberta_sensitive_filter \
  --learning-rate 5e-6 \
  --epochs 2 \
  --output-dir outputs/incremental_training_$(date +%Y%m%d_%H%M%S)
```

#### 步骤4: 验证训练结果

训练完成后，系统会显示：
```
✅ 增量训练完成!
📁 模型保存到: models/xlm_roberta_sensitive_filter
📊 训练日志: outputs/training_20240115_143025
🎯 测试准确率: 0.9845
📈 测试F1分数: 0.9843
```

### 🔧 增量训练参数详解

| 参数 | 说明 | 默认值 | 推荐设置 |
|------|------|--------|----------|
| `--additional-data` | 新增训练数据文件路径 | 必需 | - |
| `--base-model` | 基础模型路径 | 自动检测最新 | 使用默认 |
| `--original-data` | 原始训练数据路径 | `data/formatted_sensitive_data.csv` | 保持默认 |
| `--learning-rate` | 学习率 | `1e-5` | `5e-6` (保守) 或 `1e-5` (标准) |
| `--epochs` | 训练轮数 | `2` | `1-2` 轮 |
| `--output-dir` | 输出目录 | 自动生成 | 使用默认 |

### ⚠️ 增量训练注意事项

**学习率选择原则：**
- 🔸 **1e-5**: 标准增量训练学习率，适合大多数情况
- 🔸 **5e-6**: 更保守的学习率，适合微调或少量数据
- 🔸 **2e-5**: 较大学习率，适合大量新数据或需要显著改进
- ❌ **避免使用过大学习率** (>5e-5)，可能导致灾难性遗忘

**训练轮数建议：**
- 🔸 **1轮**: 少量新数据 (<100条)
- 🔸 **2轮**: 标准设置，适合大多数情况
- 🔸 **3轮**: 大量新数据 (>500条) 或需要显著改进
- ❌ **避免过多轮数**，可能导致过拟合

**数据质量要求：**
- ✅ 确保新数据标注准确
- ✅ 保持类别平衡 (敏感:正常 ≈ 1:1)
- ✅ 避免与原有数据重复
- ✅ 每个类别至少50条数据

### 📊 增量训练效果监控

**训练过程监控：**
```bash
# 训练过程中会显示实时指标
{'loss': 0.1658, 'learning_rate': 5e-06, 'epoch': 0.5}
{'eval_loss': 0.1290, 'eval_accuracy': 0.9658, 'eval_f1': 0.9658}
```

**最终评估指标：**
- **准确率 (Accuracy)**: 总体预测正确率
- **F1分数**: 综合考虑精确率和召回率
- **精确率 (Precision)**: 预测为敏感的内容中真正敏感的比例
- **召回率 (Recall)**: 真正敏感内容被正确识别的比例

### 🚀 增量训练最佳实践

1. **小步快跑**: 每次添加少量高质量数据，多次增量训练
2. **监控性能**: 每次训练后在测试集上验证效果
3. **备份模型**: 训练前备份当前最佳模型
4. **数据版本控制**: 记录每次增量训练使用的数据
5. **A/B测试**: 对比增量训练前后的模型效果

### 💡 常见问题解决

**Q: 增量训练后性能下降怎么办？**
A: 降低学习率 (如5e-6) 或减少训练轮数 (如1轮)

**Q: 新数据量很少怎么办？**
A: 使用更小的学习率 (5e-6) 和更少轮数 (1轮)

**Q: 想要快速适应新领域怎么办？**
A: 可以适当提高学习率 (2e-5) 但要密切监控过拟合

**Q: 如何避免灾难性遗忘？**
A: 使用较小学习率，确保新数据质量，避免过度训练

### 🎯 增量训练实际示例

假设你想要为模型添加一些网络安全相关的敏感词检测能力：

#### 1. 准备新数据
```bash
# 查看示例数据
head -5 data/additional_training_data.csv
```
```csv
text,label
"网络诈骗新手法曝光",1
"今天天气很好",0
"传播虚假疫情信息",1
"学习编程很有趣",0
```

#### 2. 执行增量训练
```bash
# 使用保守的学习率进行增量训练
python3 incremental_train.py \
  --additional-data data/additional_training_data.csv \
  --learning-rate 5e-6 \
  --epochs 2
```

#### 3. 训练过程输出
```
🚀 开始增量训练...
📊 合并训练数据: data/formatted_sensitive_data.csv + data/additional_training_data.csv
   原始数据: 7596 条
   新增数据: 20 条
   合并后数据: 7616 条
   去重后数据: 7616 条

🤖 使用基础模型: models/xlm_roberta_sensitive_filter
⚙️ 增量训练配置: 学习率=5e-06, 轮数=2

📈 训练进度:
{'loss': 0.0892, 'learning_rate': 5e-06, 'epoch': 1.0}
{'eval_loss': 0.0654, 'eval_accuracy': 0.9789, 'eval_f1': 0.9789}

✅ 增量训练完成!
📁 模型保存到: models/xlm_roberta_sensitive_filter
🎯 测试准确率: 0.9789
📈 测试F1分数: 0.9789
```

#### 4. 验证训练效果
```bash
# 使用交互式测试验证新模型
python3 interactive_test.py
```

测试新添加的敏感词：
```
请输入文本: 网络诈骗新手法曝光
✅ 检测结果: 敏感内容 (置信度: 0.9234)

请输入文本: 今天天气很好
✅ 检测结果: 正常内容 (置信度: 0.8876)
```

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

指定版本模型进行交互型测试
python3 interactive_test.py
outputs/
training_20250902_153103/
final_model




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