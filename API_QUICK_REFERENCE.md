# Safe-Text API 快速参考

## 🚀 基础信息
- **地址**: `http://localhost:9900`
- **格式**: JSON
- **认证**: Bearer Token
- **Token**: `safe-text-api-2025-9d8f7e6c5b4a3210`
- **当前模型**: `outputs/training_20250904_114529/final_model`

## 📡 核心接口

### 1️⃣ 文本检测 (主要接口)
```bash
POST /predict
{
  "text": "要检测的文本内容"
}
```
**响应**:
```json
{
  "is_sensitive": false,
  "confidence": 0.6746,
  "label": "正常内容",
  "text": "要检测的文本内容"
}
```

### 2️⃣ 批量检测
```bash
POST /predict/batch
{
  "texts": ["文本1", "文本2", "文本3"]
}
```

### 3️⃣ 健康检查
```bash
GET /health
```

### 4️⃣ 模型状态
```bash
GET /model/status
```

## ⚡ 快速测试
```bash
# 认证信息（无需Token）
curl -X GET http://localhost:9900/auth/info

# 正常文本
curl -X POST http://localhost:9900/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer safe-text-api-2025-9d8f7e6c5b4a3210" \
  -d '{"text": "今天天气很好"}'

# 敏感文本  
curl -X POST http://localhost:9900/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer safe-text-api-2025-9d8f7e6c5b4a3210" \
  -d '{"text": "政府腐败严重，应该推翻政权"}'
```

## 🎯 关键字段
- `is_sensitive`: 是否敏感 (boolean)
- `confidence`: 置信度 0.0-1.0 (float) 
- `label`: 标签 "正常内容"/"敏感内容"
