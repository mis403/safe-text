# Safe-Text API 接口文档

## 📋 概述

Safe-Text 是一个基于AI的智能文本安全检测系统，使用XLM-RoBERTa模型对中文文本进行敏感内容识别。本API提供了完整的文本安全检测服务。

## 🌐 基础信息

- **Base URL**: `http://localhost:9900` 或 `http://your-server-ip:9900`
- **协议**: HTTP/HTTPS
- **数据格式**: JSON
- **字符编码**: UTF-8
- **认证方式**: Bearer Token

## 🔐 API认证

所有API请求都需要在请求头中包含有效的Bearer Token：

```
Authorization: Bearer safe-text-api-2025-9d8f7e6c5b4a3210
```

### 认证配置
- **Token位置**: 配置文件 `config/training.yaml` 中的 `security.api_token`
- **当前Token**: `safe-text-api-2025-9d8f7e6c5b4a3210`
- **启用状态**: 默认启用（可在配置文件中设置 `security.token_required: false` 禁用）

## 📝 通用响应格式

### 成功响应
- **状态码**: 200
- **Content-Type**: `application/json`

### 错误响应
- **状态码**: 400, 401, 403, 500, 503
- **格式**:
```json
{
  "error": "错误描述信息",
  "message": "详细错误说明"
}
```

### 认证错误
- **401 Unauthorized**: 缺少或格式错误的Authorization头
- **403 Forbidden**: 无效的API Token

---

## 🔗 API 端点详情

### 0. 认证信息查询

获取API认证相关信息（无需Token）。

**端点**: `GET /auth/info`

#### 请求示例
```bash
curl -X GET http://localhost:9900/auth/info
```

#### 响应示例
```json
{
  "token_required": true,
  "auth_method": "Bearer Token",
  "header_format": "Authorization: Bearer <token>",
  "message": "请联系管理员获取API Token"
}
```

---

### 1. 健康检查

检查API服务状态和模型加载情况。

**端点**: `GET /health`

#### 请求示例
```bash
curl -X GET http://localhost:8080/health
```

#### 响应示例
```json
{
  "status": "healthy",
  "service": "safe-text",
  "model_status": "loaded"
}
```

#### 响应字段说明
| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | 服务状态 (`healthy`/`unhealthy`) |
| service | string | 服务名称 |
| model_status | string | 模型状态 (`loaded`/`not_loaded`) |

---

