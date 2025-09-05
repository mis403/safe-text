#!/usr/bin/env python3
"""
Gunicorn配置文件 - 多进程部署配置
支持20并发，4个工作进程
"""

import multiprocessing
import os

# 绑定地址和端口
bind = "0.0.0.0:9900"

# 工作进程配置
workers = 4                          # 4个工作进程
worker_class = "sync"               # 同步工作模式
worker_connections = 8              # 每个进程最多8个连接
max_requests = 1000                 # 进程处理最大请求数后重启
max_requests_jitter = 50            # 请求数抖动范围

# 超时配置
timeout = 120                       # 工作进程超时时间(秒)
keepalive = 5                       # Keep-Alive连接时间

# 应用配置
preload_app = False                 # 不预加载应用(每个进程独立加载模型)
reload = False                      # 生产环境不自动重载

# 进程管理
daemon = False                      # 不以守护进程运行
pidfile = "/tmp/gunicorn_safe_text.pid"
user = None
group = None
tmp_upload_dir = None

# 日志配置
loglevel = "info"
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# SSL配置 (如需要)
# keyfile = None
# certfile = None

# 工作进程内存限制
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

def when_ready(server):
    """服务器启动完成回调"""
    server.log.info("🚀 Safe-Text API 多进程服务器已启动")
    server.log.info(f"⚙️  工作进程数: {workers}")
    server.log.info(f"📡 服务地址: http://0.0.0.0:9900")
    server.log.info(f"🔄 最大并发: {workers * worker_connections}")

def worker_int(worker):
    """工作进程中断处理"""
    worker.log.info(f"🔄 工作进程 {worker.pid} 收到中断信号")

def pre_fork(server, worker):
    """工作进程启动前回调"""
    server.log.info(f"🔧 启动工作进程 {worker.age}")

def post_fork(server, worker):
    """工作进程启动后回调"""
    server.log.info(f"✅ 工作进程 {worker.pid} 已启动")

def worker_exit(server, worker):
    """工作进程退出回调"""
    server.log.info(f"👋 工作进程 {worker.pid} 已退出")
