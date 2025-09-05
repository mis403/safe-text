#!/bin/bash
"""
Safe-Text Multi-Process Server Startup Script
Automatically checks environment, starts service, and verifies status
"""

echo "🎯 Safe-Text Multi-Process Server Startup"
echo "=========================================="

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. Environment check
echo -e "${BLUE}1️⃣ Environment Check...${NC}"

# Check Python
if command -v python3 &> /dev/null; then
    echo -e "✅ Python3: $(python3 --version)"
else
    echo -e "${RED}❌ Python3 not installed${NC}"
    exit 1
fi

# Check project directory
if [ ! -f "api_server.py" ]; then
    echo -e "${RED}❌ Please run this script from project root directory${NC}"
    exit 1
fi

# Check dependencies
echo -e "${BLUE}2️⃣ Checking Dependencies...${NC}"
if python3 -c "import flask, torch, transformers" 2>/dev/null; then
    echo -e "✅ Core dependencies installed"
else
    echo -e "${YELLOW}⚠️  Installing dependencies...${NC}"
    pip3 install -r requirements.txt
fi

# Check gunicorn
if [ -f "/Users/ws/Library/Python/3.9/bin/gunicorn" ]; then
    echo -e "✅ Gunicorn installed"
    GUNICORN_PATH="/Users/ws/Library/Python/3.9/bin/gunicorn"
elif command -v gunicorn &> /dev/null; then
    echo -e "✅ Gunicorn installed"
    GUNICORN_PATH="gunicorn"
else
    echo -e "${YELLOW}⚠️  Installing Gunicorn...${NC}"
    pip3 install gunicorn==21.2.0
    GUNICORN_PATH="/Users/ws/Library/Python/3.9/bin/gunicorn"
fi

# 3. Check trained model
echo -e "${BLUE}3️⃣ Checking Trained Model...${NC}"
if ls outputs/training_*/final_model/config.json 1> /dev/null 2>&1; then
    echo -e "✅ Found trained model"
    ls outputs/training_*/final_model/ | head -3
else
    echo -e "${RED}❌ No trained model found${NC}"
    echo -e "${YELLOW}Please run: python3 train.py${NC}"
    exit 1
fi

# 4. Stop existing service
echo -e "${BLUE}4️⃣ Checking Existing Service...${NC}"
if [ -f "/tmp/gunicorn_safe_text.pid" ]; then
    echo -e "${YELLOW}⚠️  Found existing service, stopping...${NC}"
    kill -TERM `cat /tmp/gunicorn_safe_text.pid` 2>/dev/null
    sleep 3
    rm -f /tmp/gunicorn_safe_text.pid
fi

# Check port usage
if lsof -i :9900 >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 9900 is occupied, releasing...${NC}"
    lsof -ti :9900 | xargs kill -9 2>/dev/null
    sleep 2
fi

# 5. Create log directory
mkdir -p logs

# 6. Start service
echo -e "${BLUE}5️⃣ Starting Multi-Process API Service...${NC}"
echo -e "🔧 Starting service with 4 worker processes..."

# Start gunicorn in background
nohup $GUNICORN_PATH -c config/gunicorn_config.py api_server:app > logs/startup.log 2>&1 &

# Wait for service to start
echo -e "${YELLOW}⏳ Waiting for service startup (max 30 seconds)...${NC}"
for i in {1..30}; do
    if curl -s -H "Authorization: Bearer safe-text-api-2025-9d8f7e6c5b4a3210" http://localhost:9900/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Service started successfully!${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# 7. Service verification
echo -e "\n${BLUE}6️⃣ Service Verification...${NC}"

# Health check
HEALTH_RESPONSE=$(curl -s -H "Authorization: Bearer safe-text-api-2025-9d8f7e6c5b4a3210" http://localhost:9900/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "✅ Health Check: Passed"
    echo -e "   Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}❌ Health Check: Failed${NC}"
    echo -e "   Response: $HEALTH_RESPONSE"
    exit 1
fi

# Test prediction API
echo -e "${BLUE}7️⃣ Testing Prediction API...${NC}"
PREDICT_RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer safe-text-api-2025-9d8f7e6c5b4a3210" \
    -H "Content-Type: application/json" \
    -d '{"text":"This is a test text"}' \
    http://localhost:9900/predict)

if echo "$PREDICT_RESPONSE" | grep -q "confidence"; then
    echo -e "✅ Prediction API: Normal"
    echo -e "   Test Result: $(echo $PREDICT_RESPONSE | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"Label:{data.get('label','N/A')}, Confidence:{data.get('confidence','N/A')}\")")"
else
    echo -e "${YELLOW}⚠️  Prediction API: May have issues${NC}"
    echo -e "   Response: $PREDICT_RESPONSE"
fi

# 8. Display process information
echo -e "${BLUE}8️⃣ Process Information...${NC}"
PIDS=$(pgrep -f "gunicorn.*api_server")
if [ ! -z "$PIDS" ]; then
    echo -e "✅ Found $(echo $PIDS | wc -w) Gunicorn processes:"
    ps aux | grep gunicorn | grep -v grep | while read line; do
        echo -e "   $line"
    done
else
    echo -e "${RED}❌ No Gunicorn processes found${NC}"
fi

# 9. Startup complete
echo -e "\n${GREEN}🎉 Startup Complete!${NC}"
echo -e "=========================================="
echo -e "📡 API Service URL: ${BLUE}http://localhost:9900${NC}"
echo -e "🔑 API Token: ${YELLOW}safe-text-api-2025-9d8f7e6c5b4a3210${NC}"
echo -e "📊 Worker Processes: ${GREEN}4${NC}"
echo -e "🔄 Max Concurrent: ${GREEN}32 connections${NC}"

echo -e "\n${BLUE}📋 Available APIs:${NC}"
echo -e "  GET  /health          - Health check"
echo -e "  GET  /model/status    - Model status"
echo -e "  POST /predict         - Single text prediction"
echo -e "  POST /predict/batch   - Batch text prediction"

echo -e "\n${BLUE}🔧 Management Commands:${NC}"
echo -e "  View logs: tail -f logs/gunicorn_error.log"
echo -e "  Stop service: kill -TERM \`cat /tmp/gunicorn_safe_text.pid\`"
echo -e "  Restart service: ./start_server.sh"

echo -e "\n${BLUE}🧪 Test Commands:${NC}"
echo -e "  Health check: curl -H \"Authorization: Bearer safe-text-api-2025-9d8f7e6c5b4a3210\" http://localhost:9900/health"
echo -e "  Performance test: python3 test_performance.py"

echo -e "\n${GREEN}🚀 Service is ready for use!${NC}"
