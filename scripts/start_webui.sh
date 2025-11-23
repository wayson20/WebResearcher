#!/bin/bash

# WebResearcher Web UI 启动脚本

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

# 检查依赖
echo "检查依赖..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "正在安装依赖..."
    pip install -r webui/requirements.txt
fi

# 启动服务
echo "启动 WebResearcher Web UI..."
echo "访问地址: http://localhost:8000"
echo ""

python3 -m uvicorn webui.app:app --reload --host 0.0.0.0 --port 8000
