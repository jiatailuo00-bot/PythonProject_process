#!/bin/bash

echo "🚀 启动 Script Studio 服务..."

# 检查是否在项目根目录
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 启动后端服务
echo "📡 启动后端服务..."
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 等待后端启动
sleep 3

# 启动前端服务
echo "🌐 启动前端服务..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ 服务启动完成!"
echo "📱 前端界面: http://localhost:5173"
echo "📡 后端API: http://localhost:8000"
echo "📚 API文档: http://localhost:8000/api/docs"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待用户中断
trap "echo '🛑 停止服务...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait