#!/bin/bash

case "$1" in
    start)
        echo "🚀 启动 Script Studio 服务..."
        lsof -ti:8000,5173 | xargs kill -9 2>/dev/null

        cd backend
        nohup python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
        BACKEND_PID=$!

        cd ../frontend
        nohup npm run dev > ../frontend.log 2>&1 &
        FRONTEND_PID=$!

        echo "✅ 服务启动完成!"
        echo "📱 前端: http://localhost:5173"
        echo "📡 后端: http://localhost:8000"
        echo "📚 文档: http://localhost:8000/api/docs"
        echo ""
        echo "后端PID: $BACKEND_PID"
        echo "前端PID: $FRONTEND_PID"
        echo "查看日志: tail -f backend.log frontend.log"
        ;;

    stop)
        echo "🛑 停止 Script Studio 服务..."
        KILLED=$(lsof -ti:8000,5173 | xargs kill -9 2>/dev/null)
        if [ -n "$KILLED" ]; then
            echo "✅ 已停止服务进程: $KILLED"
        else
            echo "ℹ️  没有运行中的服务"
        fi
        ;;

    restart)
        echo "🔄 重启 Script Studio 服务..."
        $0 stop
        sleep 2
        $0 start
        ;;

    status)
        echo "📊 Script Studio 服务状态:"
        echo ""

        BACKEND_PID=$(lsof -ti:8000)
        FRONTEND_PID=$(lsof -ti:5173)

        if [ -n "$BACKEND_PID" ]; then
            echo "✅ 后端服务运行中 (PID: $BACKEND_PID) - http://localhost:8000"
        else
            echo "❌ 后端服务未运行"
        fi

        if [ -n "$FRONTEND_PID" ]; then
            echo "✅ 前端服务运行中 (PID: $FRONTEND_PID) - http://localhost:5173"
        else
            echo "❌ 前端服务未运行"
        fi

        echo ""
        echo "📚 API文档: http://localhost:8000/api/docs"
        ;;

    logs)
        echo "📋 查看服务日志 (Ctrl+C 退出):"
        echo ""
        tail -f backend.log frontend.log
        ;;

    *)
        echo "Script Studio 服务管理工具"
        echo ""
        echo "用法: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "命令说明:"
        echo "  start   - 启动服务 (后台运行)"
        echo "  stop    - 停止服务"
        echo "  restart - 重启服务"
        echo "  status  - 查看服务状态"
        echo "  logs    - 查看服务日志"
        echo ""
        echo "示例:"
        echo "  $0 start    # 启动服务"
        echo "  $0 status   # 查看状态"
        echo "  $0 stop     # 停止服务"
        exit 1
        ;;
esac