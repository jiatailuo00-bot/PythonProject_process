# Vue + Element Plus 控制台

此前端基于 Vite + Vue 3 + Element Plus，实现脚本编排、参数表单、结果展示与文件管理。

## 可用命令

```bash
npm install --registry=https://registry.npmjs.org/
npm run dev    # 启动本地开发（http://localhost:5173）
npm run build  # 生产构建（输出 dist/）
npm run preview
```

## 主要特性
- 左侧脚本列表 + 搜索
- 中间参数面板（必填/可选折叠）
- 文件助手：拖拽/点击/粘贴上传，历史文件一键填充
- 右侧结果与执行历史

## 环境变量
- `VITE_API_BASE_URL`：后端 API 根地址（默认 `/api`）

## 技术栈
- [Vue 3](https://vuejs.org/) + `<script setup>`
- [Element Plus](https://element-plus.org/)
- [Vite](https://vitejs.dev/)
