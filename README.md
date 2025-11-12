# Script Control Center

一套统一托管脚本的全栈方案：后端使用 FastAPI 暴露本地 Python 工具，前端基于 Vue 3 + Element Plus 提供更友好的可视化控制台，支持一键运行脚本并查看日志、输出。

## 快速开始

### 1. 启动后端
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```
- API 文档：<http://localhost:8000/api/docs>
- 已接入脚本
- 1. `同步最新客户消息`
- 2. `SOP 流程标注`
- 3. `预期话术拆分`
- 4. `橙啦客户回流预处理`
- 5. `真实场景案例筛选`
- 6. `CSM 强遵循识别`
- 7. `SOP 节点 ID 映射`
- 8. `之了课堂回流预处理`
- 9. `Excel 批量合并`
- 10. `之了课堂 SOP 参考匹配`
- 11. `挖需 BadCase 清洗`

### 脚本输入要点

| 功能 | 说明 | 关键输入列/要求 |
| --- | --- | --- |
| 同步最新客户消息 | 从“最终传参上下文”抽取最新 `[客户]` 话术覆盖“最新客户消息”。 | Excel 需含 `最终传参上下文`、`最新客户消息`（可自定义列名）。 |
| SOP 流程标注 | 纯改进版 SOP 算法，输出 `_pure_improved.xlsx`。 | 输入 Excel 需含 `最终传参上下文`，可选 `输出文件名`；逻辑树可选。 |
| 预期话术拆分 | 针对纯改进版结果拆分“预期话术”。 | 输入 Excel 需含 `预期话术` 列（来自 SOP 流程标注）。 |
| 橙啦客户回流预处理 | 将发送方为客户的回流案例清洗成测试集并生成统计。 | 输入需含 `历史对话`、`最新客户消息`、`rag`、`thought_unit` 等原始字段（脚本默认读取预置列）。 |
| 真实场景案例筛选 | 依据周期/销售分布挑选测试用例并产出报告。 | 输入必须是“橙啦客户回流预处理”生成的 Excel，包含 `测试集_标准格式` 工作表（含 `最新客户消息`、`周期标签`、`发送时间` 等列）。 |
| CSM 强遵循识别 | 使用 `force_patterns_config.json` 对 CSM 回复打强遵循标签。 | 输入需含 CSM 文本列（默认 `发送消息内容`），可自定义列名；配置可覆盖 `force_patterns_config.json`。 |
| SOP 节点 ID 映射 | 依据 SOP 定义表为案例补齐 `SOP一级/二级节点ID`。 | 案例表需含 `SOP一级节点`、`SOP二级节点`；SOP 定义表需含 `任务主题（一级节点）`、`子任务主题（二级节点）`、`ID`、`子任务ID`（默认使用 `backend/resources/data/珍酒sop1.xlsx`，可自定义）。 |
| Excel 批量合并 | 将一个目录下的多份 Excel 纵向合并，可指定 sheet、追加来源列。 | 提供包含待合并文件的目录；各文件应有一致的列结构。 |
| 之了课堂回流预处理 | 对之了课堂租户线上回流案例做预处理，输出 SOP 补全、客户测试集与重点抽样三份结果。 | 输入为原始回流案例 Excel（脚本会自动在同目录生成 `_preprocessed_sop.xlsx`、`_customer_dataset.xlsx`、`_customer_dataset_sampled.xlsx`）。 |
| 之了课堂 SOP 参考匹配 | 对回流预处理输出的文件执行 SOP 节点校验/自动定位，生成命中情况与参考话术。 | 输入需为 `之了课堂回流预处理` 产出的 Excel（含历史对话与 SOP 标签）；另需提供 SOP JSON（如 `logic_tree_zlkt_origin2.json`）。 |
| 挖需 BadCase 清洗 | 封装挖需 BadCase 处理脚本，输出整理后的测试集。 | 输入需包含历史对话与 `rag` 字段，遵循原脚本要求。 |

### 2. 启动前端
```bash
cd frontend
npm install --registry=https://registry.npmjs.org/
npm run dev
```
- 默认开发地址：<http://localhost:5173>
- Vite 已配置 `/api` 代理，可直接请求本地 8000 端口。若后端地址变化，可在 `frontend/.env` 中设置：
  ```bash
  VITE_API_BASE_URL=http://127.0.0.1:8000/api
  ```

### 3. 生产构建
```bash
# 前端构建产物输出到 frontend/dist
cd frontend
npm run build
```

## 目录结构
```
backend/
  app/
    main.py                 # FastAPI 入口
    models.py               # Pydantic 模型
    script_registry.py      # 已注册脚本
    scripts/                # 各脚本适配器
      update_latest_customer_message.py
      get_sop_pipeline.py
      process_waxu_badcase.py
    utils/module_loader.py  # 动态加载 backend/resources/scripts 下的原脚本
  resources/
    scripts/               # 迁移自 .claude 的原始脚本
    data/                  # 逻辑树、客户模式等 JSON
frontend/
  src/
    App.vue                # 主界面（Vue 3 + Element Plus）
    api.ts                 # 调用后端 API
    components/            # 侧边栏、参数表单、文件助手等
    types.ts
```

## 扩展：新增脚本
1. 在 `backend/app/scripts/` 下新建一个适配文件，封装输入参数、执行逻辑并返回 `ScriptRunResponse`。
2. 在 `backend/app/script_registry.py` 注册脚本，使其出现在 `/api/scripts` 列表中。
3. 前端会自动读取脚本元数据并生成表单，无需额外改动。若需要自定义 UI，可在 `src/components` 中扩展。

## 常见命令
| 功能 | 命令 |
| --- | --- |
| 启动后端 | `uvicorn backend.app.main:app --reload` |
| 启动前端 | `cd frontend && npm run dev` |
| 前端构建 | `cd frontend && npm run build` |

## 后续方向
- 接入更多脚本：按照“脚本适配器+注册”的模式扩展。
- 增加任务队列（如 Celery / RQ）以支持长耗时异步任务。
- 前端加入运行进度、参数模板、权限控制等高级能力。
