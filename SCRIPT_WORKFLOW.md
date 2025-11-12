# Script Studio 脚本执行流程分析

## 🔄 核心执行流程

```mermaid
graph TD
    A[用户选择脚本] --> B[前端参数映射]
    B --> C[API请求 /api/scripts/{id}/run]
    C --> D[FastAPI接收请求]
    D --> E[ScriptRegistry获取脚本]
    E --> F[参数验证与预处理]
    F --> G[线程池执行脚本]
    G --> H[脚本执行完成]
    H --> I[返回执行结果]
    I --> J[前端显示结果]

    K[文件上传] --> L[保存到uploads目录]
    L --> M[前端记录文件信息]
    M --> B
```

## 📋 数据流向

### 1. 前端参数映射 (StudioLayout.vue)
```javascript
// 原始表单参数
const params = { ...formValues.value }

// 脚本参数映射
if (selectedScript.value.parameters.some(p => p.name === 'excel_path')) {
  params.excel_path = params.corpus_path
  delete params.corpus_path
}

if (selectedScript.value.parameters.some(p => p.name === 'input_file')) {
  params.input_file = params.corpus_path
  delete params.corpus_path
}
```

### 2. API调用 (api.ts)
```typescript
export function runScript(scriptId: string, params: Record<string, unknown>) {
  return unwrap(
    client.post<ScriptRunResponse>(`/scripts/${scriptId}/run`, {
      params,
    }),
  );
}
```

### 3. 后端接收 (main.py)
```python
@app.post("/api/scripts/{script_id}/run", response_model=ScriptRunResponse)
async def run_script(script_id: str, payload: ScriptRunRequest):
    script = get_script(script_id)

    # 调试日志
    print(f"执行脚本: {script_id}")
    print(f"传递的参数: {payload.params}")

    result = await script.run(payload.params)
    return result
```

### 4. 脚本执行 (base.py)
```python
async def run(self, params: Dict[str, Any]) -> ScriptRunResponse:
    return await run_in_threadpool(self.runner, params)
```

## 🛠 脚本注册系统

### 注册表 (script_registry.py)
```python
_SCRIPT_DEFINITIONS: Dict[str, ScriptDefinition] = {
    definition.metadata.id: definition
    for definition in [
        UPDATE_LATEST_CUSTOMER,
        SOP_PIPELINE,
        WAXU_BADCASE
    ]
}

def get_script(script_id: str) -> ScriptDefinition | None:
    return _SCRIPT_DEFINITIONS.get(script_id)
```

### 脚本定义示例 (get_sop_pipeline.py)
```python
SCRIPT_DEFINITION = ScriptDefinition(
    metadata=ScriptMetadata(
        id="run_sop_pipeline",
        name="SOP流程标注",
        description="上传Excel文件，生成SOP标注结果",
        category="SOP分析",
        parameters=[...]  # 参数定义
    ),
    runner=_run,  # 执行函数
)
```

## 📁 可用脚本

### 1. 同步最新客户消息 (`update_latest_customer_message`)
- **ID**: `update_latest_customer_message`
- **参数**: `excel_path`, `sheet_name`, `context_column`, `latest_customer_column`, `output_path`
- **功能**: 从历史对话中提取最新客户消息

### 2. SOP流程标注 (`run_sop_pipeline`)
- **ID**: `run_sop_pipeline`
- **参数**: `corpus_path`, `output_dir`, `output_filename`, `logic_tree_path`, `similarity`, `batch_size`
- **功能**: 基于逻辑树对对话进行SOP标签标注

### 3. 挖需BadCase清洗 (`process_waxu_badcase`)
- **ID**: `process_waxu_badcase`
- **参数**: `input_file`, `output_file`
- **功能**: 处理挖需回流的BadCase数据

## 🔧 核心组件关系

### 数据模型层
```
ScriptParameter → ScriptMetadata → ScriptDefinition
     ↓              ↓                    ↓
  参数定义       脚本元信息           执行配置
```

### API层
```
FastAPI → ScriptRegistry → ScriptDefinition → ScriptRunner
  ↓            ↓                ↓               ↓
 路由处理    脚本查找        元数据管理     异步执行
```

### 前端层
```
StudioLayout → API调用 → 脚本Sidebar → 结果Panel
     ↓          ↓           ↓           ↓
  参数映射    HTTP请求     脚本选择    执行反馈
```