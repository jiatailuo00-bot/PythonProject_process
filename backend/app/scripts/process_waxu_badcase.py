from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict

from ..models import ScriptMetadata, ScriptParameter, ScriptRunResponse
from ..utils.module_loader import load_module
from .base import ScriptDefinition

_waxu_module = load_module("claude_waxu_badcase", "backend/resources/scripts/process_waxu_badcase-send_by_CSM_v2.py")
_processor_cls = getattr(_waxu_module, "WaxuBadcaseProcessor", None)
if _processor_cls is None:
    raise AttributeError("process_waxu_badcase-send_by_CSM_v2.py 中缺少 WaxuBadcaseProcessor 类")


def _run(params: Dict) -> ScriptRunResponse:
    input_path = Path(params.get("input_file", "")).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_path = Path(params.get("output_file") or input_path.with_name(f"{input_path.stem}_processed.xlsx"))
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor = _processor_cls()

    log_buffer = io.StringIO()
    success = False
    with redirect_stdout(log_buffer):
        if not processor.read_and_process_excel(str(input_path)):
            raise RuntimeError("读取文件失败")
        if not processor.process_data_by_sales_id():
            raise RuntimeError("数据处理失败")
        success = processor.save_results(output_file=str(output_path))

    if not success:
        raise RuntimeError("保存结果失败")

    logs = log_buffer.getvalue()
    message = "挖需BadCase处理完成"
    data = {"input_file": str(input_path), "output_file": str(output_path)}
    return ScriptRunResponse(success=True, message=message, data=data, logs=logs)


SCRIPT_DEFINITION = ScriptDefinition(
    metadata=ScriptMetadata(
        id="process_waxu_badcase",
        name="挖需BadCase清洗",
        description="调用挖需回流BadCase脚本，生成测试集与标签分析文件",
        category="数据治理",
        parameters=[
            ScriptParameter(
                name="input_file",
                label="输入Excel路径",
                type="path",
                description="包含历史对话和rag字段的源文件",
            ),
            ScriptParameter(
                name="output_file",
                label="输出Excel路径",
                type="path",
                required=False,
                description="可自定义输出文件名，默认在原目录下生成 *_processed.xlsx",
            ),
        ],
        output_description="返回生成的结果文件路径",
    ),
    runner=_run,
)
