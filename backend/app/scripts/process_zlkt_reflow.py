from __future__ import annotations

from pathlib import Path

from ..models import ScriptMetadata, ScriptParameter, ScriptRunResponse
from ..utils.module_loader import load_module
from .base import ScriptDefinition

_module = load_module(
    "zlkt_combined_processor_v2",
    "backend/resources/scripts/zlkt_combined_processor_v2.py",
)
run_pipeline = getattr(_module, "run_pipeline", None)
if run_pipeline is None:
    raise AttributeError("zlkt_combined_processor_v2.py 缺少 run_pipeline 函数")


def _resolve_input(params: dict) -> Path:
    input_file = Path(params.get("input_file", "")).expanduser()
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    return input_file


def _run(params: dict) -> ScriptRunResponse:
    input_file = _resolve_input(params)
    run_pipeline(input_file)

    base = input_file.with_suffix("")
    outputs = {
        "sop_file": str(base.with_name(f"{input_file.stem}_preprocessed_sop.xlsx")),
        "customer_dataset": str(base.with_name(f"{input_file.stem}_customer_dataset.xlsx")),
        "customer_sampled": str(base.with_name(f"{input_file.stem}_customer_dataset_sampled.xlsx")),
    }

    message = "之了课堂回流案例预处理完成"
    return ScriptRunResponse(success=True, message=message, data={"input_file": str(input_file), **outputs})


SCRIPT_DEFINITION = ScriptDefinition(
    metadata=ScriptMetadata(
        id="process_zlkt_reflow",
        name="之了课堂回流预处理",
        description="上传线上回流原始案例，自动生成 SOP 补全、客户测试集及重点抽样三份结果。",
        category="数据治理",
        parameters=[
            ScriptParameter(
                name="input_file",
                label="回流案例Excel",
                type="path",
                description="包含原始之了课堂回流案例的 Excel 文件。",
            ),
        ],
        output_description="返回三份 Excel：*_preprocessed_sop.xlsx、*_customer_dataset.xlsx、*_customer_dataset_sampled.xlsx。",
    ),
    runner=_run,
)
