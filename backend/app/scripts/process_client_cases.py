from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from ..models import ScriptMetadata, ScriptParameter, ScriptRunResponse
from ..utils.module_loader import load_module
from .base import ScriptDefinition

_module = load_module(
    "pre_process_improved_send_by_client_thought_unit_fixed_v4",
    "backend/resources/scripts/pre_process_improved_send_by_client_thought_unit_fixed_v4.py",
)

_read_excel = getattr(_module, "read_and_process_excel")
_add_csm = getattr(_module, "add_csm_replies_to_customer_messages")
_propagate = getattr(_module, "propagate_thought_unit_to_customers")
_extract = getattr(_module, "extract_thought_unit_fields")
_process = getattr(_module, "process_data_by_sales_id")
_save_results = getattr(_module, "save_test_dataset_results")


def _resolve_output_path(input_path: Path, custom_path: str | None) -> Path:
    if custom_path:
        candidate = Path(custom_path).expanduser()
        if candidate.is_dir() or not candidate.suffix:
            candidate = candidate / f"{input_path.stem}_client_processed.xlsx"
    else:
        candidate = input_path.with_name(f"{input_path.stem}_client_processed{input_path.suffix}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _run(params: dict) -> ScriptRunResponse:
    input_file = Path(params.get("input_file", "")).expanduser()
    if not input_file.exists():
        raise FileNotFoundError(f"输入Excel不存在: {input_file}")

    output_file = _resolve_output_path(input_file, params.get("output_file"))

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        df = _read_excel(str(input_file))
        df = _add_csm(df)
        df = _propagate(df)
        df = _extract(df)
        test_df, sales_summary, cycle_label_stats = _process(df)
        _save_results(test_df, sales_summary, cycle_label_stats, str(output_file))

    data = {
        "input_file": str(input_file),
        "output_file": str(output_file),
    }
    return ScriptRunResponse(
        success=True,
        message="按客户视角处理完成",
        data=data,
        logs=buffer.getvalue(),
    )


SCRIPT_DEFINITION = ScriptDefinition(
    metadata=ScriptMetadata(
        id="preprocess_client_cases",
        name="橙啦客户回流预处理",
        description="对橙啦回流案例（发送方为客户）进行清洗，生成包含销售汇总/周期标签的测试集。",
        category="数据治理",
        parameters=[
            ScriptParameter(
                name="input_file",
                label="回流案例Excel",
                type="path",
                description="请选择需要处理的橙啦回流案例（发送方为客户）的 Excel。",
            ),
            ScriptParameter(
                name="output_file",
                label="输出文件路径",
                type="path",
                required=False,
                description="留空则在原目录生成 *_client_processed.xlsx。",
            ),
        ],
        output_description="返回清洗后 Excel 的保存路径，并可查看日志。",
    ),
    runner=_run,
)
