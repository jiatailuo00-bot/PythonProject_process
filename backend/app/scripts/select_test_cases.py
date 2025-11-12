from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from ..models import ScriptMetadata, ScriptParameter, ScriptRunResponse
from ..utils.module_loader import load_module
from .base import ScriptDefinition

_module = load_module(
    "select_case3",
    "backend/resources/scripts/select_case3.py",
)
SelectorCls = getattr(_module, "RealisticSalesTestSelector", None)
if SelectorCls is None:
    raise AttributeError("select_case3.py 中缺少 RealisticSalesTestSelector 类")


def _resolve_output_path(input_path: Path, custom_path: str | None) -> Path:
    if custom_path:
        target = Path(custom_path).expanduser()
        if target.is_dir():
            target = target / f"{input_path.stem}_selected_cases.xlsx"
        elif not target.suffix:
            target = target.with_suffix(".xlsx")
    else:
        target = input_path.with_name(f"{input_path.stem}_selected_cases{input_path.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _resolve_report_path(output_path: Path, report_path: str | None) -> Path:
    if report_path:
        target = Path(report_path).expanduser()
    else:
        target = output_path.with_name(f"{output_path.stem}_report.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _run(params: dict) -> ScriptRunResponse:
    input_file = Path(params.get("input_file", "")).expanduser()
    if not input_file.exists():
        raise FileNotFoundError(f"输入Excel不存在: {input_file}")

    output_file = _resolve_output_path(input_file, params.get("output_file"))
    report_file = _resolve_report_path(output_file, params.get("report_file"))

    selector = SelectorCls()

    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        selector.load_data(str(input_file))
        test_cases = selector.select_realistic_test_cases()
        selector.export_realistic_test_cases(test_cases, str(output_file))
        report = selector.generate_realistic_test_report(test_cases)

    report_file.write_text(report, encoding="utf-8")

    data = {
        "input_file": str(input_file),
        "selected_cases_file": str(output_file),
        "report_file": str(report_file),
        "case_count": len(test_cases),
    }
    message = f"已筛选 {len(test_cases)} 条测试用例"
    return ScriptRunResponse(
        success=True,
        message=message,
        data=data,
        logs=log_buffer.getvalue(),
    )


SCRIPT_DEFINITION = ScriptDefinition(
    metadata=ScriptMetadata(
        id="select_client_cases",
        name="真实场景案例筛选",
        description="基于预处理后的橙啦客户回流案例，按周期标签/销售分布挑选真实测试用例并生成报告。",
        category="数据治理",
        parameters=[
            ScriptParameter(
                name="input_file",
                label="预处理结果Excel",
                type="path",
                description="请选择由“橙啦客户回流预处理”脚本生成的测试集 Excel（包含《测试集_标准格式》工作表）。",
            ),
            ScriptParameter(
                name="output_file",
                label="测试用例输出路径",
                type="path",
                required=False,
                description="默认为输入同目录生成 *_selected_cases.xlsx，可自定义保存位置。",
            ),
            ScriptParameter(
                name="report_file",
                label="报告输出路径",
                type="path",
                required=False,
                description="默认为测试用例同目录生成 Markdown 报告，可自定义路径。",
            ),
        ],
        output_description="返回筛选后的测试用例 Excel 与报告路径。",
    ),
    runner=_run,
)
