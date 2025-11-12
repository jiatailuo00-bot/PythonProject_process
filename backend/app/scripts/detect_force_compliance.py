from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from ..models import ScriptMetadata, ScriptParameter, ScriptRunResponse
from ..utils.module_loader import ROOT_DIR, load_module
from .base import ScriptDefinition

DATA_DIR = ROOT_DIR / "backend/resources/data"
DEFAULT_CONFIG = DATA_DIR / "force_patterns_config.json"

_module = load_module(
    "detect_csm_force_compliance",
    "backend/resources/scripts/detect_csm_force_compliance.py",
)
Detector = getattr(_module, "CSMForceComplianceDetector")
load_dataframe = getattr(_module, "load_dataframe")
guess_default_output = getattr(_module, "guess_default_output")


def _resolve_output_path(input_path: Path, custom_path: str | None) -> Path:
    if custom_path:
        candidate = Path(custom_path).expanduser()
        if candidate.is_dir():
            candidate = candidate / f"{input_path.stem}_force_detected.xlsx"
        elif not candidate.suffix:
            candidate = candidate.with_suffix(".xlsx")
    else:
        candidate = Path(guess_default_output(str(input_path)))
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _resolve_config_path(custom_path: str | None) -> Path:
    candidate = Path(custom_path).expanduser() if custom_path else DEFAULT_CONFIG
    if not candidate.exists():
        raise FileNotFoundError(f"强遵循配置文件不存在: {candidate}")
    return candidate


def _run(params: dict) -> ScriptRunResponse:
    input_file = Path(params.get("input_file", "")).expanduser()
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    text_column = (params.get("text_column") or "发送消息内容").strip() or "发送消息内容"
    output_file = _resolve_output_path(input_file, params.get("output_file"))
    config_path = _resolve_config_path(params.get("config_path"))

    detector = Detector(str(config_path))
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        df = load_dataframe(str(input_file))
        annotated_df = detector.annotate_dataframe(df, text_column)
        annotated_df.to_excel(output_file, index=False, engine="openpyxl")

    total = len(annotated_df)
    hits = int((annotated_df["强遵循识别结果"] == "是").sum())
    logs = buffer.getvalue()

    data = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "config_file": str(config_path),
        "text_column": text_column,
        "rows": total,
        "hits": hits,
    }
    message = f"强遵循识别完成：{hits}/{total} 条命中"
    return ScriptRunResponse(success=True, message=message, data=data, logs=logs)


SCRIPT_DEFINITION = ScriptDefinition(
    metadata=ScriptMetadata(
        id="detect_force_compliance",
        name="CSM强遵循识别",
        description="使用 force_patterns_config.json 中的模式识别 CSM 回复是否命中强遵循话术。",
        category="数据治理",
        parameters=[
            ScriptParameter(
                name="input_file",
                label="输入Excel/CSV",
                type="path",
                description="包含 CSM 回复的文件，支持 xlsx/xls/csv。",
            ),
            ScriptParameter(
                name="text_column",
                label="回复列名",
                type="string",
                required=False,
                description="CSM 文本所在列，默认“发送消息内容”。",
                example="发送消息内容",
            ),
            ScriptParameter(
                name="output_file",
                label="输出文件路径",
                type="path",
                required=False,
                description="可自定义结果保存路径；留空则生成 *_force_detected.xlsx。",
            ),
            ScriptParameter(
                name="config_path",
                label="模式配置文件",
                type="path",
                required=False,
                description=f"默认使用 {DEFAULT_CONFIG.name}，如需临时替换可在此指定。",
            ),
        ],
        output_description="返回附加强遵循标签的 Excel 路径，并统计命中条数。",
    ),
    runner=_run,
)
