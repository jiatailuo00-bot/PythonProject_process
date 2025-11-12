from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from ..models import ScriptMetadata, ScriptParameter, ScriptRunResponse
from ..utils.module_loader import ROOT_DIR, load_module
from .base import ScriptDefinition

DATA_DIR = ROOT_DIR / "backend/resources/data"
DEFAULT_LOGIC_TREE = DATA_DIR / "chengla_wx.json"

_module = load_module(
    "extract_expected_utterance_parts",
    "backend/resources/scripts/extract_expected_utterance_parts.py",
)
_process_file = getattr(_module, "process_file", None)
if _process_file is None:
    raise AttributeError("extract_expected_utterance_parts.py 缺少 process_file 函数")


def _resolve_logic_tree(path: str | None) -> Path:
    if path:
        candidate = Path(path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"逻辑树文件不存在: {candidate}")
        return candidate
    if DEFAULT_LOGIC_TREE.exists():
        return DEFAULT_LOGIC_TREE
    raise FileNotFoundError("缺少默认逻辑树文件 chengla_wx.json")


def _resolve_output_path(input_path: Path, custom_path: str | None) -> Path:
    if custom_path:
        candidate = Path(custom_path).expanduser()
        if candidate.is_dir() or not candidate.suffix:
            candidate = candidate / f"{input_path.stem}_with_expected_parts.xlsx"
    else:
        candidate = input_path.with_name(f"{input_path.stem}_with_expected_parts{input_path.suffix}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _run(params: dict) -> ScriptRunResponse:
    input_file = Path(params.get("input_file", "")).expanduser()
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    output_file = _resolve_output_path(input_file, params.get("output_file"))
    logic_tree = _resolve_logic_tree(params.get("logic_tree_path"))

    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        _process_file(str(input_file), str(output_file), str(logic_tree))

    data = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "logic_tree": str(logic_tree),
    }
    return ScriptRunResponse(
        success=True,
        message="预期话术拆分完成",
        data=data,
        logs=log_buffer.getvalue(),
    )


SCRIPT_DEFINITION = ScriptDefinition(
    metadata=ScriptMetadata(
        id="extract_expected_utterance_parts",
        name="预期话术拆分",
        description="对纯改进版SOP结果的“预期话术”列进行二次解析，拆分出“传递/动作”两列。",
        category="SOP分析",
        parameters=[
            ScriptParameter(
                name="input_file",
                label="SOP结果Excel",
                type="path",
                description="请选择已完成SOP流程标注的Excel（纯改进版输出）。",
            ),
            ScriptParameter(
                name="output_file",
                label="输出文件路径",
                type="path",
                required=False,
                description="可自定义保存位置；留空默认在原目录生成 *_with_expected_parts.xlsx。",
            ),
            ScriptParameter(
                name="logic_tree_path",
                label="自定义逻辑树",
                type="path",
                required=False,
                description="如需替换默认 chengla_wx.json，可填写新的逻辑树路径。",
            ),
        ],
        output_description="返回拆分后的Excel路径，并附带执行日志。",
    ),
    runner=_run,
)
