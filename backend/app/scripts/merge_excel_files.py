from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..models import ScriptMetadata, ScriptParameter, ScriptRunResponse
from ..utils.module_loader import load_module
from .base import ScriptDefinition

_module = load_module("merge_excel_files", "backend/resources/scripts/merge_excel_files.py")
ensure_directory = getattr(_module, "ensure_directory")
collect_excel_files = getattr(_module, "collect_excel_files")
parse_sheet_arg = getattr(_module, "parse_sheet_arg")
load_frames = getattr(_module, "load_frames")


def _resolve_paths(params: dict) -> tuple[Path, Path]:
    directory_param = params.get("directory") or ""
    directory = ensure_directory(Path(directory_param).expanduser())

    output_param = params.get("output_file")
    if output_param:
        output_path = Path(output_param).expanduser()
        if output_path.is_dir():
            output_path = output_path / "merged.xlsx"
        elif not output_path.suffix:
            output_path = output_path.with_suffix(".xlsx")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = directory / f"merged_{timestamp}.xlsx"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return directory, output_path


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _run(params: dict) -> ScriptRunResponse:
    directory, output_file = _resolve_paths(params)
    pattern = params.get("pattern") or "*.xlsx"
    include_source = _to_bool(params.get("include_source_column", False))
    sheet_param = params.get("sheet_name")

    files = collect_excel_files(directory, pattern)
    if not files:
        raise FileNotFoundError(f"未在 {directory} 中找到匹配 '{pattern}' 的 Excel 文件")

    sheet = parse_sheet_arg(sheet_param)
    frames = load_frames(files, sheet, include_source)
    merged = pd.concat(frames, ignore_index=True)
    merged.to_excel(output_file, index=False)

    data = {
        "directory": str(directory),
        "pattern": pattern,
        "file_count": len(files),
        "output_file": str(output_file),
    }
    message = f"已合并 {len(files)} 个文件"
    return ScriptRunResponse(success=True, message=message, data=data)


SCRIPT_DEFINITION = ScriptDefinition(
    metadata=ScriptMetadata(
        id="merge_excel_files",
        name="Excel 批量合并",
        description="扫描目录下的多个 Excel，按顺序合并为一份工作簿，可选添加来源列。",
        category="通用工具",
        parameters=[
            ScriptParameter(
                name="directory",
                label="Excel目录",
                type="path",
                description="需要合并的 Excel 所在目录。",
            ),
            ScriptParameter(
                name="pattern",
                label="文件匹配模式",
                type="string",
                required=False,
                description="glob 模式，默认 '*.xlsx'。",
                example="*.xlsx",
            ),
            ScriptParameter(
                name="sheet_name",
                label="Sheet 名称/序号",
                type="string",
                required=False,
                description="可填写 sheet 名称或索引（0 开始），默认读取首个 sheet。",
            ),
            ScriptParameter(
                name="include_source_column",
                label="追加来源列",
                type="boolean",
                required=False,
                description="是否在结果中增加 `source_file` 列指出原文件。",
            ),
            ScriptParameter(
                name="output_file",
                label="输出文件",
                type="path",
                required=False,
                description="默认在目录下生成 merged_时间戳.xlsx，可自定义保存路径。",
            ),
        ],
        output_description="返回合并后的 Excel 路径及统计信息。",
    ),
    runner=_run,
)
