from __future__ import annotations

import sys
from pathlib import Path

from ..models import ScriptMetadata, ScriptParameter, ScriptRunResponse
from ..utils.module_loader import load_module
from .base import ScriptDefinition

_module = load_module(
    "zlkt_sop_reference_matcher",
    "backend/resources/scripts/zlkt_sop_reference_matcher.py",
)
build_argument_parser = getattr(_module, "build_argument_parser", None)
main_impl = getattr(_module, "main", None)
MODULE_DEFAULT_SOP_PATH = getattr(_module, "DEFAULT_SOP_JSON_PATH", None)

if build_argument_parser is None or main_impl is None:
    raise AttributeError("zlkt_sop_reference_matcher.py 缺少 build_argument_parser 或 main 函数")


def _resolve_excel_path(params: dict) -> Path:
    explicit = params.get("excel_file")
    if explicit:
        candidate = Path(explicit).expanduser()
    else:
        corpus = params.get("corpus_path")
        if not corpus:
            raise FileNotFoundError("请提供输入Excel，或先上传文件后选择为输入。")
        candidate = Path(corpus).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {candidate}")
    return candidate


def _resolve_sop_json(params: dict) -> Path:
    sop_json = params.get("sop_json")
    if sop_json:
        candidate = Path(sop_json).expanduser()
    elif MODULE_DEFAULT_SOP_PATH is not None:
        candidate = Path(MODULE_DEFAULT_SOP_PATH)
    else:
        candidate = Path(__file__).resolve().parents[2] / "resources/data/logic_tree_zlkt_origin2.json"
    if not candidate.exists():
        raise FileNotFoundError(f"SOP JSON 不存在: {candidate}")
    return candidate


def _run(params: dict) -> ScriptRunResponse:
    excel_path = _resolve_excel_path(params)
    sop_json = _resolve_sop_json(params)

    mode = params.get("mode") or "both"
    if mode not in {"verify", "infer", "both"}:
        raise ValueError("mode 需为 verify / infer / both 之一")

    output_param = params.get("output_file")
    output_path = Path(output_param).expanduser() if output_param else None
    history_col = params.get("history_column") or "历史对话"
    level1_col = params.get("level1_column") or "SOP一级节点"
    level2_col = params.get("level2_column") or "SOP二级节点"
    similarity = float(params.get("similarity_threshold") or 0.75)
    sales_keywords = params.get("sales_keywords") or "销售,CSM"

    parser = build_argument_parser()
    cli_args = [
        "--mode",
        mode,
        "--excel",
        str(excel_path),
        "--sop-json",
        str(sop_json),
        "--history-column",
        history_col,
        "--level1-column",
        level1_col,
        "--level2-column",
        level2_col,
        "--similarity-threshold",
        str(similarity),
        "--sales-keywords",
        sales_keywords,
    ]
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cli_args.extend(["--output", str(output_path)])
    else:
        output_path = excel_path.with_name(f"{excel_path.stem}_{mode}.xlsx")

    parser.parse_args(cli_args)
    prev_argv = sys.argv
    sys.argv = ["zlkt_sop_reference_matcher.py"] + cli_args
    try:
        exit_code = main_impl()
    finally:
        sys.argv = prev_argv
    if exit_code not in (0, None):
        raise RuntimeError("SOP参考匹配脚本执行失败")

    data = {
        "input_excel": str(excel_path),
        "sop_json": str(sop_json),
        "output_file": str(output_path),
        "mode": mode,
    }
    message = "SOP参考匹配完成"
    return ScriptRunResponse(success=True, message=message, data=data)


SCRIPT_DEFINITION = ScriptDefinition(
    metadata=ScriptMetadata(
        id="zlkt_sop_reference_matcher",
        name="之了课堂SOP参考匹配",
        description="对 `zlkt_combined_processor_v2` 的输出进行 SOP 节点校验/自动定位，生成命中情况与参考话术。",
        category="SOP分析",
        parameters=[
            ScriptParameter(
                name="excel_file",
                label="输入Excel",
                type="path",
                description="来自 `之了课堂回流预处理` 的输出文件，包含历史对话与 SOP 标签。",
            ),
            ScriptParameter(
                name="sop_json",
                label="SOP逻辑树",
                type="path",
                description="包含参考话术的 SOP JSON（如 logic_tree_zlkt_origin2.json）。",
            ),
            ScriptParameter(
                name="mode",
                label="模式",
                type="select",
                required=False,
                options=["verify", "infer", "both"],
                description="选择校验、自动定位或同时输出两种结果，默认 both。",
            ),
            ScriptParameter(
                name="output_file",
                label="输出文件",
                type="path",
                required=False,
                description="默认在输入文件旁生成 *_mode.xlsx，可自定义路径。",
            ),
        ],
        output_description="生成包含校验/定位字段的 Excel，输出路径可自定义。",
    ),
    runner=_run,
)
