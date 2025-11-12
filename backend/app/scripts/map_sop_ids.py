from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..models import ScriptMetadata, ScriptParameter, ScriptRunResponse
from ..utils.module_loader import ROOT_DIR, load_module
from .base import ScriptDefinition

_module = load_module("map_sop_ids", "backend/resources/scripts/map_sop_ids.py")
build_lookup = getattr(_module, "build_lookup")
build_level1_index = getattr(_module, "build_level1_index")
map_case_ids = getattr(_module, "map_case_ids")

DEFAULT_MASTER = (ROOT_DIR / "backend/resources/data/珍酒sop1.xlsx").resolve()

CASE_L1 = getattr(_module, "CASE_LEVEL1_COL")
CASE_L2 = getattr(_module, "CASE_LEVEL2_COL")
MASTER_L1 = getattr(_module, "MASTER_LEVEL1_COL")
MASTER_L2 = getattr(_module, "MASTER_LEVEL2_COL")
MASTER_ID = getattr(_module, "MASTER_ID_COL")
MASTER_SUB_ID = getattr(_module, "MASTER_SUBTASK_ID_COL")


def _resolve_paths(params: dict) -> tuple[Path, Path, Path]:
    cases_path = Path(params.get("cases_file", "")).expanduser()
    if not cases_path.exists():
        raise FileNotFoundError(f"案例Excel不存在: {cases_path}")

    master_param = params.get("sop_master_file")
    master_path = Path(master_param).expanduser() if master_param else DEFAULT_MASTER
    if not master_path.exists():
        raise FileNotFoundError(f"SOP定义Excel不存在: {master_path}")

    output_param = params.get("output_file")
    if output_param:
        output_path = Path(output_param).expanduser()
        if output_path.is_dir() or not output_path.suffix:
            output_path = output_path / cases_path.name
    else:
        output_path = cases_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cases_path, master_path, output_path


def _run(params: dict) -> ScriptRunResponse:
    cases_path, master_path, output_path = _resolve_paths(params)

    cases_df = pd.read_excel(cases_path)
    master_df = pd.read_excel(master_path)

    for required in (CASE_L1, CASE_L2):
        if required not in cases_df.columns:
            raise KeyError(f"案例文件缺少列: {required}")
    for required in (MASTER_L1, MASTER_L2, MASTER_ID, MASTER_SUB_ID):
        if required not in master_df.columns:
            raise KeyError(f"SOP定义文件缺少列: {required}")

    lookup = build_lookup(master_df)
    index = build_level1_index(lookup)
    enriched = map_case_ids(cases_df, lookup, index)
    enriched.to_excel(output_path, index=False)

    data = {
        "cases_file": str(cases_path),
        "sop_master_file": str(master_path),
        "output_file": str(output_path),
        "mapped_rows": int(enriched[CASE_L1].notna().sum()),
    }
    message = f"节点ID映射完成: {data['mapped_rows']} 行已附加 ID"
    return ScriptRunResponse(success=True, message=message, data=data)


SCRIPT_DEFINITION = ScriptDefinition(
    metadata=ScriptMetadata(
        id="map_sop_ids",
        name="SOP节点ID映射",
        description=(
            "依据 SOP 定义表为案例文件的 `SOP一级节点/二级节点` 补齐 `SOP一级节点ID/二级节点ID`。"
        ),
        category="SOP分析",
        parameters=[
            ScriptParameter(
                name="cases_file",
                label="案例Excel",
                type="path",
                description="需包含 `SOP一级节点` 与 `SOP二级节点` 列的案例文件。",
            ),
            ScriptParameter(
                name="sop_master_file",
                label="SOP定义Excel",
                type="path",
                description="需包含 `任务主题（一级节点）`、`子任务主题（二级节点）`、`ID`、`子任务ID` 列。",
            ),
            ScriptParameter(
                name="output_file",
                label="输出文件",
                type="path",
                required=False,
                description="默认覆盖案例文件；可自定义新路径。",
            ),
        ],
        output_description="在案例Excel中新增 `SOP一级节点ID` / `SOP二级节点ID` 两列。",
    ),
    runner=_run,
)
