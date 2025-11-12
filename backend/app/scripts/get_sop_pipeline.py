from __future__ import annotations

import io
import json
import os
import shutil
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..models import ScriptMetadata, ScriptParameter, ScriptRunResponse
from ..utils.module_loader import ROOT_DIR, load_module
from .base import ScriptDefinition

DEFAULT_CONFIG_FILE = ROOT_DIR / "backend/config/sop_defaults.json"
SCRIPTS_DIR = ROOT_DIR / "backend/resources/scripts"
DATA_DIR = ROOT_DIR / "backend/resources/data"
DEFAULT_LOGIC_TREE = DATA_DIR / "chengla_wx.json"
CUSTOMER_PATTERNS_SOURCE = DATA_DIR / "customer_response_patterns.json"
CUSTOMER_PATTERNS_TARGET = SCRIPTS_DIR / "customer_response_patterns.json"

# Ensure resource directories are importable for multiprocessing code paths.
for path in {SCRIPTS_DIR, ROOT_DIR / ".claude", ROOT_DIR}:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

env_paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
if str(SCRIPTS_DIR) not in env_paths:
    env_paths.insert(0, str(SCRIPTS_DIR))
    os.environ["PYTHONPATH"] = os.pathsep.join(env_paths)

_sop_module = load_module(
    "get_sop_improved_la_new",
    "backend/resources/scripts/get_sop_improved_la_new.py",
)
_func_main = getattr(_sop_module, "func_main_pure_improved", None)
if _func_main is None:
    raise AttributeError("get_sop_improved_la_new.py 缺少 func_main_pure_improved 入口函数")

SENTENCEWISE_DEFAULTS = getattr(_sop_module, "SENTENCEWISE_DEFAULT_THRESHOLDS", None)


def _load_default_config() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "logic_tree_path": str(DEFAULT_LOGIC_TREE),
        "similarity": 0.9,
        "batch_size": 100,
        "sentencewise": {
            "enabled": True,
            "similarity": 0.8,
            "seq_ratio": 0.72,
            "level_thresholds": "__USE_DEFAULT__",
            "min_level_matches": 1,
            "trigger_on_partial": True,
            "parallel_workers": 4,
        },
    }
    if DEFAULT_CONFIG_FILE.exists():
        try:
            with DEFAULT_CONFIG_FILE.open("r", encoding="utf-8") as f:
                file_config = json.load(f)
            defaults.update(file_config)
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


GLOBAL_DEFAULTS = _load_default_config()


def _resolve_logic_tree(custom_path: str | None) -> Path:
    if custom_path:
        candidate = Path(custom_path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"SOP逻辑树文件不存在: {candidate}")
        return candidate

    fallback = Path(GLOBAL_DEFAULTS.get("logic_tree_path", DEFAULT_LOGIC_TREE)).expanduser()
    if fallback.exists():
        return fallback

    if DEFAULT_LOGIC_TREE.exists():
        return DEFAULT_LOGIC_TREE

    raise FileNotFoundError(f"找不到默认逻辑树文件: {DEFAULT_LOGIC_TREE}")


def _coerce_float(value: Any, default: float) -> float:
    if value in (None, "", []):
        return float(default)
    return float(value)


def _coerce_int(value: Any, default: int) -> int:
    if value in (None, "", []):
        return int(default)
    return int(value)


def _normalize_output_path(corpus_path: Path, params: Dict[str, Any]) -> Path:
    output_dir_param = params.get("output_dir")
    output_filename_param = params.get("output_filename")

    if output_dir_param:
        directory = Path(output_dir_param).expanduser()
    else:
        directory = corpus_path.parent

    if output_filename_param:
        filename = output_filename_param
    else:
        filename = f"{corpus_path.stem}_sop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    output_path = directory / filename
    if not output_path.suffix:
        output_path = output_path.with_suffix(".xlsx")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _build_config(
    corpus_path: Path,
    output_path: Path,
    logic_tree_path: Path,
    similarity: float,
    batch_size: int,
    sentencewise: Dict[str, Any],
) -> Dict[str, Any]:
    functions: list[Dict[str, Any]] = [
        {
            "name": "get_sop_case.func_main",
            "similarity": similarity,
            "batch_size": batch_size,
        }
    ]

    if sentencewise.get("enabled"):
        level_thresholds = sentencewise.get("level_thresholds")
        if (level_thresholds in (None, "__USE_DEFAULT__")) and SENTENCEWISE_DEFAULTS:
            level_thresholds = SENTENCEWISE_DEFAULTS

        functions.append(
            {
                "name": "sentencewise.fallback",
                "enabled": True,
                "similarity": sentencewise.get("similarity", 0.8),
                "seq_ratio": sentencewise.get("seq_ratio", 0.72),
                "level_thresholds": level_thresholds,
                "min_level_matches": sentencewise.get("min_level_matches", 1),
                "trigger_on_partial": sentencewise.get("trigger_on_partial", True),
                "parallel_workers": sentencewise.get("parallel_workers", 4),
            }
        )

    return {
        "corpus_dir": str(corpus_path),
        "pipeline_case_path": str(output_path),
        "sop_logic_tree": str(logic_tree_path),
        "functions": functions,
    }


def _ensure_customer_patterns_file() -> None:
    if not CUSTOMER_PATTERNS_SOURCE.exists():
        return
    try:
        if (
            not CUSTOMER_PATTERNS_TARGET.exists()
            or CUSTOMER_PATTERNS_SOURCE.stat().st_mtime > CUSTOMER_PATTERNS_TARGET.stat().st_mtime
        ):
            shutil.copy2(CUSTOMER_PATTERNS_SOURCE, CUSTOMER_PATTERNS_TARGET)
    except OSError:
        # 非关键文件，复制失败时允许脚本继续运行（脚本将回退到内建配置）
        pass


def _resolve_sentencewise_config(params: Dict[str, Any]) -> Dict[str, Any]:
    defaults = GLOBAL_DEFAULTS.get("sentencewise", {})
    config = defaults.copy() if isinstance(defaults, dict) else {}
    override = params.get("sentencewise")
    if isinstance(override, dict):
        config.update(override)
    return config


def _run(params: Dict[str, Any]) -> ScriptRunResponse:
    _ensure_customer_patterns_file()

    corpus_path = Path(params.get("corpus_path", "")).expanduser()
    if not corpus_path.exists():
        raise FileNotFoundError(f"对话Excel不存在: {corpus_path}")

    logic_tree_path = _resolve_logic_tree(params.get("logic_tree_path"))
    output_path = _normalize_output_path(corpus_path, params)

    similarity = _coerce_float(params.get("similarity"), GLOBAL_DEFAULTS.get("similarity", 0.9))
    batch_size = _coerce_int(params.get("batch_size"), GLOBAL_DEFAULTS.get("batch_size", 100))
    sentencewise_config = _resolve_sentencewise_config(params)

    config_data = _build_config(
        corpus_path,
        output_path,
        logic_tree_path,
        similarity,
        batch_size,
        sentencewise_config,
    )

    log_buffer = io.StringIO()
    try:
        with redirect_stdout(log_buffer):
            _func_main(config_data=config_data)
    except Exception as exc:
        logs = log_buffer.getvalue()
        raise RuntimeError(f"SOP流程执行失败: {exc}\n{logs}") from exc

    generated = output_path.with_name(f"{output_path.stem}_pure_improved{output_path.suffix}")
    final_output = generated if generated.exists() else output_path
    if not final_output.exists():
        raise FileNotFoundError("未找到SOP输出文件，请检查日志")

    logs = log_buffer.getvalue()
    data = {
        "input_file": str(corpus_path),
        "output_file": str(final_output),
        "logic_tree": str(logic_tree_path),
    }
    message = "SOP流程标注完成（纯改进版）"
    return ScriptRunResponse(success=True, message=message, data=data, logs=logs)


SCRIPT_DEFINITION = ScriptDefinition(
    metadata=ScriptMetadata(
        id="run_sop_pipeline",
        name="SOP流程标注（纯改进版）",
        description="直接调用 get_sop_improved_la_new.py，基于最新算法输出 SOP 标签结果。",
        category="SOP分析",
        parameters=[
            ScriptParameter(
                name="corpus_path",
                label="对话Excel文件",
                type="path",
                description="选择或上传需要打SOP标签的Excel文件",
                example="uploads/对话样例.xlsx",
            ),
            ScriptParameter(
                name="output_dir",
                label="结果输出目录",
                type="path",
                required=False,
                description="默认写入与输入文件相同目录，可自定义保存位置",
            ),
            ScriptParameter(
                name="output_filename",
                label="结果文件名",
                type="string",
                required=False,
                description='默认为"输入文件名_sop_时间戳.xlsx"',
            ),
            ScriptParameter(
                name="logic_tree_path",
                label="自定义逻辑树",
                type="path",
                required=False,
                description="若不填写则使用后端内置的SOP逻辑树",
            ),
            ScriptParameter(
                name="similarity",
                label="匹配相似度",
                type="number",
                required=False,
                description="介于0-1，默认按配置文件设置",
                example=0.9,
            ),
            ScriptParameter(
                name="batch_size",
                label="批次大小",
                type="number",
                required=False,
                description="每处理多少行保存一次临时文件，默认按配置文件设置",
                example=100,
            ),
        ],
        output_description="返回生成的纯改进版SOP标注结果路径，并附带执行日志",
    ),
    runner=_run,
)
