from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

ROOT_DIR = Path(__file__).resolve().parents[3]


def _resolve_path(relative_path: str) -> Path:
    candidate = ROOT_DIR / relative_path
    if not candidate.exists():
        raise FileNotFoundError(f"无法找到脚本路径: {candidate}")
    return candidate


@lru_cache(maxsize=None)
def load_module(module_name: str, relative_path: str) -> ModuleType:
    """Dynamically load a module from a file path and cache the result."""

    target_path = _resolve_path(relative_path)
    spec = importlib.util.spec_from_file_location(module_name, target_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {module_name} 创建模块定义")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]

    # 确保模块的父目录在sys.path中，以便多进程环境下的子进程能找到它
    parent_dir = str(target_path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    return module
