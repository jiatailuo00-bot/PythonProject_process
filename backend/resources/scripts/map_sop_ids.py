#!/usr/bin/env python3
"""
Map SOP level IDs from the master definition workbook into the case workbook.

Usage:
    python map_sop_ids.py --cases 初版案例.xlsx --sop 珍酒sop1.xlsx --output 初版案例.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

MASTER_LEVEL1_COL = "任务主题（一级节点）"
MASTER_LEVEL2_COL = "子任务主题（二级节点）"
MASTER_ID_COL = "ID"
MASTER_SUBTASK_ID_COL = "子任务ID"

CASE_LEVEL1_COL = "SOP一级节点"
CASE_LEVEL2_COL = "SOP二级节点"
CASE_LEVEL1_ID_COL = "SOP一级节点ID"
CASE_LEVEL2_ID_COL = "SOP二级节点ID"


def normalise_series(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to stripped strings, preserving NaN."""
    return series.astype(str).str.strip()


def build_lookup(master_df: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Create a mapping of (一级节点, 二级节点) -> (ID, 子任务ID)."""
    master = master_df.copy()
    master[[MASTER_ID_COL, MASTER_LEVEL1_COL]] = master[[MASTER_ID_COL, MASTER_LEVEL1_COL]].ffill()
    master = master.dropna(subset=[MASTER_LEVEL2_COL])
    master = master[master[MASTER_ID_COL].notna()]
    master = master[normalise_series(master[MASTER_ID_COL]) != "任务ID"]

    for col in (MASTER_LEVEL1_COL, MASTER_LEVEL2_COL, MASTER_ID_COL, MASTER_SUBTASK_ID_COL):
        master[col] = normalise_series(master[col])

    duplicates = master.duplicated(subset=[MASTER_LEVEL1_COL, MASTER_LEVEL2_COL], keep=False)
    if duplicates.any():
        dup_rows = master.loc[duplicates, [MASTER_LEVEL1_COL, MASTER_LEVEL2_COL, MASTER_ID_COL, MASTER_SUBTASK_ID_COL]]
        raise ValueError(
            "Duplicate master entries detected for the following combinations:\n"
            f"{dup_rows.to_string(index=False)}"
        )

    return {
        (row[MASTER_LEVEL1_COL], row[MASTER_LEVEL2_COL]): (row[MASTER_ID_COL], row[MASTER_SUBTASK_ID_COL])
        for _, row in master.iterrows()
    }


def build_level1_index(lookup: Dict[Tuple[str, str], Tuple[str, str]]) -> Dict[str, List[Tuple[str, str, str]]]:
    """Group lookup entries by level1 for fallback matching."""
    level1_index: Dict[str, List[Tuple[str, str, str]]] = {}
    for (level1, level2), (id1, id2) in lookup.items():
        level1_index.setdefault(level1, []).append((level2, id1, id2))
    return level1_index


def resolve_ids(
    level1: str,
    level2: str,
    lookup: Dict[Tuple[str, str], Tuple[str, str]],
    level1_index: Dict[str, List[Tuple[str, str, str]]],
) -> Tuple[str, str]:
    """Return (ID, 子任务ID) with graceful fallbacks; raise if unresolved."""
    key = (level1, level2)
    if key in lookup:
        return lookup[key]

    candidates = level1_index.get(level1, [])
    if not level2:
        raise KeyError(f"Missing 二级节点 for 一级节点 '{level1}'")

    startswith_matches = [entry for entry in candidates if entry[0].startswith(level2)]
    if len(startswith_matches) == 1:
        _, id1, id2 = startswith_matches[0]
        return id1, id2

    reverse_matches = [entry for entry in candidates if level2.startswith(entry[0])]
    if len(reverse_matches) == 1:
        _, id1, id2 = reverse_matches[0]
        return id1, id2

    raise KeyError(f"Unable to map '{level1}' / '{level2}' to master SOP IDs")


def map_case_ids(
    cases_df: pd.DataFrame,
    lookup: Dict[Tuple[str, str], Tuple[str, str]],
    level1_index: Dict[str, List[Tuple[str, str, str]]],
) -> pd.DataFrame:
    """Append ID columns to the case DataFrame."""
    level1_ids: List[str] = []
    level2_ids: List[str] = []
    missing: List[Tuple[str, str]] = []

    for _, row in cases_df.iterrows():
        level1_raw = row.get(CASE_LEVEL1_COL, "")
        level2_raw = row.get(CASE_LEVEL2_COL, "")

        level1 = str(level1_raw).strip() if pd.notna(level1_raw) else ""
        level2 = str(level2_raw).strip() if pd.notna(level2_raw) else ""

        if not level1:
            level1_ids.append("")
            level2_ids.append("")
            continue

        try:
            id1, id2 = resolve_ids(level1, level2, lookup, level1_index)
            level1_ids.append(id1)
            level2_ids.append(id2)
        except KeyError:
            missing.append((level1, level2))
            level1_ids.append("")
            level2_ids.append("")

    if missing:
        unique_missing = sorted(set(missing))
        details = "\n".join(f"- {level1} / {level2}" for level1, level2 in unique_missing)
        raise ValueError(f"Unmapped SOP节点组合:\n{details}")

    result = cases_df.copy()
    result[CASE_LEVEL1_ID_COL] = level1_ids
    result[CASE_LEVEL2_ID_COL] = level2_ids
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map SOP IDs into the case workbook.")
    parser.add_argument("--cases", default="quit_1.xlsx",help="Path to 初版案例.xlsx or equivalent case workbook.")
    parser.add_argument("--sop", default="珍酒sop1.xlsx", help="Path to 珍酒sop1.xlsx or equivalent SOP definition workbook.")
    parser.add_argument(
        "--output",
        help="Output path for the updated case workbook (defaults to in-place overwrite).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    sop_path = Path(args.sop)
    output_path = Path(args.output) if args.output else cases_path

    cases_df = pd.read_excel(cases_path)
    master_df = pd.read_excel(sop_path)

    lookup = build_lookup(master_df)
    level1_index = build_level1_index(lookup)
    enriched_cases = map_case_ids(cases_df, lookup, level1_index)

    enriched_cases.to_excel(output_path, index=False)
    print(f"写入完成: {output_path}")


if __name__ == "__main__":
    main()
