#!/usr/bin/env python3
"""Utility script to merge multiple Excel files from a directory into one workbook."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Union

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge all Excel files under a directory into a single workbook."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the Excel files to merge.",
    )
    parser.add_argument(
        "--pattern",
        default="*.xlsx",
        help="Glob pattern for files to include (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="merged.xlsx",
        help="Path of the merged Excel file to write (default: %(default)s).",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Sheet name or index to read from each workbook (default: first sheet).",
    )
    parser.add_argument(
        "--include-source-column",
        action="store_true",
        help="Prepend a 'source_file' column indicating where each row originated.",
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {path}")
    return path.resolve()


def collect_excel_files(directory: Path, pattern: str) -> Sequence[Path]:
    files = sorted(directory.glob(pattern))
    return [path for path in files if path.is_file()]


def parse_sheet_arg(sheet_arg: Union[str, None]) -> Union[str, int, None]:
    if sheet_arg is None:
        return None
    try:
        return int(sheet_arg)
    except ValueError:
        return sheet_arg


def load_frames(files: Sequence[Path], sheet, include_source: bool) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for file_path in files:
        try:
            frame = pd.read_excel(file_path, sheet_name=sheet)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"读取文件失败: {file_path}") from exc

        if include_source:
            frame.insert(0, "source_file", file_path.name)
        frames.append(frame)
    return frames


def main() -> None:
    args = parse_arguments()
    directory = ensure_directory(args.directory)
    files = collect_excel_files(directory, args.pattern)
    if not files:
        raise SystemExit(f"未在 {directory} 中找到匹配 '{args.pattern}' 的 Excel 文件。")

    sheet = parse_sheet_arg(args.sheet)
    frames = load_frames(files, sheet, args.include_source_column)
    merged = pd.concat(frames, ignore_index=True)
    output_path = Path(args.output).resolve()
    merged.to_excel(output_path, index=False)
    print(f"✅ 已合并 {len(files)} 个文件 -> {output_path}")


if __name__ == "__main__":
    main()
