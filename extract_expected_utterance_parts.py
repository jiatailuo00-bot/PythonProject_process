#!/usr/bin/env python3
"""
Post-process pure improved SOP output tables.

Given an Excel exported by get_sop_improved_latest.py, this script inspects the
“预期话术” column and derives two helper columns:
    - 预期话术-传递: the informative statement delivered to the customer.
    - 预期话术-动作: the follow-up action/question posed to the customer.

Rules:
    * Only handle rows whose expected utterance resolves to a dictionary with a
      single entry. Multiple keys imply branching scenarios – keep the helper
      columns empty.
    * The script relies on the SOP logic tree (chengla_wx.json) to build a
      lookup table of canonical utterances. Each canonical utterance is split
      into “transmit” and “action” parts using lightweight heuristics (the last
      question clause becomes the action).
    * For unmatched utterances (e.g. model-generated variants), the same
      heuristic is applied on the fly.

Usage:
    python3 extract_expected_utterance_parts.py \
        --input badcase-3-new_sop-new_pure_improved.xlsx \
        --output badcase-3-new_sop-new_pure_improved_with_parts.xlsx \
        --logictree chengla_wx.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Dict, Iterable, Tuple, Union

import pandas as pd

# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def _normalize_text(text: str) -> str:
    """Normalize spacing while preserving <newline> tokens."""
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").strip()
    # collapse duplicated whitespace but preserve explicit <newline> markers
    return text


def _analysis_view(text: str) -> str:
    """Convert placeholders into analysis-friendly form (replace <newline> with real newline)."""
    return text.replace("<newline>", "\n")


def _restore_tokens(text: str) -> str:
    """Convert analysis-friendly representation back to storage form."""
    return text.replace("\n", "<newline>").strip()


# Utterances that should be treated entirely as actions.
ACTION_ONLY_UTTERANCES = {
    "嗯嗯,咱们在本次直播课中有需要老师重点讲的吗?可以直接发给我,我来反馈",
    "嗯嗯,咱们在本次直播课中有需要老师重点讲的吗？可以直接发给我,我来反馈",
}

# Manual overrides: exact utterance -> (transmit, action)
CUSTOM_SPLIT_OVERRIDES = {
    "了解,自学确实比较难,要自己去找资源、找资料,耗费时间长,但整体学习效果不明显,很多同学刷题会发现速度上去,正确率下降,正确率上来了,速度慢,咱们有这种情况吗?": (
        "了解,自学确实比较难,要自己去找资源、找资料,耗费时间长,但整体学习效果不明显,很多同学刷题会发现速度上去,正确率下降,正确率上来了,速度慢,",
        "咱们有这种情况吗?"
    ),
    "了解,自学确实比较难,要自己去找资源、找资料,耗费时间长,但整体学习效果不明显,很多同学刷题会发现速度上去,正确率下降,正确率上来了,速度慢,咱们有这种情况吗？": (
        "了解,自学确实比较难,要自己去找资源、找资料,耗费时间长,但整体学习效果不明显,很多同学刷题会发现速度上去,正确率下降,正确率上来了,速度慢,",
        "咱们有这种情况吗？"
    ),
    "好的，现在时间越来越紧张了,咱们一定规划好时间,下半年机会多,老师建议只要有考试机会就都报名参加试试,多练练来提升自己的应试能力,这样上岸机会也会更多一些，这几天直播课也会讲更多关于如何备考和考公的做题技巧方法,咱们可以先认真来听,晚上我也会提醒咱们,有问题随时问我哈": (
        "好的，现在时间越来越紧张了,咱们一定规划好时间,下半年机会多,老师建议只要有考试机会就都报名参加试试,多练练来提升自己的应试能力,这样上岸机会也会更多一些，",
        "这几天直播课也会讲更多关于如何备考和考公的做题技巧方法,咱们可以先认真来听,晚上我也会提醒咱们,有问题随时问我哈"
    ),
}

FILLER_PREFIXES = [
    "嗯嗯",
    "嗯呐",
    "嗯",
    "恩恩",
    "哦哦",
    "哦",
    "好嘞",
    "好啦",
    "好哒",
    "好滴",
    "好呢",
    "好呀",
    "好啊",
    "好的",
    "好",
    "行的",
    "行吧",
    "行啦",
    "行呢",
    "行呀",
    "行",
    "那好",
    "那行",
    "收到",
    "了解",
    "知道了",
    "ok",
    "OK",
]


def _extract_leading_filler(text: str) -> Tuple[str, str]:
    """Split off conversational fillers (e.g. 嗯嗯, 好的) from the front of a clause."""
    if not text:
        return "", text

    working = text.lstrip()
    leading_ws_len = len(text) - len(working)
    prefix_ws = text[:leading_ws_len]

    for prefix in sorted(FILLER_PREFIXES, key=len, reverse=True):
        if working.startswith(prefix):
            remainder = working[len(prefix) :]

            punctuation = ""
            while remainder and remainder[0] in {"，", ",", "。", "！", "!", "；", ";", "、"}:
                punctuation += remainder[0]
                remainder = remainder[1:]

            filler = (prefix_ws + prefix + punctuation).rstrip()
            remainder = remainder.lstrip()
            if remainder:
                return filler, remainder

    return "", text


def split_transmit_action(raw_text: str) -> Tuple[str, str]:
    """
    Split an utterance into transmit/action pieces.

    The action is assumed to be the last clause ending with ? or ？.
    Everything before that clause is considered the transmit portion.
    If no question mark exists, the whole text is treated as transmit.
    """
    if not raw_text:
        return "", ""

    normalized = _normalize_text(raw_text)
    analysis = _analysis_view(normalized)

    q_idx_cn = analysis.rfind("？")
    q_idx_en = analysis.rfind("?")
    q_idx = max(q_idx_cn, q_idx_en)
    if q_idx == -1:
        return _restore_tokens(analysis), ""

    # Seek sentence boundary before the question.
    punctuation_stops = ["。", "！", "!", "；", ";", "\n"]
    boundary = -1
    for token in punctuation_stops:
        candidate = analysis.rfind(token, 0, q_idx)
        if candidate > boundary:
            boundary = candidate

    if boundary != -1:
        action_start = boundary + 1
    else:
        comma_positions = [
            idx
            for idx, ch in enumerate(analysis[:q_idx])
            if ch in {"，", ","}
        ]
        if comma_positions:
            action_start = comma_positions[-1] + 1
            action_candidate = analysis[action_start : q_idx + 1].strip()
            if len(action_candidate.replace("\n", "").strip()) <= 12 and len(comma_positions) >= 2:
                action_start = comma_positions[-2] + 1
        else:
            action_start = 0

    raw_transmit = analysis[:action_start]
    raw_action = analysis[action_start : q_idx + 1]

    transmit_part = raw_transmit.rstrip()
    action_clause = raw_action.strip()

    if not transmit_part:
        filler, remainder = _extract_leading_filler(action_clause)
        if filler and remainder:
            transmit_part = filler.rstrip()
            action_clause = remainder.strip()
        else:
            # If everything fell into the action bucket, degrade gracefully.
            return "", _restore_tokens(action_clause)

    transmit = _restore_tokens(transmit_part)
    action = _restore_tokens(action_clause)
    return transmit, action


# --------------------------------------------------------------------------- #
# SOP logic tree traversal
# --------------------------------------------------------------------------- #


def _iter_expected_strings(node: Union[Dict, Iterable, str]) -> Iterable[str]:
    """Yield every string under '预期话术' nodes from the SOP logic tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "预期话术":
                # value can be str, list, or nested dict
                yield from _coerce_expected_values(value)
            else:
                yield from _iter_expected_strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_expected_strings(item)


def _coerce_expected_values(value: Union[Dict, Iterable, str]) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            if isinstance(item, str):
                yield item
    elif isinstance(value, Iterable):
        for item in value:
            if isinstance(item, str):
                yield item


def build_expected_lookup(logictree_path: str) -> Dict[str, Tuple[str, str]]:
    """Pre-compute transmit/action mapping from the SOP logic tree."""
    if not logictree_path or not os.path.exists(logictree_path):
        return {}

    with open(logictree_path, "r", encoding="utf-8") as f:
        sop_data = json.load(f)

    lookup: Dict[str, Tuple[str, str]] = {}
    for text in _iter_expected_strings(sop_data):
        normalized = _normalize_text(text)
        if normalized in lookup:
            continue
        transmit, action = split_transmit_action(normalized)
        lookup[normalized] = (transmit, action)
    return lookup


# --------------------------------------------------------------------------- #
# Dataframe processing
# --------------------------------------------------------------------------- #


def parse_expected_cell(cell_value) -> Union[str, Dict[str, str]]:
    """Normalize the dataframe cell content for the 预期话术 column."""
    if cell_value is None or (isinstance(cell_value, float) and math.isnan(cell_value)):
        return ""

    if isinstance(cell_value, dict):
        return cell_value

    text = str(cell_value).strip()
    if not text:
        return ""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return parsed


def extract_parts_from_cell(cell_value, lookup: Dict[str, Tuple[str, str]]) -> Tuple[str, str]:
    """Return transmit/action strings for a dataframe cell value."""
    parsed = parse_expected_cell(cell_value)

    if isinstance(parsed, dict):
        if len(parsed) != 1:
            return "", ""
        key, value = next(iter(parsed.items()))
        if not isinstance(value, str):
            return "", ""
        normalized = _normalize_text(value)
        if normalized in CUSTOM_SPLIT_OVERRIDES:
            return CUSTOM_SPLIT_OVERRIDES[normalized]
        # special-case keys that should be treated as pure actions
        if key in {"提醒上课"}:
            return "", normalized
        # special-case utterances that should be entirely actions
        if normalized in ACTION_ONLY_UTTERANCES:
            return "", normalized
        if normalized in lookup:
            return lookup[normalized]
        return split_transmit_action(normalized)

    if isinstance(parsed, str) and parsed:
        normalized = _normalize_text(parsed)
        if normalized in CUSTOM_SPLIT_OVERRIDES:
            return CUSTOM_SPLIT_OVERRIDES[normalized]
        if normalized in ACTION_ONLY_UTTERANCES:
            return "", normalized
        if normalized in lookup:
            return lookup[normalized]
        return split_transmit_action(normalized)

    return "", ""


def process_file(input_path: str, output_path: str, logictree_path: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    lookup = build_expected_lookup(logictree_path)

    df = pd.read_excel(input_path)
    transmit_list = []
    action_list = []

    for cell in df.get("预期话术", []):
        transmit, action = extract_parts_from_cell(cell, lookup)
        transmit_list.append(transmit)
        action_list.append(action)

    df["预期话术-传递"] = transmit_list
    df["预期话术-动作"] = action_list

    df.to_excel(output_path, index=False)
    print(f"Processed {len(df)} rows. Output saved to: {output_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract expected utterance transmit/action fields.")
    parser.add_argument("--input", default="节点匹配22-yyds_pure_improved.xlsx", help="Path to the Excel file produced by pure improved pipeline.")
    parser.add_argument("--output", default="节点匹配22-yyds_pure_improved-ne.xlsx", help="Path to write the augmented Excel file.")
    parser.add_argument("--logictree", default="chengla_wx.json", help="Path to SOP logic tree JSON.")
    args = parser.parse_args()

    process_file(args.input, args.output, args.logictree)


if __name__ == "__main__":
    main()
