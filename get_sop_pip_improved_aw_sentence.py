#!/usr/bin/env python3
"""
挖需节点高阈值逐句匹配实验脚本
================================================

思路：
1. 仅针对挖需逻辑树（logictree_aw.json）；
2. 将最终传参上下文中的销售话术逐句拆分；
3. 每句分别与逻辑树中的参考/相似话术做高阈值匹配；
4. 汇总所有命中节点后，在逻辑树中寻找能够完全覆盖的SOP路径；
5. 返回覆盖度最高、相似度最高的路径结果。
"""

from __future__ import annotations

import os
import json
import re
import traceback
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from get_sop_pip import (
    calculate_sentence_similarity,
    list_of_dicts_to_xlsx,
    parse_conversation_history,
    preprocess_sentence,
    read_json_file,
)

# 逻辑树中用于存放元数据的保留字段
META_KEYS = {"参考话术", "相似话术", "下一步动作", "预期话术", "关键信息"}

# SOP 节点层级字段，对齐主流程输出结构
NODE_KEYS = [
    "first_node",
    "second_node",
    "third_node",
    "fourth_node",
    "fifth_node",
    "sixth_node",
    "seventh_node",
    "eighth_node",
    "ninth_node",
]

DEFAULT_LEVEL_THRESHOLDS: Dict[int, float] = {
    2: 0.90,
    3: 0.85
}


@dataclass
class SentenceMatch:
    """单句匹配的结构化结果"""

    path: Tuple[str, ...]
    level: int
    score: float
    cosine_similarity: float
    sequence_ratio: float
    matched_reference: str
    match_type: str
    sentence: str
    sentence_index: int
    sales_message: str
    sales_index: int
    sales_time: str

    def as_level_entry(self) -> Dict[str, Any]:
        return {
            "similarity": self.score,
            "matched_reference": self.matched_reference,
            "match_type": self.match_type,
            "sales_message": self.sales_message,
            "sentence": self.sentence,
            "sentence_index": self.sentence_index,
            "sales_index": self.sales_index,
            "sales_time": self.sales_time,
        }


def _normalize_message(message: str) -> str:
    """统一清洗销售话术，便于拆句"""
    if not isinstance(message, str):
        return ""
    normalized = (
        message.replace("<newline>", "\n")
        .replace("\\n", "\n")
        .replace("\r\n", "\n")
        .replace("：", ":")
    )
    return normalized.strip()


def _split_into_sentences(message: str) -> List[str]:
    """将一条销售话术拆成句子列表"""
    normalized = _normalize_message(message)
    if not normalized:
        return []

    # 先按换行拆分，再细分标点
    candidates: List[str] = []
    for segment in normalized.split("\n"):
        segment = segment.strip()
        if not segment:
            continue
        parts = re.split(r"[。！？?!]+", segment)
        for part in parts:
            part = part.strip()
            if part:
                candidates.append(part)

    if not candidates:
        candidates.append(normalized)
    return candidates


def _iter_scripts(node: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    """遍历节点的参考/相似话术"""
    for script_key in ("参考话术", "相似话术"):
        scripts = node.get(script_key, [])
        if isinstance(scripts, str):
            scripts = [scripts]
        if not isinstance(scripts, Sequence):
            continue
        for script in scripts:
            if isinstance(script, str) and script.strip():
                yield script_key, script.strip()


def _traverse_sop_tree(
    sop_tree: Dict[str, Any],
    similarity_threshold: float,
    seq_ratio_threshold: float,
    level_thresholds: Dict[int, float],
    sentence: str,
    sales_entry: Dict[str, Any],
    sales_idx: int,
    sentence_idx: int,
) -> List[SentenceMatch]:
    """遍历逻辑树，为单句收集所有符合阈值的节点"""

    matches: List[SentenceMatch] = []

    def dfs(current: Dict[str, Any], path: List[str]) -> None:
        if not isinstance(current, dict):
            return

        for script_type, reference in _iter_scripts(current):
            cosine_sim = calculate_sentence_similarity(sentence, reference)
            ratio = SequenceMatcher(None, preprocess_sentence(sentence), preprocess_sentence(reference)).ratio()
            score = max(cosine_sim, ratio)

            level = len(path)
            level_threshold = level_thresholds.get(level, similarity_threshold)

            if cosine_sim >= level_threshold or ratio >= seq_ratio_threshold:
                matches.append(
                    SentenceMatch(
                        path=tuple(path),
                        level=len(path),
                        score=score,
                        cosine_similarity=cosine_sim,
                        sequence_ratio=ratio,
                        matched_reference=reference,
                        match_type=script_type,
                        sentence=sentence,
                        sentence_index=sentence_idx,
                        sales_message=sales_entry["content"],
                        sales_index=sales_idx,
                        sales_time=sales_entry.get("time", ""),
                    )
                )

        for key, value in current.items():
            if isinstance(value, dict) and key not in META_KEYS:
                dfs(value, path + [key])

    for root_name, root_node in sop_tree.items():
        if isinstance(root_node, dict):
            dfs(root_node, [root_name])

    return matches


def _collect_sentence_matches(
    conversations: List[Dict[str, Any]],
    sop_tree: Dict[str, Any],
    similarity_threshold: float,
    seq_ratio_threshold: float,
    level_thresholds: Dict[int, float],
) -> List[SentenceMatch]:
    """针对所有销售话术收集句级匹配结果"""
    all_matches: List[SentenceMatch] = []
    for sales_idx, entry in enumerate(conversations):
        if entry.get("role") != "销售":
            continue
        sentences = _split_into_sentences(entry.get("content", ""))
        if not sentences:
            continue
        for sentence_idx, sentence in enumerate(sentences):
            sentence_matches = _traverse_sop_tree(
                sop_tree=sop_tree,
                similarity_threshold=similarity_threshold,
                seq_ratio_threshold=seq_ratio_threshold,
                level_thresholds=level_thresholds,
                sentence=sentence,
                sales_entry=entry,
                sales_idx=sales_idx,
                sentence_idx=sentence_idx,
            )
            all_matches.extend(sentence_matches)
    # 按消息顺序 + 句子顺序整理，便于后续排布
    all_matches.sort(key=lambda m: (m.sales_index, m.sentence_index, -m.score))
    return all_matches


def _enumerate_branch_paths(sop_tree: Dict[str, Any]) -> List[Tuple[str, ...]]:
    """收集逻辑树中所有可能的分支（包含前缀路径）"""
    paths: List[Tuple[str, ...]] = []

    def dfs(current: Dict[str, Any], path: List[str]) -> None:
        if not isinstance(current, dict):
            return
        # 只保留长度>=2的路径（一级节点不需要匹配）
        if len(path) >= 2:
            paths.append(tuple(path))

        has_child = False
        for key, value in current.items():
            if isinstance(value, dict) and key not in META_KEYS:
                has_child = True
                dfs(value, path + [key])
        if not has_child and len(path) >= 2:
            # 叶子节点已经在上面加入，此处无需重复
            return

    for root_name, root in sop_tree.items():
        if isinstance(root, dict):
            dfs(root, [root_name])

    # 按路径长度降序，优先尝试覆盖更深层级
    paths.sort(key=lambda p: (-len(p), p))
    return paths


def _get_nested_value(sop_tree: Dict[str, Any], path: Sequence[str]) -> Optional[Dict[str, Any]]:
    """安全地按路径取节点"""
    current: Any = sop_tree
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current if isinstance(current, dict) else None


def _assemble_path_matches(
    branch_path: Tuple[str, ...],
    sentence_matches: List[SentenceMatch],
) -> Optional[Dict[int, SentenceMatch]]:
    """
    根据候选句级匹配，在给定路径上寻找一个有序的覆盖链。
    要求：同一路径上各级节点的消息/句子顺序必须递增（允许同一条消息内的后续句子）。
    """
    expected_levels = len(branch_path)
    if expected_levels < 2:
        return None

    result: Dict[int, SentenceMatch] = {}
    current_sales_idx = -1
    current_sentence_idx = -1

    for level in range(2, expected_levels + 1):
        level_path = branch_path[:level]
        candidates = [
            match
            for match in sentence_matches
            if match.path == level_path
        ]
        if not candidates:
            return None

        # 先按相似度排序，再按消息顺序排序，优先选择当前序列之后的句子
        candidates.sort(key=lambda m: (-m.score, m.sales_index, m.sentence_index))

        selected: Optional[SentenceMatch] = None
        for candidate in candidates:
            if candidate.sales_index > current_sales_idx:
                selected = candidate
                break
            if (
                candidate.sales_index == current_sales_idx
                and candidate.sentence_index >= current_sentence_idx
            ):
                selected = candidate
                break

        if not selected:
            # 若没有找到顺序满足的候选，允许退化为最相似的一条（提供诊断信息）
            selected = candidates[0]

        result[level] = selected
        current_sales_idx = selected.sales_index
        current_sentence_idx = selected.sentence_index

    return result


def _build_match_payload(
    branch_path: Tuple[str, ...],
    chain: Dict[int, SentenceMatch],
    sop_tree: Dict[str, Any],
    last_customer_msg: str,
) -> Dict[str, Any]:
    """
    将路径覆盖结果整理为主流程可复用的结构。
    """
    node_info = {NODE_KEYS[idx]: branch_path[idx] for idx in range(min(len(branch_path), len(NODE_KEYS)))}
    all_level_matches: Dict[int, Dict[str, Any]] = {
        level: chain[level].as_level_entry() for level in chain
    }

    final_match: Dict[str, Any] = {
        **node_info,
        "best_path": list(branch_path),
        "all_level_matches": all_level_matches,
        "matched_levels": sorted(all_level_matches.keys()),
        "sop_node_path_str": " -> ".join(branch_path),
        "sales_message": chain[max(chain.keys())].sales_message if chain else "",
        "sales_time": chain[max(chain.keys())].sales_time if chain else "",
        "similarity": chain[max(chain.keys())].score if chain else 0.0,
        "matched_reference": chain[max(chain.keys())].matched_reference if chain else "",
        "match_type": chain[max(chain.keys())].match_type if chain else "",
        "sentence_trace": [
            {
                "level": level,
                "node": branch_path[level - 1],
                "sentence": match.sentence,
                "sales_message": match.sales_message,
                "similarity": match.score,
                "cosine_similarity": match.cosine_similarity,
                "sequence_ratio": match.sequence_ratio,
                "sales_index": match.sales_index,
                "sentence_index": match.sentence_index,
                "matched_reference": match.matched_reference,
                "match_type": match.match_type,
            }
            for level, match in sorted(chain.items())
        ],
        "missing_levels": [
            level for level in range(2, len(branch_path) + 1) if level not in chain
        ],
        "expected_utterance": "",
        "next_action": "",
        "last_customer_msg": last_customer_msg,
    }

    final_node = _get_nested_value(sop_tree, branch_path)
    if isinstance(final_node, dict):
        next_action = final_node.get("下一步动作", "")
        expected = final_node.get("预期话术", "")
        final_match["next_action"] = next_action
        final_match["expected_utterance"] = expected

    return final_match


def find_best_sop_match_sentencewise(
    conversations: List[Dict[str, Any]],
    sop_tree: Dict[str, Any],
    similarity_threshold: float = 0.80,
    seq_ratio_threshold: float = 0.72,
    level_thresholds: Optional[Dict[int, float]] = None,
    min_level_matches: int = 1,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    逐句匹配挖需逻辑树，返回覆盖度最高的路径。
    """
    debug_info: Dict[str, Any] = {
        "reason": "",
        "matched_sales_message": "",
        "max_similarity": 0.0,
        "max_similarity_path": [],
        "max_similarity_reference": "",
        "sentence_match_count": 0,
    }

    if not conversations:
        debug_info["reason"] = "无有效对话"
        return None, debug_info

    last_customer_msg = ""
    for conv in reversed(conversations):
        if conv.get("role") == "客户":
            last_customer_msg = conv.get("content", "")
            break

    level_thresholds = level_thresholds or DEFAULT_LEVEL_THRESHOLDS

    sentence_matches = _collect_sentence_matches(
        conversations=conversations,
        sop_tree=sop_tree,
        similarity_threshold=similarity_threshold,
        seq_ratio_threshold=seq_ratio_threshold,
        level_thresholds=level_thresholds,
    )

    if not sentence_matches:
        debug_info["reason"] = "未找到命中阈值的销售话术"
        return None, debug_info

    debug_info["sentence_match_count"] = len(sentence_matches)
    best_match = max(sentence_matches, key=lambda m: m.score)
    debug_info["max_similarity"] = best_match.score
    debug_info["max_similarity_reference"] = best_match.matched_reference
    debug_info["max_similarity_path"] = list(best_match.path)

    branch_paths = _enumerate_branch_paths(sop_tree)

    def select_best(candidate_paths: List[Tuple[str, ...]], require_full: bool = False) -> Optional[Dict[str, Any]]:
        best_local_payload: Optional[Dict[str, Any]] = None
        best_local_rank: Optional[Tuple[int, float, float, float, float, int]] = None

        for branch_path in candidate_paths:
            chain = _assemble_path_matches(branch_path, sentence_matches)
            if not chain:
                continue
            coverage = len(chain)
            if coverage < min_level_matches:
                continue
            expected = len(branch_path) - 1
             # require all下级命中时，用于拓展
            if require_full and coverage < expected:
                continue
            coverage_ratio = coverage / max(expected, 1)
            min_similarity = min(match.score for match in chain.values())
            avg_similarity = sum(match.score for match in chain.values()) / coverage
            deepest_message_index = max(match.sales_index for match in chain.values())
            shallowest_message_index = min(match.sales_index for match in chain.values())

            rank_key = (
                deepest_message_index,
                coverage,
                coverage_ratio,
                min_similarity,
                avg_similarity,
                len(branch_path),
                -shallowest_message_index,
            )

            if best_local_rank is None or rank_key > best_local_rank:
                best_local_rank = rank_key
                best_local_payload = _build_match_payload(branch_path, chain, sop_tree, last_customer_msg)

        return best_local_payload

    # Step 1: 优先寻找二级节点（长度为2的路径）
    level2_paths = [path for path in branch_paths if len(path) == 2]
    best_payload = select_best(level2_paths)
    if best_payload:
        chosen_prefix = tuple(best_payload.get("best_path", []))
        extended_paths = [
            path for path in branch_paths
            if len(path) > len(chosen_prefix) and tuple(path[:len(chosen_prefix)]) == chosen_prefix
        ]
        extended_payload = select_best(extended_paths, require_full=True) if extended_paths else None
        final_payload = extended_payload or best_payload
        debug_info["matched_sales_message"] = final_payload.get("sales_message", "")
        debug_info["reason"] = ""
        return final_payload, debug_info

    # Step 2: 若未定位到二级节点，再尝试更长的路径
    remaining_paths = sorted([p for p in branch_paths if len(p) != 2], key=lambda p: (-len(p), p))
    best_payload = select_best(remaining_paths)

    if best_payload:
        debug_info["matched_sales_message"] = best_payload.get("sales_message", "")
        debug_info["reason"] = ""
        return best_payload, debug_info

    debug_info["reason"] = "未能组合出符合顺序要求的SOP路径"
    return None, debug_info


def analyze_sentencewise_logic(
    sales_corpus_xlsx: str,
    sop_logic_tree_path: str,
    similarity_threshold: float = 0.80,
    seq_ratio_threshold: float = 0.72,
    level_thresholds: Optional[Dict[int, float]] = None,
    min_level_matches: int = 1,
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    主流程：按行读取 Excel，对每条对话执行逐句匹配。
    """
    try:
        df = pd.read_excel(sales_corpus_xlsx, engine="openpyxl")
        print(f"成功读取销售对话汇总文件，共 {len(df)} 条记录")

        conversation_column = None
        for col in ["最终传参上下文", "对话历史", "历史对话"]:
            if col in df.columns:
                conversation_column = col
                break

        if not conversation_column:
            print("错误：未找到对话历史列")
            return []

        print(f"使用对话列：{conversation_column}")

        sop_tree = read_json_file(sop_logic_tree_path)
        if not sop_tree:
            print("无法读取SOP逻辑树文件")
            return []

        labeled_records: List[Dict[str, Any]] = []
        temp_dir = os.path.dirname(sales_corpus_xlsx)
        temp_base = os.path.splitext(os.path.basename(sales_corpus_xlsx))[0]
        temp_file_path = os.path.join(temp_dir, f"{temp_base}_sentencewise_temp.xlsx")

        improved_matches = 0
        no_matches = 0

        for idx, row in df.iterrows():
            history_str = row.get(conversation_column, "")
            conversations = parse_conversation_history(history_str)
            match_result, debug_info = find_best_sop_match_sentencewise(
                conversations=conversations,
                sop_tree=sop_tree,
                similarity_threshold=similarity_threshold,
                seq_ratio_threshold=seq_ratio_threshold,
                level_thresholds=level_thresholds,
                min_level_matches=min_level_matches,
            )

            last_customer_msg = ""
            for conv in reversed(conversations):
                if conv.get("role") == "客户":
                    last_customer_msg = conv.get("content", "")
                    break

            reasons: List[str] = []
            if not match_result:
                no_matches += 1
                if debug_info.get("reason"):
                    reasons.append(debug_info["reason"])
                if debug_info.get("max_similarity"):
                    reasons.append(f"最高相似度: {debug_info['max_similarity']:.2f}")
                if debug_info.get("max_similarity_path"):
                    reasons.append("最高相似度节点路径: " + " -> ".join(debug_info["max_similarity_path"]))
                if debug_info.get("max_similarity_reference"):
                    reasons.append(f"最高相似度参考话术: {debug_info['max_similarity_reference']}")
            else:
                improved_matches += 1

            labeled_record = {
                "最后客户消息": last_customer_msg,
                "匹配方法": "句级高阈值匹配" if match_result else "无匹配",
                "最近销售消息": match_result.get("sales_message", "") if match_result else "",
                "销售消息时间": match_result.get("sales_time", "") if match_result else "",
                "SOP一级节点": match_result.get("first_node", "") if match_result else "",
                "SOP二级节点": match_result.get("second_node", "") if match_result else "",
                "SOP三级节点": match_result.get("third_node", "") if match_result else "",
                "SOP四级节点": match_result.get("fourth_node", "") if match_result else "",
                "SOP五级节点": match_result.get("fifth_node", "") if match_result else "",
                "SOP六级节点": match_result.get("sixth_node", "") if match_result else "",
                "SOP七级节点": match_result.get("seventh_node", "") if match_result else "",
                "SOP八级节点": match_result.get("eighth_node", "") if match_result else "",
                "SOP九级节点": match_result.get("ninth_node", "") if match_result else "",
                "匹配相似度": match_result.get("similarity", 0.0) if match_result else 0.0,
                "匹配的参考话术": match_result.get("matched_reference", "") if match_result else "",
                "匹配类型": match_result.get("match_type", "") if match_result else "",
                "预期话术": match_result.get("expected_utterance", "") if match_result else "",
                "下一步动作": match_result.get("next_action", "") if match_result else "",
                "SOP节点路径": match_result.get("sop_node_path_str", "") if match_result else "",
                "诊断最高相似度": debug_info.get("max_similarity", 0.0),
                "诊断候选节点路径": " -> ".join(debug_info.get("max_similarity_path", [])),
                "诊断候选参考话术": debug_info.get("max_similarity_reference", ""),
                "匹配备注": "; ".join(dict.fromkeys([r for r in reasons if r])),
            }

            sentence_trace = match_result.get("sentence_trace", []) if match_result else []
            labeled_record["命中句详情"] = json.dumps(sentence_trace, ensure_ascii=False) if sentence_trace else ""

            level_names = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
            all_level_matches = match_result.get("all_level_matches", {}) if match_result else {}
            for level in range(1, 10):
                level_name = level_names[level]
                level_match = all_level_matches.get(level)
                if level_match:
                    labeled_record[f"SOP{level_name}级节点匹配相似度"] = level_match.get("similarity", 0.0)
                    labeled_record[f"SOP{level_name}级节点匹配参考话术"] = level_match.get("matched_reference", "")
                else:
                    labeled_record[f"SOP{level_name}级节点匹配相似度"] = 0.0
                    labeled_record[f"SOP{level_name}级节点匹配参考话术"] = ""

            for col in df.columns:
                if col not in labeled_record:
                    labeled_record[col] = row.get(col, "")

            labeled_records.append(labeled_record)

            if (idx + 1) % batch_size == 0:
                try:
                    pd.DataFrame(labeled_records).to_excel(temp_file_path, index=False, engine="openpyxl")
                    print(f"已处理 {idx + 1} 行，保存进度到临时文件")
                    print(f"匹配成功: {improved_matches}，无匹配: {no_matches}")
                except Exception as save_error:
                    print(f"警告：保存临时文件失败 - {save_error}")

        if labeled_records:
            try:
                pd.DataFrame(labeled_records).to_excel(temp_file_path, index=False, engine="openpyxl")
                print(f"最终保存 {len(labeled_records)} 条记录到临时文件：{temp_file_path}")
            except Exception as save_error:
                print(f"警告：最终保存临时文件失败 - {save_error}")

        print("\n=== 句级高阈值匹配统计 ===")
        print(f"匹配成功: {improved_matches} 条")
        print(f"无匹配: {no_matches} 条")
        if labeled_records:
            print(f"匹配率: {improved_matches / len(labeled_records) * 100:.1f}%")

        return labeled_records

    except Exception as exc:  # pragma: no cover - 记录异常
        print(f"处理过程中发生错误：{exc}")
        print(traceback.format_exc())
        return []


def func_main_sentencewise(**kwargs: Any) -> None:
    """CLI 入口函数"""
    config_data = kwargs["config_data"]
    corpus_path = config_data["corpus_dir"]
    output_path = config_data["pipeline_case_path"]
    sop_path = config_data["sop_logic_tree"]

    similarity = 0.80
    seq_ratio = 0.72
    level_thresholds = None
    min_level_matches = 1
    batch_size = 100

    for func_cfg in config_data.get("functions", []):
        if func_cfg.get("name") == "sentencewise.func_main":
            similarity = func_cfg.get("similarity", similarity)
            seq_ratio = func_cfg.get("seq_ratio", seq_ratio)
            level_thresholds = func_cfg.get("level_thresholds", level_thresholds)
            min_level_matches = func_cfg.get("min_level_matches", min_level_matches)
            batch_size = func_cfg.get("batch_size", batch_size)

    print("=== 句级高阈值匹配实验脚本 ===")
    print(f"输入文件: {corpus_path}")
    print(f"输出文件: {output_path}")
    print(f"SOP逻辑树: {sop_path}")
    print(f"相似度阈值: {similarity}")
    print(f"序列匹配阈值: {seq_ratio}")
    if level_thresholds:
        print(f"分层阈值设置: {level_thresholds}")
    print(f"最少命中层级数: {min_level_matches}")

    labeled_records = analyze_sentencewise_logic(
        sales_corpus_xlsx=corpus_path,
        sop_logic_tree_path=sop_path,
        similarity_threshold=similarity,
        seq_ratio_threshold=seq_ratio,
        level_thresholds=level_thresholds,
        min_level_matches=min_level_matches,
        batch_size=batch_size,
    )

    if labeled_records:
        if list_of_dicts_to_xlsx(labeled_records, output_path):
            print(f"结果已保存到 {output_path}")
    else:
        print("未生成有效结果文件")


if __name__ == "__main__":
    default_config = {
        "corpus_dir": "badcase9.xlsx",
        "pipeline_case_path": "badcase9_sentencewise.xlsx",
        "sop_logic_tree": "logictree_aw.json",
        "functions": [
            {
                "name": "sentencewise.func_main",
                "similarity": 0.80,
                "seq_ratio": 0.72,
                "level_thresholds": DEFAULT_LEVEL_THRESHOLDS,
                "min_level_matches": 1,
                "batch_size": 100,
            }
        ],
    }
    func_main_sentencewise(config_data=default_config)
