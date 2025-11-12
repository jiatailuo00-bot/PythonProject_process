#!/usr/bin/env python3
"""
纯改进版SOP识别脚本 - 直接使用改进版验证逻辑
不使用原版逻辑作为备选，完全依赖改进版的"查找符合期望SOP节点的销售话术"逻辑
"""

import traceback
import jieba
import pandas as pd
import os
import json
import re
import math

from typing import Dict, Any, Tuple, List, Optional, Set
from collections import defaultdict, Counter
import copy
from difflib import SequenceMatcher
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

# 从原文件导入基础函数
from get_sop_pip import (
    preprocess_sentence,
    calculate_sentence_similarity,
    parse_conversation_history,
    find_all_level_sop_matches,
    find_nearest_sales_sop_match,
    read_json_file,
    list_of_dicts_to_xlsx,
    find_best_similarity_candidate
)

from get_sop_pip_improved import find_targeted_sop_match
from get_sop_pip_improved_v2 import find_best_sop_match_improved as fallback_find_best

try:
    from get_sop_pip_improved_aw_sentence import (
        find_best_sop_match_sentencewise as sentencewise_matcher,
        DEFAULT_LEVEL_THRESHOLDS as SENTENCEWISE_DEFAULT_THRESHOLDS,
        _collect_sentence_matches
    )
except ImportError:
    sentencewise_matcher = None
    SENTENCEWISE_DEFAULT_THRESHOLDS = {}


def _get_nested_value(d: Dict, keys: list) -> Any:
    """
    Safely retrieves a value from a nested dictionary using a list of keys.
    Returns None if any key in the path is not found.
    """
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


SOP_META_KEYS = {"参考话术", "相似话术", "下一步动作", "预期话术", "关键信息"}

CHINESE_NUMERALS = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
}

STRONG_VALIDATION_NODE_LABELS = {
    level: f"强校验SOP{CHINESE_NUMERALS[level]}级节点"
    for level in range(1, 10)
}

SOP_CACHE_REGISTRY: Dict[int, List[Dict[str, Any]]] = {}


@lru_cache(maxsize=200000)
def _sentence_features(text: str) -> Tuple[str, Tuple[str, ...], Counter, float, frozenset]:
    """
    返回句子的预处理结果、分词序列、词频、范数以及中文字符集合。
    结果会被缓存，避免重复分词。
    """
    processed = preprocess_sentence(text) if text else ""
    if not processed:
        return "", tuple(), Counter(), 0.0, frozenset()
    tokens = tuple(jieba.cut(processed))
    counter = Counter(tokens)
    norm = math.sqrt(sum(v * v for v in counter.values()))
    char_set = frozenset(ch for ch in processed if "\u4e00" <= ch <= "\u9fff")
    return processed, tokens, counter, norm, char_set


def _cosine_from_counters(counter_a: Counter, norm_a: float, counter_b: Counter, norm_b: float) -> float:
    """
    基于词频向量计算余弦相似度，传入预计算的范数以避免重复开方。
    """
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    if len(counter_a) <= len(counter_b):
        smaller, larger = counter_a, counter_b
    else:
        smaller, larger = counter_b, counter_a
    dot = 0.0
    for token, count in smaller.items():
        other = larger.get(token)
        if other:
            dot += count * other
    if dot == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


@lru_cache(maxsize=50000)
def _cached_similarity(text_a: str, text_b: str) -> float:
    """
    基于词频特征的句对相似度缓存。
    """
    _, _, counter_a, norm_a, char_set_a = _sentence_features(text_a)
    _, _, counter_b, norm_b, char_set_b = _sentence_features(text_b)
    if char_set_a and char_set_b and not (char_set_a & char_set_b):
        return 0.0
    return _cosine_from_counters(counter_a, norm_a, counter_b, norm_b)


def _preprocess_sop_tree(sop_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将SOP树扁平化为可快速遍历的节点列表，减少重复DFS。
    """
    cache_key = id(sop_data)
    cached = SOP_CACHE_REGISTRY.get(cache_key)
    if cached is not None:
        return cached

    entries: List[Dict[str, Any]] = []

    def _walk(node: Dict[str, Any], path: List[str]):
        if not isinstance(node, dict):
            return

        scripts: List[Dict[str, Any]] = []
        for key in ("参考话术", "相似话术"):
            scripts_val = node.get(key, [])
            if isinstance(scripts_val, list):
                iterable = scripts_val
            elif isinstance(scripts_val, str):
                iterable = [scripts_val]
            else:
                iterable = []
            for script in iterable:
                if not isinstance(script, str) or not script.strip():
                    continue
                processed, _, counter, norm, char_set = _sentence_features(script)
                scripts.append({
                    "text": script,
                    "processed": processed,
                    "type": key,
                    "counter": counter,
                    "norm": norm,
                    "char_set": char_set,
                })

        if scripts and path:
            entries.append({
                "path": tuple(path),
                "depth": len(path),
                "scripts": scripts,
            })

        for child_key, child_value in node.items():
            if isinstance(child_value, dict) and child_key not in SOP_META_KEYS:
                _walk(child_value, path + [child_key])

    for root_key, root_value in sop_data.items():
        if root_key in SOP_META_KEYS:
            continue
        if isinstance(root_value, dict):
            _walk(root_value, [root_key])

    SOP_CACHE_REGISTRY[cache_key] = entries
    return entries


def _find_best_match_in_entries(path_tuple: Tuple[str, ...],
                                entries: List[Dict[str, Any]],
                                similarity_threshold: float) -> Optional[Dict[str, Any]]:
    """
    在给定 entries 中寻找与 path_tuple 匹配度最高的节点信息。
    """
    best_entry = None
    best_info = None
    best_sim = similarity_threshold
    for entry in entries:
        info = entry.get("matches_by_path", {}).get(path_tuple)
        if not info:
            continue
        sim = info.get("similarity", 0.0)
        if sim < similarity_threshold:
            continue
        if sim > best_sim or best_info is None:
            best_sim = sim
            best_entry = entry
            best_info = info
    if best_info and best_entry:
        return {
            'similarity': best_info.get('similarity', 0.0),
            'matched_reference': best_info.get('matched_reference', ''),
            'match_type': best_info.get('match_type', ''),
            'sales_message': best_entry.get('sales_message', ''),
            'sales_time': best_entry.get('sales_time'),
            'sales_index': best_entry.get('index', best_entry.get('sales_index')),
            'path': list(path_tuple)
        }
    return None


_WORKER_SOP_DATA: Optional[Dict[str, Any]] = None
_WORKER_SIMILARITY_THRESHOLD: float = 0.90
_WORKER_STRONG_THRESHOLD: float = 0.95
_WORKER_ERCI_THRESHOLD: float = 0.85


def _init_parallel_worker(sop_data: Dict[str, Any],
                          similarity_threshold: float,
                          strong_threshold: float,
                          erci_threshold: float) -> None:
    """
    初始化并行进程的全局变量，避免重复序列化配置。
    """
    global _WORKER_SOP_DATA, _WORKER_SIMILARITY_THRESHOLD, _WORKER_STRONG_THRESHOLD, _WORKER_ERCI_THRESHOLD
    _WORKER_SOP_DATA = sop_data
    _WORKER_SIMILARITY_THRESHOLD = similarity_threshold
    _WORKER_STRONG_THRESHOLD = strong_threshold
    _WORKER_ERCI_THRESHOLD = erci_threshold
    # 预热一次SOP缓存以减少首批开销
    _preprocess_sop_tree(_WORKER_SOP_DATA)


def _parallel_match_worker(task: Tuple[int, str]) -> Tuple[
    int, Optional[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """
    并行执行单条对话的SOP匹配，返回索引、匹配结果、调试信息和错误描述。
    """
    idx, history_str = task
    if not history_str:
        return idx, None, {"reason": "对话历史为空"}, "empty_history"
    try:
        conversations = parse_conversation_history(history_str)
    except Exception as exc:
        return idx, None, {"reason": f"对话解析异常: {exc}"}, repr(exc)

    if not conversations:
        return idx, None, {"reason": "解析后无有效对话"}, "empty_conversations"

    try:
        match, debug = find_best_sop_match_improved(
            conversations,
            _WORKER_SOP_DATA,
            similarity_threshold=_WORKER_SIMILARITY_THRESHOLD,
            strong_validation_threshold=_WORKER_STRONG_THRESHOLD,
            erci_strong_threshold=_WORKER_ERCI_THRESHOLD,
        )
        return idx, match, debug, None
    except Exception as exc:
        return idx, None, {"reason": f"并行匹配异常: {exc}"}, repr(exc)


def _predict_full_strong_path(strong_path: List[str], sop_data: Dict[str, Any]) -> List[str]:
    """
    根据强校验路径，沿着SOP树推断剩余层级（选择首个可用子节点，保证顺序可重复）。
    """
    if not strong_path:
        return []

    predicted = list(strong_path)
    current_node = sop_data
    for key in strong_path:
        if not isinstance(current_node, dict) or key not in current_node:
            return predicted
        current_node = current_node.get(key, {})

    visited = set()
    while isinstance(current_node, dict):
        candidates = [
            child_key for child_key, child_value in current_node.items()
            if isinstance(child_value, dict) and child_key not in SOP_META_KEYS
        ]
        if not candidates:
            break
        candidates.sort()
        next_key = candidates[0]
        predicted.append(next_key)
        if next_key in visited:
            break
        visited.add(next_key)
        current_node = current_node.get(next_key, {})

        if len(predicted) >= 9:
            break
    return predicted


def _extract_best_path(match_obj: Optional[Dict[str, Any]]) -> List[str]:
    if not match_obj:
        return []
    path = match_obj.get('best_path')
    if path:
        return list(path)
    node_keys = ['first_node', 'second_node', 'third_node', 'fourth_node',
                 'fifth_node', 'sixth_node', 'seventh_node', 'eighth_node', 'ninth_node']
    inferred = [match_obj.get(key) for key in node_keys if match_obj.get(key)]
    return inferred


def _attach_regular_tail_metadata(match_obj: Optional[Dict[str, Any]],
                                  sop_data: Dict[str, Any],
                                  last_customer_msg: str) -> Dict[str, Any]:
    if not match_obj:
        return {'expected_utterance': '', 'next_action': '', 'core_info': ''}
    path = _extract_best_path(match_obj)
    meta = _resolve_expected_fields(sop_data, path, last_customer_msg) if path else {
        'expected_utterance': '', 'next_action': '', 'core_info': ''
    }
    match_obj['regular_tail_expected'] = meta.get('expected_utterance', '')
    match_obj['regular_tail_core'] = meta.get('core_info', '')
    match_obj['regular_tail_next_action'] = meta.get('next_action', '')
    return meta


def _resolve_expected_fields(sop_data: Dict[str, Any],
                             node_path: List[str],
                             last_customer_msg: str,
                             predicted_path: Optional[List[str]] = None) -> Dict[str, Any]:
    result = {
        'expected_utterance': '',
        'next_action': '',
        'core_info': ''
    }
    if not node_path:
        return result

    current_sop_node = _get_nested_value(sop_data, node_path)
    if not isinstance(current_sop_node, dict):
        return result

    result['next_action'] = current_sop_node.get('下一步动作', '')

    depth = len(node_path)
    child_nodes = {
        key: value for key, value in current_sop_node.items()
        if isinstance(value, dict) and key not in SOP_META_KEYS
    }

    def _normalize_core_info(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item not in (None, '')]
        if isinstance(value, dict):
            if not value:
                return []
            try:
                return [json.dumps(value, ensure_ascii=False)]
            except (TypeError, ValueError):
                return [str(value)]
        text = str(value)
        return [text] if text else []

    core_info_map: Dict[str, List[str]] = {}

    if depth <= 3:
        # 一级或二级、三级节点：罗列所有子节点关键信息，便于人工选择分支
        for name, node in sorted(child_nodes.items()):
            info_list = _normalize_core_info(node.get('关键信息', ''))
            if info_list:
                core_info_map[name] = info_list
    else:
        # 四级及以上：优先按照预测路径或下一步动作取下一节点的关键信息
        next_node = None
        next_node_name = None
        if predicted_path and len(predicted_path) > depth:
            next_name = predicted_path[depth]
            if next_name in child_nodes:
                next_node_name = next_name
                next_node = child_nodes[next_name]
        if not next_node:
            next_action = current_sop_node.get('下一步动作')
            if isinstance(next_action, str):
                if next_action in child_nodes:
                    next_node_name = next_action
                    next_node = child_nodes[next_action]
                else:
                    possible = [name for name in child_nodes if name in next_action]
                    if possible:
                        next_node_name = possible[0]
                        next_node = child_nodes[next_node_name]
        if not next_node and len(child_nodes) == 1:
            next_node_name, next_node = next(iter(child_nodes.items()))
        if next_node and child_nodes and next_node_name:
            info_list = _normalize_core_info(next_node.get('关键信息', ''))
            if info_list:
                core_info_map[next_node_name] = info_list

    if core_info_map:
        if len(core_info_map) == 1:
            # 单个键时只返回值列表
            single_value = next(iter(core_info_map.values())) or []
            result['core_info'] = json.dumps(single_value, ensure_ascii=False) if single_value else ''
        else:
            result['core_info'] = json.dumps(core_info_map, ensure_ascii=False)
    else:
        result['core_info'] = ''

    potential_utterances = current_sop_node.get('预期话术')
    if potential_utterances is None:
        return result

    customer_patterns_path = os.path.join(os.path.dirname(__file__), "customer_response_patterns.json")
    customer_response_patterns = {}
    if os.path.exists(customer_patterns_path):
        try:
            with open(customer_patterns_path, 'r', encoding='utf-8') as f:
                customer_response_patterns = json.load(f)
        except json.JSONDecodeError:
            pass

    module_aliases_map = customer_response_patterns.get('module_aliases', {})
    alias_to_canonical_module = {}
    for canonical, aliases in module_aliases_map.items():
        alias_to_canonical_module[canonical.lower()] = canonical
        for alias in aliases:
            alias_to_canonical_module[alias.lower()] = canonical

    combined_module_aliases_map = customer_response_patterns.get('combined_module_aliases', {})
    combined_module_aliases = {
        alias.lower(): list(canonicals)
        for alias, canonicals in combined_module_aliases_map.items()
        if isinstance(alias, str) and isinstance(canonicals, list)
    }

    province_aliases_map = customer_response_patterns.get('province_aliases', {})
    alias_to_canonical_province = {}
    for canonical, aliases in province_aliases_map.items():
        alias_to_canonical_province[canonical.lower()] = canonical
        for alias in aliases:
            alias_to_canonical_province[alias.lower()] = canonical

    module_key_fallbacks: Dict[str, List[str]] = {
        "资料": ["资料", "料"],
    }

    readiness_aliases_map = customer_response_patterns.get('readiness_aliases', {})
    alias_to_readiness = {}
    for canonical, aliases in readiness_aliases_map.items():
        alias_to_readiness[canonical.lower()] = canonical
        for alias in aliases:
            alias_to_readiness[alias.lower()] = canonical

    exam_type_aliases_map = customer_response_patterns.get('exam_type_aliases', {})
    alias_to_exam_type = {}
    for canonical, aliases in exam_type_aliases_map.items():
        alias_to_exam_type[canonical.lower()] = canonical
        for alias in aliases:
            alias_to_exam_type[alias.lower()] = canonical

    previous_preparation_aliases_map = customer_response_patterns.get('previous_preparation_aliases', {})
    alias_to_preparation = {}
    for canonical, aliases in previous_preparation_aliases_map.items():
        alias_to_preparation[canonical.lower()] = canonical
        for alias in aliases:
            alias_to_preparation[alias.lower()] = canonical

    if isinstance(potential_utterances, dict):
        last_customer_msg_lower = last_customer_msg.lower() if last_customer_msg else ''
        identified_items = set()

        sorted_module_aliases = sorted(alias_to_canonical_module.keys(), key=len, reverse=True)
        for alias_term in sorted_module_aliases:
            if alias_term in last_customer_msg_lower:
                identified_items.add(alias_to_canonical_module[alias_term])

        sorted_combined_aliases = sorted(combined_module_aliases.items(), key=lambda item: len(item[0]), reverse=True)
        for alias_term, canonicals in sorted_combined_aliases:
            if alias_term and alias_term in last_customer_msg_lower:
                identified_items.update(canonicals)

        if not identified_items:
            sorted_province_aliases = sorted(alias_to_canonical_province.keys(), key=len, reverse=True)
            for alias_term in sorted_province_aliases:
                if alias_term in last_customer_msg_lower:
                    identified_items.add(alias_to_canonical_province[alias_term])

        sorted_readiness_aliases = sorted(alias_to_readiness.keys(), key=len, reverse=True)
        for alias_term in sorted_readiness_aliases:
            if alias_term in last_customer_msg_lower:
                identified_items.add(alias_to_readiness[alias_term])

        sorted_exam_aliases = sorted(alias_to_exam_type.keys(), key=len, reverse=True)
        for alias_term in sorted_exam_aliases:
            if alias_term in last_customer_msg_lower:
                identified_items.add(alias_to_exam_type[alias_term])

        sorted_preparation_aliases = sorted(alias_to_preparation.keys(), key=len, reverse=True)
        for alias_term in sorted_preparation_aliases:
            if alias_term in last_customer_msg_lower:
                identified_items.add(alias_to_preparation[alias_term])

        matched_dialogues = {}
        for item_name in identified_items:
            candidate_keys = []
            if isinstance(item_name, str) and item_name.startswith("客户回复"):
                candidate_keys.append(item_name)
            else:
                prefixed_key = f"客户回复{item_name}"
                candidate_keys.extend([prefixed_key, item_name])
            if item_name in module_key_fallbacks:
                candidate_keys.extend(module_key_fallbacks[item_name])
            selected_value = None
            for key in candidate_keys:
                if key in potential_utterances:
                    selected_value = potential_utterances[key]
                    break
            if selected_value is not None:
                storage_key = item_name[4:] if isinstance(item_name, str) and item_name.startswith(
                    "客户回复") else item_name
                matched_dialogues[storage_key] = selected_value

        if matched_dialogues:
            if len(matched_dialogues) == 1:
                result['expected_utterance'] = next(iter(matched_dialogues.values()))
            else:
                result['expected_utterance'] = matched_dialogues
        else:
            if isinstance(potential_utterances, dict):
                if len(potential_utterances) == 1:
                    result['expected_utterance'] = next(iter(potential_utterances.values()))
                else:
                    result['expected_utterance'] = potential_utterances
            else:
                result['expected_utterance'] = potential_utterances if potential_utterances else ''
    elif isinstance(potential_utterances, list) and potential_utterances:
        result['expected_utterance'] = potential_utterances[0]
    elif isinstance(potential_utterances, str):
        result['expected_utterance'] = potential_utterances

    return result


def _extend_sentencewise_branch(sentencewise_match: Dict[str, Any],
                                conversations: List[Dict[str, Any]],
                                sop_data: Dict[str, Any],
                                params: Dict[str, Any]) -> Dict[str, Any]:
    """尝试沿句级分支向下扩展，匹配更深层节点"""
    try:
        if not sentencewise_match:
            return sentencewise_match

        match = copy.deepcopy(sentencewise_match)
        base_path = list(match.get('best_path') or [])
        if len(base_path) < 2:
            return match

        sentence_matches = _collect_sentence_matches(
            conversations=conversations,
            sop_tree=sop_data,
            similarity_threshold=params.get("similarity", 0.80),
            seq_ratio_threshold=params.get("seq_ratio", 0.72),
            level_thresholds=params.get("level_thresholds", SENTENCEWISE_DEFAULT_THRESHOLDS)
        )

        matches_by_path: Dict[Tuple[str, ...], List[Any]] = defaultdict(list)
        for m in sentence_matches:
            matches_by_path[tuple(m.path)].append(m)
        for lst in matches_by_path.values():
            lst.sort(key=lambda m: (-m.score, m.sales_index, m.sentence_index))

        sentence_trace = list(match.get('sentence_trace', []))
        all_level_matches = match.get('all_level_matches', {})

        if all_level_matches:
            deepest_existing_level = max(all_level_matches.keys())
            deepest_info = all_level_matches[deepest_existing_level]
            last_sales_idx = deepest_info.get('sales_index', -1)
            last_sentence_idx = deepest_info.get('sentence_index', -1)
        else:
            deepest_existing_level = 1
            last_sales_idx = -1
            last_sentence_idx = -1

        current_path = base_path[:]
        extended = False

        while True:
            parent_node = _get_nested_value(sop_data, current_path)
            if not isinstance(parent_node, dict):
                break

            child_candidates = []
            for child_key, child_value in parent_node.items():
                if not isinstance(child_value, dict) or child_key in SOP_META_KEYS:
                    continue
                new_path = tuple(current_path + [child_key])
                for cand in matches_by_path.get(new_path, []):
                    if last_sales_idx != -1:
                        if cand.sales_index < last_sales_idx:
                            continue
                        if cand.sales_index == last_sales_idx and cand.sentence_index <= last_sentence_idx:
                            continue
                    child_candidates.append((cand.score, -cand.sales_index, -cand.sentence_index, child_key, cand))

            if not child_candidates:
                break

            child_candidates.sort(reverse=True)
            _, _, _, selected_child_key, selected_cand = child_candidates[0]

            current_path.append(selected_child_key)
            level = len(current_path)
            entry = {
                'similarity': selected_cand.score,
                'matched_reference': selected_cand.matched_reference,
                'match_type': selected_cand.match_type,
                'sales_message': selected_cand.sales_message,
                'sales_time': selected_cand.sales_time,
                'sentence': selected_cand.sentence,
                'sentence_index': selected_cand.sentence_index,
                'sales_index': selected_cand.sales_index,
                'path': list(selected_cand.path)
            }
            all_level_matches[level] = entry
            sentence_trace.append({
                'level': level,
                'node': current_path[-1],
                'sentence': selected_cand.sentence,
                'sales_message': selected_cand.sales_message,
                'similarity': selected_cand.score,
                'cosine_similarity': selected_cand.cosine_similarity,
                'sequence_ratio': selected_cand.sequence_ratio,
                'sales_index': selected_cand.sales_index,
                'sentence_index': selected_cand.sentence_index,
                'matched_reference': selected_cand.matched_reference,
                'match_type': selected_cand.match_type,
            })
            last_sales_idx = selected_cand.sales_index
            last_sentence_idx = selected_cand.sentence_index
            extended = True

        if extended:
            match['best_path'] = current_path
            match['sop_node_path_str'] = ' -> '.join(current_path)
            match['all_level_matches'] = all_level_matches
            match['matched_levels'] = sorted(all_level_matches.keys())
            match['missing_levels'] = [lvl for lvl in range(2, len(current_path) + 1) if lvl not in all_level_matches]
            deepest_level = max(all_level_matches.keys())
            deepest_info = all_level_matches[deepest_level]
            match['sales_message'] = deepest_info.get('sales_message', '')
            match['sales_time'] = deepest_info.get('sales_time', '')
            match['similarity'] = deepest_info.get('similarity', 0.0)
            match['matched_reference'] = deepest_info.get('matched_reference', '')
            match['match_type'] = deepest_info.get('match_type', '')
            match['sentence_trace'] = sentence_trace

        return match
    except Exception:
        return sentencewise_match


def _collect_matches_by_path(sales_message: str,
                             sop_data: Dict,
                             similarity_threshold: float,
                             sop_cache: Optional[List[Dict[str, Any]]] = None) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    """收集满足阈值的节点匹配，按路径聚合"""
    matches: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    cleaned_message = (sales_message or "").strip()
    if not cleaned_message:
        return matches

    msg_processed, _, msg_counter, msg_norm, msg_char_set = _sentence_features(cleaned_message)
    if msg_norm == 0.0:
        return matches

    cache = sop_cache or _preprocess_sop_tree(sop_data)
    for entry in cache:
        path_tuple = entry.get("path")
        if not path_tuple:
            continue

        existing = matches.get(path_tuple)
        for script_meta in entry.get("scripts", []):
            script_text = script_meta.get("text", "")
            if not script_text:
                continue
            script_counter = script_meta.get("counter", Counter())
            script_norm = script_meta.get("norm", 0.0)
            if script_norm == 0.0 or msg_norm == 0.0:
                continue
            script_chars = script_meta.get("char_set", frozenset())
            if msg_char_set and script_chars and not (msg_char_set & script_chars):
                continue

            base_similarity = _cosine_from_counters(msg_counter, msg_norm, script_counter, script_norm)
            if base_similarity < similarity_threshold:
                seq_ratio = SequenceMatcher(None, msg_processed, script_meta.get("processed", script_text)).ratio()
                if seq_ratio < 0.4:
                    continue
            else:
                seq_ratio = base_similarity

            effective_similarity = base_similarity if base_similarity >= seq_ratio else seq_ratio
            if effective_similarity < similarity_threshold and seq_ratio < 0.4:
                continue

            if existing is None or effective_similarity > existing["similarity"]:
                matches[path_tuple] = {
                    "similarity": effective_similarity,
                    "matched_reference": script_text,
                    "match_type": script_meta.get("type", ""),
                    "depth": entry.get("depth", len(path_tuple)),
                    "path": list(path_tuple)
                }
                existing = matches[path_tuple]

    return matches


def _merge_strong_with_regular(strong_match: Dict[str, Any],
                               regular_match: Optional[Dict[str, Any]],
                               sop_data: Dict[str, Any],
                               sales_entries: List[Dict[str, Any]],
                               regular_tail_meta: Dict[str, Any],
                               last_customer_msg: str) -> Dict[str, Any]:
    """
    将强校验结果覆盖到常规匹配的前缀层级，保留常规结果中更深的层级信息。
    """
    if not regular_match:
        return copy.deepcopy(strong_match)

    merged = copy.deepcopy(regular_match)
    original_regular_levels = copy.deepcopy(regular_match.get('all_level_matches', {}))
    strong_levels = copy.deepcopy(strong_match.get('all_level_matches', {}))
    regular_levels = merged.get('all_level_matches', {}).copy()
    sales_entries = sales_entries or []
    sales_entries = sales_entries or []
    regular_tail_meta = regular_tail_meta or {'expected_utterance': '', 'core_info': '', 'next_action': ''}
    for level, info in strong_levels.items():
        regular_levels[level] = info
    merged['all_level_matches'] = regular_levels
    merged['matched_levels'] = sorted(regular_levels.keys())
    merged['regular_tail_expected'] = regular_tail_meta.get('expected_utterance', '')
    merged['regular_tail_core'] = regular_tail_meta.get('core_info', '')
    merged['regular_tail_next_action'] = regular_tail_meta.get('next_action', '')

    strong_path = list(strong_match.get('best_path') or [])
    original_best_path = list(regular_match.get('best_path') or [])

    def _tail_in_branch(level_dict, path_hint):
        if not strong_path:
            return True
        matched = sorted(level_dict.keys())
        if not matched:
            return True
        last_level = matched[-1]
        last_info = level_dict.get(last_level, {})
        last_path = list(last_info.get('path') or [])
        if not last_path and path_hint:
            last_path = list(path_hint[:last_level])
        if not last_path:
            return False
        if len(last_path) >= len(strong_path):
            return last_path[:len(strong_path)] == strong_path
        return False

    original_in_branch = _tail_in_branch(original_regular_levels, original_best_path)
    regular_path = list(merged.get('best_path') or [])
    if strong_path:
        if not regular_path:
            merged['best_path'] = strong_path
        elif len(strong_path) > len(regular_path):
            merged['best_path'] = strong_path
        else:
            merged_path = regular_path
            merged_path[:len(strong_path)] = strong_path
            merged['best_path'] = merged_path

    best_path = merged.get('best_path') or regular_path or strong_path
    merged['sop_node_path_str'] = ' -> '.join(best_path) if best_path else ''

    predicted_path = _predict_full_strong_path(strong_path, sop_data) if sop_data else list(strong_path)
    merged['strong_validation_expected_path'] = ' -> '.join(predicted_path) if predicted_path else ''

    node_keys = ['first_node', 'second_node', 'third_node', 'fourth_node',
                 'fifth_node', 'sixth_node', 'seventh_node', 'eighth_node', 'ninth_node']
    strong_levels_set = set(strong_levels.keys())

    for idx, key in enumerate(node_keys):
        level = idx + 1
        node_name = ''
        if level == 1:
            if strong_path:
                node_name = strong_path[0]
            elif regular_path:
                node_name = regular_path[0]
            merged[key] = node_name
            continue

        if level in strong_levels_set and level <= len(strong_path):
            node_name = strong_path[level - 1]
        else:
            level_info = merged['all_level_matches'].get(level)
            if level_info:
                path_list = list(level_info.get('path') or [])
                node_name = path_list[-1] if path_list else ''
        merged[key] = node_name

    tail_adjusted_success = False
    tail_sales_index = max(
        (info.get('sales_index', -1) for info in strong_levels.values()),
        default=-1,
    )
    remaining_pred_levels: Set[int] = set()
    if not original_in_branch and len(predicted_path) > len(strong_path):
        leftover_entries = [
            (lvl, copy.deepcopy(original_regular_levels.get(lvl, {})))
            for lvl in sorted(original_regular_levels.keys())
            if (
                    lvl not in strong_levels_set
                    and lvl >= len(strong_path) + 1
                    and original_regular_levels.get(lvl, {}).get('sales_index', -1) > tail_sales_index
            )
        ]

        remaining_pred_levels = set(range(len(strong_path) + 1, len(predicted_path) + 1))

        for lvl, info in leftover_entries:
            sales_msg = info.get('sales_message', '')
            if not sales_msg or not remaining_pred_levels:
                continue

            best_pred_level = None
            best_sim = 0.0
            best_script = ''
            for pred_level in sorted(remaining_pred_levels):
                node_name = predicted_path[pred_level - 1]
                node_data = _get_nested_value(sop_data, predicted_path[:pred_level]) if sop_data else None
                scripts = []
                if isinstance(node_data, dict):
                    for meta in ("参考话术", "相似话术"):
                        meta_val = node_data.get(meta, [])
                        if isinstance(meta_val, list):
                            scripts.extend(meta_val)
                        elif isinstance(meta_val, str):
                            scripts.append(meta_val)
                if not scripts:
                    continue
                for script in scripts:
                    sim = _cached_similarity(sales_msg or '', script)
                    if sim > best_sim:
                        best_sim = sim
                        best_pred_level = pred_level
                        best_script = script

            if best_pred_level is not None and best_sim >= 0.6:
                remaining_pred_levels.remove(best_pred_level)
                merged['all_level_matches'].pop(lvl, None)
                info['path'] = list(predicted_path[:best_pred_level])
                if best_script:
                    info['matched_reference'] = best_script
                info['similarity'] = max(best_sim, info.get('similarity', 0.0))
                info['match_type'] = info.get('match_type', '强校验补齐')
                merged['all_level_matches'][best_pred_level] = info
                merged[node_keys[best_pred_level - 1]] = predicted_path[best_pred_level - 1]
                tail_adjusted_success = True
    else:
        tail_adjusted_success = False
        remaining_pred_levels = set()

    if not original_in_branch and remaining_pred_levels and sales_entries:
        used_keys = {
            (info.get('sales_message'), info.get('sales_time'))
            for info in merged['all_level_matches'].values()
        }
        for pred_level in sorted(list(remaining_pred_levels)):
            node_name = predicted_path[pred_level - 1] if pred_level - 1 < len(predicted_path) else ''
            if not node_name:
                continue
            node_data = _get_nested_value(sop_data, predicted_path[:pred_level]) if sop_data else None
            scripts = []
            if isinstance(node_data, dict):
                for meta in ("参考话术", "相似话术"):
                    meta_val = node_data.get(meta, [])
                    if isinstance(meta_val, list):
                        scripts.extend(meta_val)
                    elif isinstance(meta_val, str):
                        scripts.append(meta_val)
            if not scripts:
                continue

            best_entry = None
            best_sim = 0.0
            best_script = ''
            for entry in sales_entries:
                entry_idx = entry.get('index', entry.get('sales_index', -1))
                if entry_idx <= tail_sales_index:
                    continue
                key = (entry.get('sales_message'), entry.get('sales_time'))
                if key in used_keys:
                    continue
                sales_msg = entry.get('sales_message')
                if not sales_msg:
                    continue
                for script in scripts:
                    sim = _cached_similarity(sales_msg or '', script)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry
                        best_script = script

            if best_entry and best_sim >= 0.55:
                key = (best_entry.get('sales_message'), best_entry.get('sales_time'))
                used_keys.add(key)
                info = {
                    'sales_message': best_entry.get('sales_message', ''),
                    'sales_time': best_entry.get('sales_time'),
                    'sales_index': entry_idx,
                    'similarity': best_sim,
                    'matched_reference': best_script,
                    'match_type': '强校验补齐',
                    'path': list(predicted_path[:pred_level])
                }
                merged['all_level_matches'][pred_level] = info
                merged[node_keys[pred_level - 1]] = node_name
                tail_adjusted_success = True

    for level, label in STRONG_VALIDATION_NODE_LABELS.items():
        if level in strong_levels_set and level <= len(strong_path):
            merged[label] = strong_path[level - 1]
        else:
            merged[label] = ''

    merged['matched_levels'] = sorted(merged['all_level_matches'].keys())

    if not original_in_branch:
        merged['strong_validation_tail_adjusted'] = '成功' if tail_adjusted_success else '失败'
    else:
        merged['strong_validation_tail_adjusted'] = ''

    max_level = 1
    if best_path:
        max_level = max(max_level, len(best_path))
    if merged['all_level_matches']:
        max_level = max(max_level, max(merged['all_level_matches'].keys()))
    merged['missing_levels'] = [lvl for lvl in range(2, max_level + 1) if lvl not in merged['all_level_matches']]

    actual_best_path: List[str] = list(best_path)
    anchor_path = list(strong_path) if strong_path else list(best_path)
    aligned_path = list(anchor_path)
    aligned_levels: Dict[int, Dict[str, Any]] = {}
    if merged['all_level_matches']:
        for level in sorted(merged['all_level_matches'].keys()):
            info = merged['all_level_matches'][level]
            path_list = list(info.get('path') or [])
            if not path_list:
                continue
            if anchor_path:
                common_len = min(len(anchor_path), len(path_list))
                if common_len and path_list[:common_len] != anchor_path[:common_len]:
                    continue
            if aligned_path:
                if len(path_list) < len(aligned_path):
                    continue
                if aligned_path and path_list[:len(aligned_path)] != aligned_path:
                    continue
                aligned_path = path_list
            else:
                aligned_path = path_list
            aligned_levels[level] = info
        selected_levels = aligned_levels if aligned_levels else merged['all_level_matches']
        deepest_level = max(selected_levels.keys())
        deepest_info = selected_levels[deepest_level]
        merged['sales_message'] = deepest_info.get('sales_message', '')
        merged['sales_time'] = deepest_info.get('sales_time')
        merged['similarity'] = deepest_info.get('similarity', 0.0)
        merged['matched_reference'] = deepest_info.get('matched_reference', '')
        merged['match_type'] = deepest_info.get('match_type', '')
    if aligned_levels:
        merged['all_level_matches'] = aligned_levels
        merged['matched_levels'] = sorted(aligned_levels.keys())
    if aligned_path:
        actual_best_path = aligned_path

    merged['expected_utterance'] = strong_match.get('expected_utterance', merged.get('expected_utterance', ''))
    merged['next_action'] = strong_match.get('next_action', merged.get('next_action', ''))
    merged['match_method'] = strong_match.get('match_method', merged.get('match_method', '纯改进版匹配'))

    merged['strong_validation_used'] = True
    for key, value in strong_match.items():
        if key.startswith('strong_validation'):
            merged[key] = value

    in_branch = _tail_in_branch(merged['all_level_matches'], merged.get('best_path'))
    merged['strong_validation_tail_in_branch'] = in_branch

    if not actual_best_path:
        actual_best_path = merged.get('best_path') or best_path or []

    if actual_best_path:
        merged['best_path'] = list(actual_best_path)

    for idx, key in enumerate(node_keys):
        level = idx + 1
        node_val = actual_best_path[idx] if actual_best_path and idx < len(actual_best_path) else ''
        merged[key] = node_val

    merged['actual_best_path'] = ' -> '.join(actual_best_path) if actual_best_path else ''
    merged['sop_node_path_str'] = merged['actual_best_path']

    metadata = _resolve_expected_fields(
        sop_data,
        actual_best_path,
        last_customer_msg,
        predicted_path=predicted_path
    )
    merged['expected_utterance'] = metadata['expected_utterance']
    merged['next_action'] = metadata['next_action']
    merged['core_info'] = metadata['core_info']

    return merged


def _run_strong_validation(
        sales_entries: List[Dict[str, Any]],
        sop_data: Dict[str, Any],
        last_customer_msg: str,
        default_threshold: float,
        erci_threshold: float,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    对销售话术进行强校验，以最高相似度锁定二级节点并沿分支定位子节点。
    二次跟进节点命中时优先返回。
    """
    info: Dict[str, Any] = {
        "default_threshold": default_threshold,
        "erci_threshold": erci_threshold,
        "triggered": False,
        "best_level2_similarity": 0.0,
        "best_level2_name": "",
        "best_level2_path": [],
        "best_level2_threshold": 0.0,
        "best_level2_is_erci": False,
        "preferred_node": "",
        "reason": ""
    }
    if not sales_entries:
        info["reason"] = "强校验未找到销售侧话术"
        return None, info

    all_candidates: List[Dict[str, Any]] = []
    for entry_idx, entry in enumerate(sales_entries):
        matches = entry.get("matches_by_path") or {}
        if not matches:
            continue
        for path_tuple, _ in matches.items():
            if len(path_tuple) < 2:
                continue
            chain = _build_chain_for_path(path_tuple, sales_entries, entry_idx)
            if not chain or 2 not in chain:
                continue
            level2_info = chain.get(2) or {}
            level2_similarity = level2_info.get("similarity", 0.0)
            expected_levels = max(len(path_tuple) - 1, 1)
            coverage = len(chain)
            coverage_ratio = coverage / expected_levels
            min_chain_similarity = (
                min(info_item.get("similarity", 0.0) for info_item in chain.values())
                if chain else 0.0
            )
            max_chain_similarity = (
                max(info_item.get("similarity", 0.0) for info_item in chain.values())
                if chain else 0.0
            )
            level2_sales_index = level2_info.get("sales_index")
            candidate = {
                "path": tuple(path_tuple),
                "chain": chain,
                "level2_name": path_tuple[1],
                "level1_name": path_tuple[0] if path_tuple else "",
                "similarity": level2_similarity,
                "entry_idx": entry_idx,
                "level2_sales_index": level2_sales_index if level2_sales_index is not None else entry.get("index"),
                "max_chain_similarity": max_chain_similarity,
                "min_chain_similarity": min_chain_similarity,
                "coverage": coverage,
                "expected_levels": expected_levels,
                "coverage_ratio": coverage_ratio,
                "level2_reference": level2_info.get("matched_reference", ""),
                "level2_sales_message": level2_info.get("sales_message", ""),
                "level2_sales_time": level2_info.get("sales_time"),
                "is_erci": "二次跟进" in path_tuple[1],
            }
            candidate["threshold"] = (
                erci_threshold if candidate["is_erci"] else default_threshold
            )
            all_candidates.append(candidate)

    if not all_candidates:
        info["reason"] = "强校验未找到有效的二级节点匹配"
        return None, info

    def _candidate_sort_key(cand: Dict[str, Any]) -> Tuple[float, int, float, float, float, int]:
        level2_idx = cand["level2_sales_index"] if cand["level2_sales_index"] is not None else -1
        return (
            cand["coverage_ratio"],
            cand["coverage"],
            level2_idx,
            cand["min_chain_similarity"],
            cand["similarity"],
            cand["max_chain_similarity"],
            -cand["expected_levels"],
            cand["entry_idx"],
        )

    best_overall = max(all_candidates, key=_candidate_sort_key, default=None)
    if best_overall:
        info["best_level2_similarity"] = best_overall["similarity"]
        info["best_level2_name"] = best_overall["level2_name"]
        info["best_level2_path"] = list(best_overall["path"])
        info["best_level2_threshold"] = best_overall["threshold"]
        info["best_level2_is_erci"] = best_overall["is_erci"]
        info["best_level2_coverage_ratio"] = best_overall["coverage_ratio"]
        info["best_level2_coverage"] = best_overall["coverage"]
        info["best_level2_sales_index"] = best_overall.get("level2_sales_index")

    eligible_candidates = [
        cand for cand in all_candidates if cand["similarity"] >= cand["threshold"]
    ]
    if not eligible_candidates:
        if best_overall:
            info["reason"] = (
                f"强校验最高相似度 {best_overall['similarity']:.2f} "
                f"低于所需阈值 {best_overall['threshold']:.2f}（{best_overall['level2_name']}）"
            )
        else:
            info["reason"] = (
                f"强校验未达到阈值（默认 {default_threshold:.2f} / 二次跟进 {erci_threshold:.2f}）"
            )
        return None, info

    erci_candidates = [
        cand for cand in eligible_candidates if cand["is_erci"]
    ]
    priority_candidates = erci_candidates if erci_candidates else eligible_candidates
    best_candidate = max(priority_candidates, key=_candidate_sort_key)

    info["triggered"] = True
    info["preferred_node"] = (
        "二次跟进" if erci_candidates else best_candidate["level2_name"]
    )
    info["best_level2_similarity"] = best_candidate["similarity"]
    info["best_level2_name"] = best_candidate["level2_name"]
    info["best_level2_path"] = list(best_candidate["path"])
    info["best_level2_threshold"] = best_candidate["threshold"]
    info["best_level2_is_erci"] = best_candidate["is_erci"]
    info["best_level2_coverage_ratio"] = best_candidate["coverage_ratio"]
    info["best_level2_coverage"] = best_candidate["coverage"]
    info["best_level2_sales_index"] = best_candidate.get("level2_sales_index")
    info["reason"] = (
        f"强校验命中: {' -> '.join(best_candidate['path'])} "
        f"(相似度 {best_candidate['similarity']:.2f})"
    )

    final_match = _assemble_match_from_chain(
        best_candidate["path"],
        best_candidate["chain"],
        sop_data,
        last_customer_msg,
    )
    level2_chain = best_candidate["chain"].get(2, {})

    final_match["strong_validation_used"] = True
    final_match["strong_validation_threshold"] = best_candidate["threshold"]
    final_match["strong_validation_path"] = " -> ".join(best_candidate["path"])
    final_match["strong_validation_level2"] = best_candidate["level2_name"]
    final_match["strong_validation_level2_similarity"] = best_candidate["similarity"]
    final_match["strong_validation_level2_reference"] = best_candidate.get(
        "level2_reference",
        "",
    )
    final_match["strong_validation_level2_sales_message"] = best_candidate.get(
        "level2_sales_message",
        level2_chain.get("sales_message", ""),
    )
    final_match["strong_validation_level2_sales_time"] = best_candidate.get(
        "level2_sales_time",
        level2_chain.get("sales_time"),
    )
    final_match["strong_validation_level2_sales_index"] = best_candidate.get(
        "level2_sales_index",
        level2_chain.get("sales_index"),
    )
    final_match["strong_validation_is_erci"] = best_candidate["is_erci"]
    final_match["strong_validation_min_chain_similarity"] = best_candidate[
        "min_chain_similarity"
    ]
    final_match["strong_validation_max_chain_similarity"] = best_candidate[
        "max_chain_similarity"
    ]
    final_match["strong_validation_coverage_ratio"] = best_candidate["coverage_ratio"]
    final_match["strong_validation_coverage"] = best_candidate["coverage"]
    final_match["strong_validation_level2_match"] = {
        key: value for key, value in level2_chain.items() if key != "entry_index"
    }
    final_match["match_method"] = "强校验匹配"
    final_match["confidence"] = best_candidate["similarity"]

    for level, node_name in enumerate(final_match.get("best_path", []), start=1):
        label = STRONG_VALIDATION_NODE_LABELS.get(level)
        if label:
            final_match[label] = node_name

    return final_match, info


def _max_similarity_to_scripts(sales_messages, scripts):
    """计算所有销售话术与节点脚本的最大相似度"""
    if not scripts:
        return 0.0
    best_sim = 0.0
    for msg in sales_messages:
        if not msg:
            continue
        for script in scripts:
            sim = _cached_similarity(msg or '', script)
            if sim > best_sim:
                best_sim = sim
    return best_sim


def _describe_unmatched_levels(node_path, aggregated_matches, sop_data, similarity_threshold, sales_messages):
    """
    给出未命中的层级原因说明
    """
    reasons = []
    for idx, node_name in enumerate(node_path):
        level = idx + 1
        if level == 1:
            # 一级节点无需匹配，直接跳过
            continue

        if level in aggregated_matches:
            continue

        sub_path = node_path[:level]
        sop_node = _get_nested_value(sop_data, sub_path)
        if not isinstance(sop_node, dict):
            reasons.append(f"SOP{level}级节点“{node_name}”在逻辑树中缺失，无法匹配")
            continue

        scripts = []
        for key in ("参考话术", "相似话术"):
            scripts.extend([s for s in sop_node.get(key, []) if isinstance(s, str) and s.strip()])

        if not scripts:
            reasons.append(f"SOP{level}级节点“{node_name}”缺少参考/相似话术配置，建议补充 logictree.json")
        else:
            best_sim = _max_similarity_to_scripts(sales_messages, scripts)
            if best_sim == 0.0:
                reasons.append(f"SOP{level}级节点“{node_name}”未在对话中找到对应的销售话术")
            else:
                reasons.append(
                    f"SOP{level}级节点“{node_name}”存在话术，但对话中的最高相似度仅 {best_sim:.2f}，低于阈值 {similarity_threshold}"
                )
    return reasons


def _build_chain_for_path(path_tuple: Tuple[str, ...], entries: List[Dict[str, Any]], upto_index: int) -> Dict[
    int, Dict[str, Any]]:
    """
    根据候选路径回溯找到每一层匹配的销售话术（一级节点除外）
    """
    chain: Dict[int, Dict[str, Any]] = {}
    max_level = len(path_tuple)
    if max_level <= 1:
        return chain

    for level in range(2, max_level + 1):
        prefix = tuple(path_tuple[:level])
        best_match = None
        for entry in reversed(entries[:upto_index + 1]):
            match_info = entry['matches_by_path'].get(prefix)
            if not match_info:
                continue
            candidate = {
                'similarity': match_info['similarity'],
                'matched_reference': match_info['matched_reference'],
                'match_type': match_info['match_type'],
                'sales_message': entry['sales_message'],
                'sales_time': entry['sales_time'],
                'entry_index': entry['index'],
                'sales_index': entry['index'],
                'path': list(prefix)
            }
            if (best_match is None
                    or candidate['similarity'] > best_match['similarity']
                    or (candidate['similarity'] == best_match['similarity']
                        and candidate['entry_index'] > best_match['entry_index'])):
                best_match = candidate
        if best_match:
            chain[level] = best_match
    return chain


def _assemble_match_from_chain(path_tuple: Tuple[str, ...],
                               chain: Dict[int, Dict[str, Any]],
                               sop_data: Dict,
                               last_customer_msg: str) -> Dict[str, Any]:
    """根据匹配链构造最终的匹配结果"""
    node_path = list(path_tuple)
    aggregated_public = {
        level: {k: v for k, v in info.items() if k != 'entry_index'}
        for level, info in chain.items()
    }

    node_keys = ['first_node', 'second_node', 'third_node', 'fourth_node',
                 'fifth_node', 'sixth_node', 'seventh_node', 'eighth_node', 'ninth_node']

    final_match = {
        'best_path': node_path,
        'all_level_matches': aggregated_public,
        'matched_levels': sorted(aggregated_public.keys()),
        'missing_levels': [lvl for lvl in range(2, len(node_path) + 1) if lvl not in aggregated_public],
        'sop_node_path_str': ' -> '.join(node_path) if node_path else ''
    }

    for idx, node_name in enumerate(node_path):
        if idx >= len(node_keys):
            break
        level = idx + 1
        if level == 1 or level in aggregated_public:
            final_match[node_keys[idx]] = node_name
        else:
            final_match[node_keys[idx]] = ''

    if chain:
        deepest_level = max(chain.keys())
        deepest_info = chain[deepest_level]
        final_match['sales_message'] = deepest_info.get('sales_message', '')
        final_match['sales_time'] = deepest_info.get('sales_time')
        final_match['similarity'] = deepest_info.get('similarity', 0.0)
        final_match['matched_reference'] = deepest_info.get('matched_reference', '')
        final_match['match_type'] = deepest_info.get('match_type', '')
    else:
        final_match['sales_message'] = ''
        final_match['sales_time'] = None
        final_match['similarity'] = 0.0
        final_match['matched_reference'] = ''
        final_match['match_type'] = ''

    # 计算预期话术与下一步动作
    metadata = _resolve_expected_fields(sop_data, node_path, last_customer_msg)
    final_match['expected_utterance'] = metadata['expected_utterance']
    final_match['next_action'] = metadata['next_action']
    final_match['core_info'] = metadata['core_info']

    return final_match


def find_best_sop_match_improved(
        conversations,
        sop_data,
        similarity_threshold=0.90,
        strong_validation_threshold=0.95,
        erci_strong_threshold=0.85,
):
    """
    纯改进版逻辑：结合原版深层级匹配 + 改进版"从客户消息往上查找"逻辑

    参数:
        conversations: 对话列表
        sop_data: SOP逻辑树数据
        similarity_threshold: 相似度阈值
        strong_validation_threshold: 强校验定位二级节点的默认相似度阈值
        erci_strong_threshold: 强校验命中“二次跟进”节点的相似度阈值

    返回:
        tuple[dict|None, dict]: (最佳匹配结果, 调试信息)
    """
    debug_info = {
        "reason": "",
        "max_similarity": 0.0,
        "max_similarity_path": [],
        "max_similarity_reference": "",
        "candidate_sales_message": "",
        "checked_sales_messages": 0,
        "sales_messages": []
    }

    if not conversations:
        debug_info["reason"] = "无有效对话"
        return None, debug_info

    # 找最后一条客户消息位置
    last_customer_idx = None
    for idx in range(len(conversations) - 1, -1, -1):
        if conversations[idx].get('role') == '客户':
            last_customer_idx = idx
            break
    if last_customer_idx is None:
        debug_info["reason"] = "未找到客户消息"
        return None, debug_info

    last_customer_msg = conversations[last_customer_idx].get('content', '')

    sales_entries = []
    last_sales_idx = None
    single_best = None
    last_matched_idx = None

    sop_cache = _preprocess_sop_tree(sop_data)

    for idx, conv in enumerate(conversations[:last_customer_idx + 1]):
        if conv.get('role') != '销售':
            continue
        sales_message = conv.get('content', '')
        debug_info["checked_sales_messages"] += 1
        debug_info["candidate_sales_message"] = sales_message
        last_sales_idx = idx if last_sales_idx is None or idx > last_sales_idx else last_sales_idx

        matches_by_path = _collect_matches_by_path(
            sales_message,
            sop_data,
            similarity_threshold,
            sop_cache=sop_cache
        )
        entry = {
            "index": idx,
            "sales_message": sales_message,
            "sales_time": conv.get('time'),
            "matches_by_path": matches_by_path
        }
        sales_entries.append(entry)

        if matches_by_path:
            last_matched_idx = idx if last_matched_idx is None or idx > last_matched_idx else last_matched_idx
            local_best_path, local_best = max(matches_by_path.items(), key=lambda item: item[1]['similarity'])
            if local_best['similarity'] > debug_info['max_similarity']:
                debug_info['max_similarity'] = local_best['similarity']
                debug_info['max_similarity_path'] = list(local_best_path)
                debug_info['max_similarity_reference'] = local_best['matched_reference']
                debug_info['candidate_sales_message'] = sales_message
            contains_last = int(idx == (last_matched_idx or last_sales_idx))
            best_tuple = (contains_last, local_best['similarity'])
            if single_best is None or best_tuple > single_best[0]:
                single_best = (best_tuple, local_best_path, idx, local_best, entry)
        else:
            candidate_any = find_best_similarity_candidate(sales_message, sop_data)
            if candidate_any and candidate_any['similarity'] > debug_info['max_similarity']:
                debug_info['max_similarity'] = candidate_any['similarity']
                debug_info['max_similarity_path'] = candidate_any['path']
                debug_info['max_similarity_reference'] = candidate_any['matched_reference']

    debug_info["sales_messages"] = [entry["sales_message"] for entry in sales_entries]
    if last_matched_idx is None:
        last_matched_idx = last_sales_idx

    strong_match: Optional[Dict[str, Any]] = None
    strong_info: Dict[str, Any] = {}
    if sales_entries:
        strong_match, strong_info = _run_strong_validation(
            sales_entries=sales_entries,
            sop_data=sop_data,
            last_customer_msg=last_customer_msg,
            default_threshold=strong_validation_threshold,
            erci_threshold=erci_strong_threshold,
        )

    debug_info["strong_validation_default_threshold"] = strong_validation_threshold
    debug_info["strong_validation_erci_threshold"] = erci_strong_threshold
    debug_info["strong_validation_triggered"] = strong_info.get("triggered", False)
    debug_info["strong_validation_best_level2_similarity"] = strong_info.get(
        "best_level2_similarity",
        0.0,
    )
    debug_info["strong_validation_best_level2"] = strong_info.get(
        "best_level2_name",
        "",
    )
    debug_info["strong_validation_best_level2_path"] = strong_info.get(
        "best_level2_path",
        [],
    )
    if strong_info.get("best_level2_threshold") is not None:
        debug_info["strong_validation_best_level2_threshold"] = strong_info.get(
            "best_level2_threshold",
            0.0,
        )
    if strong_info.get("best_level2_is_erci") is not None:
        debug_info["strong_validation_best_level2_is_erci"] = strong_info.get(
            "best_level2_is_erci",
            False,
        )
    if strong_info.get("best_level2_coverage_ratio") is not None:
        debug_info["strong_validation_best_level2_coverage_ratio"] = strong_info.get(
            "best_level2_coverage_ratio",
            0.0,
        )
    if strong_info.get("best_level2_coverage") is not None:
        debug_info["strong_validation_best_level2_coverage"] = strong_info.get(
            "best_level2_coverage",
            0,
        )
    if strong_info.get("best_level2_sales_index") is not None:
        debug_info["strong_validation_best_level2_sales_index"] = strong_info.get(
            "best_level2_sales_index"
        )
    if strong_info.get("preferred_node"):
        debug_info["strong_validation_preferred_node"] = strong_info["preferred_node"]
    if strong_info.get("reason"):
        debug_info["strong_validation_reason"] = strong_info["reason"]

    regular_match: Optional[Dict[str, Any]] = None
    regular_info: Dict[str, Any] = {}
    regular_tail_meta: Dict[str, Any] = {}

    # 按时间倒序，命中挖需节点即返回
    for entry_idx in range(len(sales_entries) - 1, -1, -1):
        entry = sales_entries[entry_idx]
        matches = entry['matches_by_path']
        if not matches:
            continue
        best_entry_result = None
        for path_tuple, match_info in matches.items():
            if len(path_tuple) < 2:
                continue
            chain = _build_chain_for_path(path_tuple, sales_entries, entry_idx)
            if not chain:
                continue
            expected_levels = max(len(path_tuple) - 1, 1)
            coverage = len(chain)
            coverage_ratio = coverage / expected_levels
            min_sim = min(info['similarity'] for info in chain.values())
            max_sim = max(info['similarity'] for info in chain.values())
            score = (coverage_ratio, coverage, min_sim, max_sim, -expected_levels)
            if best_entry_result is None or score > best_entry_result[0]:
                best_entry_result = (score, path_tuple, chain)

        if best_entry_result:
            _, path_tuple, chain = best_entry_result
            final_match = _assemble_match_from_chain(path_tuple, chain, sop_data, last_customer_msg)
            final_match.setdefault('match_method', '纯改进版匹配')
            regular_match = final_match
            regular_tail_meta = _attach_regular_tail_metadata(regular_match, sop_data, last_customer_msg)
            regular_info = {
                "reason": "",
                "matched_sales_message": final_match.get('sales_message', ''),
                "aggregated_levels": final_match.get('all_level_matches', {}),
                "depth": len(path_tuple),
            }
            break

    fallback_match: Optional[Dict[str, Any]] = None
    fallback_debug: Dict[str, Any] = {}
    fallback_tail_meta: Dict[str, Any] = {'expected_utterance': '', 'next_action': '', 'core_info': ''}

    # 仅在必要时触发原版全量匹配，避免重复的全量遍历开销
    if regular_match is None:
        fallback_match, fallback_debug = fallback_find_best(conversations, sop_data, similarity_threshold)
        if fallback_match:
            fallback_match.setdefault('match_method', '原版全量匹配')
            fallback_tail_meta = _attach_regular_tail_metadata(fallback_match, sop_data, last_customer_msg)
    if strong_match:
        base_match = regular_match or fallback_match
        tail_meta_for_merge = regular_tail_meta if regular_match else fallback_tail_meta
        merged_match = _merge_strong_with_regular(
            strong_match,
            base_match,
            sop_data,
            sales_entries,
            tail_meta_for_merge,
            last_customer_msg,
        )
        debug_info.update({
            "reason": "",
            "matched_sales_message": merged_match.get('sales_message', ''),
            "aggregated_levels": merged_match.get('all_level_matches', {}),
            "depth": len(merged_match.get('best_path', [])),
            "strong_validation_base_from": "regular" if regular_match else ("fallback" if fallback_match else "strong"),
        })
        debug_info["strong_validation_tail_in_branch"] = merged_match.get('strong_validation_tail_in_branch')
        debug_info["strong_validation_tail_adjusted"] = merged_match.get('strong_validation_tail_adjusted')
        if regular_info:
            debug_info.update(regular_info)
        return merged_match, debug_info
    if regular_match:
        debug_info.update(regular_info)
        return regular_match, debug_info
    if fallback_match:
        fallback_tail_meta = _attach_regular_tail_metadata(fallback_match, sop_data, last_customer_msg)
        debug_info.update(fallback_debug)
        return fallback_match, debug_info
    return fallback_match, fallback_debug


def analyze_pure_improved_logic(
        sales_corpus_xlsx,
        sop_logic_tree_path,
        similarity_threshold=0.90,
        batch_size=100,
        sentencewise_options: Optional[Dict[str, Any]] = None,
        num_workers: Optional[int] = None,
):
    """
    纯改进版逻辑分析函数

    步骤：
    1. 对每条对话，遍历所有SOP节点
    2. 用改进版逻辑查找每个节点是否能找到对应话术
    3. 选择相似度最高的匹配结果

    参数:
        sales_corpus_xlsx: 销售对话汇总Excel文件路径
        sop_logic_tree_path: SOP逻辑树JSON文件路径
        similarity_threshold: 相似度阈值
        batch_size: 批次大小

    返回:
        包含SOP标签的记录列表
    """
    try:
        # 读取销售对话汇总文件
        df = pd.read_excel(sales_corpus_xlsx, engine='openpyxl')
        print(f"成功读取销售对话汇总文件，共 {len(df)} 条记录")

        # 检查是否有对话列
        conversation_column = None
        for col in ['对话历史', '最终传参上下文', '历史对话']:
            if col in df.columns:
                conversation_column = col
                break

        if not conversation_column:
            print("错误：未找到对话历史列")
            return []

        print(f"使用对话列：{conversation_column}")

        # 读取SOP逻辑树
        sop_data = read_json_file(sop_logic_tree_path)
        if not sop_data:
            print("无法读取SOP逻辑树文件")
            return []

        print(f"成功读取SOP逻辑树，共 {len(sop_data)} 个一级节点")

        # 为每条记录分析并打标签
        labeled_records = []
        batch_count = 0

        # 生成临时文件路径
        temp_dir = os.path.dirname(sales_corpus_xlsx)
        temp_file_base = os.path.splitext(os.path.basename(sales_corpus_xlsx))[0]
        temp_file_path = os.path.join(temp_dir, f"{temp_file_base}_pure_improved_temp_progress.xlsx")

        print(f"每处理 {batch_size} 行将保存进度到：{temp_file_path}")
        print(f"使用纯改进算法：遍历所有SOP节点，用改进版逻辑查找话术")

        sentencewise_options = sentencewise_options or {}
        sentencewise_enabled = bool(sentencewise_matcher) and sentencewise_options.get("enabled", True)
        sentencewise_params = {
            "similarity": sentencewise_options.get("similarity", 0.80),
            "seq_ratio": sentencewise_options.get("seq_ratio", 0.72),
            "level_thresholds": sentencewise_options.get("level_thresholds", SENTENCEWISE_DEFAULT_THRESHOLDS),
            "min_level_matches": sentencewise_options.get("min_level_matches", 1),
            "trigger_on_partial": sentencewise_options.get("trigger_on_partial", True),
        }
        default_threshold = sentencewise_options.get(
            "strong_validation_default_threshold",
            sentencewise_options.get("strong_validation_threshold", 0.95),
        )
        erci_threshold = sentencewise_options.get(
            "strong_validation_erci_threshold",
            0.85,
        )
        sentencewise_params["strong_validation_default_threshold"] = default_threshold
        sentencewise_params["strong_validation_erci_threshold"] = erci_threshold
        sentencewise_params["override_level2_similarity"] = sentencewise_options.get(
            "override_level2_similarity",
            0.0
        )

        requested_workers = num_workers if num_workers is not None else 1
        if sentencewise_options.get("parallel_workers") is not None:
            requested_workers = sentencewise_options.get("parallel_workers")
        requested_workers = int(requested_workers) if requested_workers else 1
        worker_count = max(1, requested_workers)
        max_cpu = os.cpu_count() or 1
        worker_count = min(worker_count, max_cpu)

        rows = list(df.iterrows())
        improved_matches = 0  # 改进版匹配计数
        no_matches = 0  # 无匹配计数

        parallel_results: Dict[int, Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[str]]] = {}
        if worker_count > 1:
            tasks = []
            for idx, (_, row) in enumerate(rows):
                history_candidate = str(row.get(conversation_column, '')).strip()
                if history_candidate and history_candidate != 'nan':
                    tasks.append((idx, history_candidate))

            if tasks:
                print(f"并行模式开启：使用 {worker_count} 个进程处理 {len(tasks)} 条对话")
                try:
                    with ProcessPoolExecutor(
                            max_workers=worker_count,
                            initializer=_init_parallel_worker,
                            initargs=(sop_data, similarity_threshold,
                                      sentencewise_params["strong_validation_default_threshold"],
                                      sentencewise_params["strong_validation_erci_threshold"])
                    ) as executor:
                        future_to_idx = {
                            executor.submit(_parallel_match_worker, task): task[0]
                            for task in tasks
                        }
                        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="并行匹配",
                                           leave=False):
                            try:
                                idx_result, match_result, debug_result, error_flag = future.result()
                                parallel_results[idx_result] = (match_result, debug_result, error_flag)
                            except Exception as exc:
                                idx_result = future_to_idx[future]
                                parallel_results[idx_result] = (None, {"reason": f"子进程异常: {exc}"}, repr(exc))
                except Exception as exc:
                    print(f"警告：并行执行失败，将退回串行模式。原因：{exc}")
                    parallel_results.clear()
                    worker_count = 1
            else:
                worker_count = 1

        for idx, (_, row) in enumerate(tqdm(rows, total=len(rows), desc="分析对话记录")):
            history_str = str(row.get(conversation_column, '')).strip()
            reasons = []
            best_sop_match = None
            match_debug = {}
            last_customer_msg = ''
            sentencewise_match = None
            sentencewise_debug: Dict[str, Any] = {}
            sentencewise_triggered = False
            conversations: List[Dict[str, Any]] = []

            if not history_str or history_str == 'nan':
                reasons.append("对话历史为空或缺失")
                no_matches += 1
            else:
                try:
                    conversations = parse_conversation_history(history_str)
                except Exception as exc:
                    reasons.append(f"对话历史解析失败: {exc}")
                    no_matches += 1
                    conversations = []

                if conversations:
                    # 记录最后一个客户消息
                    for conv in reversed(conversations):
                        if conv.get('role') == '客户':
                            last_customer_msg = conv.get('content', '')
                            break
                    if not last_customer_msg:
                        reasons.append("没有找到客户消息，无法定位SOP节点")

                    if conversations[-1]['role'] != '客户':
                        reasons.append("最后一条消息不是客户消息，建议确认对话截断位置")

                    precomputed_entry = parallel_results.get(idx) if worker_count > 1 else None
                    worker_error = None
                    if precomputed_entry:
                        best_sop_match, match_debug, worker_error = precomputed_entry
                        if worker_error:
                            reasons.append(f"并行匹配失败: {worker_error}")
                            best_sop_match = None
                            match_debug = match_debug or {}
                    if best_sop_match is None:
                        best_sop_match, match_debug = find_best_sop_match_improved(
                            conversations,
                            sop_data,
                            similarity_threshold,
                            sentencewise_params["strong_validation_default_threshold"],
                            sentencewise_params["strong_validation_erci_threshold"],
                        )

                    strong_reason = match_debug.get("strong_validation_reason")
                    if strong_reason:
                        reasons.append(strong_reason)

                    if sentencewise_enabled:
                        need_sentencewise = False
                        if not best_sop_match:
                            need_sentencewise = True
                        elif sentencewise_params["trigger_on_partial"] and best_sop_match.get('missing_levels'):
                            need_sentencewise = True

                        if need_sentencewise:
                            sentencewise_triggered = True
                            sentencewise_match, sentencewise_debug = sentencewise_matcher(
                                conversations=conversations,
                                sop_tree=sop_data,
                                similarity_threshold=sentencewise_params["similarity"],
                                seq_ratio_threshold=sentencewise_params["seq_ratio"],
                                level_thresholds=sentencewise_params["level_thresholds"],
                                min_level_matches=sentencewise_params["min_level_matches"],
                            )

                    if best_sop_match:
                        node_path = best_sop_match.get('best_path') or [
                            best_sop_match[key] for key in ['first_node', 'second_node', 'third_node', 'fourth_node',
                                                            'fifth_node', 'sixth_node', 'seventh_node', 'eighth_node',
                                                            'ninth_node']
                            if best_sop_match.get(key)
                        ]
                        level_matches = best_sop_match.get('all_level_matches', {})
                        sales_texts = match_debug.get('sales_messages', [])
                        level_reasons = _describe_unmatched_levels(node_path, level_matches, sop_data,
                                                                   similarity_threshold, sales_texts)
                        reasons.extend(level_reasons)
                    else:
                        if match_debug.get("reason"):
                            reasons.append(match_debug["reason"])
                        if match_debug.get("max_similarity_path"):
                            reasons.append(
                                "最高相似度候选节点路径: " + " -> ".join(match_debug["max_similarity_path"])
                            )
                        if match_debug.get("max_similarity_reference"):
                            reasons.append(f"最高相似度参考话术: {match_debug['max_similarity_reference']}")
                        if match_debug.get("max_similarity"):
                            reasons.append(f"最高相似度: {match_debug['max_similarity']:.2f}")
            fallback_path = sentencewise_match.get('sop_node_path_str', '') if sentencewise_match else ''
            fallback_similarity = sentencewise_match.get('similarity', 0.0) if sentencewise_match else 0.0
            fallback_reference = sentencewise_match.get('matched_reference', '') if sentencewise_match else ''
            fallback_trace = sentencewise_match.get('sentence_trace', []) if sentencewise_match else []
            fallback_trace_json = json.dumps(fallback_trace, ensure_ascii=False) if fallback_trace else ''
            fallback_notes = []
            if sentencewise_triggered:
                if sentencewise_match:
                    fallback_notes.append("句级备用命中")
                if sentencewise_debug.get("reason"):
                    fallback_notes.append(sentencewise_debug["reason"])
                if sentencewise_debug.get("max_similarity_path"):
                    fallback_notes.append(
                        "句级备用最高相似度路径: " + " -> ".join(sentencewise_debug["max_similarity_path"])
                    )
                if sentencewise_debug.get("max_similarity_reference"):
                    fallback_notes.append(
                        f"句级备用最高相似度参考话术: {sentencewise_debug['max_similarity_reference']}")
                if sentencewise_debug.get("max_similarity"):
                    fallback_notes.append(f"句级备用最高相似度: {sentencewise_debug['max_similarity']:.2f}")
            fallback_remark = '; '.join(dict.fromkeys([note for note in fallback_notes if note]))

            primary_match = best_sop_match if best_sop_match else None
            primary_method = primary_match.get('match_method', "纯改进版匹配") if primary_match else "无匹配"

            selected_match = primary_match
            selected_method = primary_method

            fallback_level2_name = ''
            fallback_level2_sim = 0.0
            fallback_path_list = sentencewise_match.get('best_path') if sentencewise_match else None
            if sentencewise_match and fallback_path_list:
                if len(fallback_path_list) >= 2:
                    fallback_level2_name = fallback_path_list[1]
                fallback_levels = sentencewise_match.get('all_level_matches', {})
                if 2 in fallback_levels:
                    fallback_level2_sim = fallback_levels[2].get('similarity',
                                                                 sentencewise_match.get('similarity', 0.0))
                else:
                    fallback_level2_sim = sentencewise_match.get('similarity', 0.0)

            override_reason = None
            if sentencewise_triggered:
                if primary_match and primary_match.get('strong_validation_used'):
                    reasons.append("强校验命中，跳过句级备用覆盖")
                elif sentencewise_match and fallback_path:
                    primary_level2 = primary_match.get('second_node', '') if primary_match else ''
                    if fallback_level2_name and fallback_level2_sim >= sentencewise_params[
                        "override_level2_similarity"] and fallback_level2_name != primary_level2:
                        selected_match = _extend_sentencewise_branch(sentencewise_match, conversations, sop_data,
                                                                     sentencewise_params)
                        sentencewise_match = selected_match
                        selected_method = "句级备用覆盖" if primary_match else "句级备用匹配"
                        override_reason = f"句级备用覆盖: {fallback_path}"
                        reasons.append(override_reason)
                    elif primary_match and fallback_path:
                        reasons.append(f"句级备用建议路径: {fallback_path}")
                elif sentencewise_triggered and sentencewise_debug.get('reason'):
                    reasons.append(f"句级备用: {sentencewise_debug['reason']}")

            if sentencewise_match:
                fallback_path = sentencewise_match.get('sop_node_path_str', fallback_path)
                fallback_similarity = sentencewise_match.get('similarity', fallback_similarity)
                fallback_reference = sentencewise_match.get('matched_reference', fallback_reference)
                fallback_trace = sentencewise_match.get('sentence_trace', fallback_trace)
                fallback_trace_json = json.dumps(fallback_trace,
                                                 ensure_ascii=False) if fallback_trace else fallback_trace_json

            if selected_match and selected_method:
                selected_match['match_method'] = selected_method

            if selected_match:
                improved_matches += 1
            else:
                no_matches += 1
                selected_method = "无匹配"

            strong_flag = bool(selected_match and selected_match.get('strong_validation_used'))
            default_threshold_value = sentencewise_params["strong_validation_default_threshold"]
            erci_threshold_value = sentencewise_params["strong_validation_erci_threshold"]
            strong_is_erci = bool(selected_match and selected_match.get('strong_validation_is_erci'))
            strong_threshold_value = default_threshold_value
            if selected_match and selected_match.get('strong_validation_threshold') is not None:
                strong_threshold_value = selected_match.get('strong_validation_threshold', default_threshold_value)
            elif strong_is_erci:
                strong_threshold_value = erci_threshold_value
            strong_level2_similarity = selected_match.get('strong_validation_level2_similarity',
                                                          0.0) if selected_match else 0.0
            strong_path = selected_match.get('strong_validation_path', '') if selected_match else ''
            strong_reference = selected_match.get('strong_validation_level2_reference', '') if selected_match else ''
            strong_sales_message = selected_match.get('strong_validation_level2_sales_message',
                                                      '') if selected_match else ''
            strong_preferred_node = match_debug.get('strong_validation_preferred_node', '')
            strong_best_node = match_debug.get('strong_validation_best_level2', '')
            strong_best_similarity = match_debug.get('strong_validation_best_level2_similarity', 0.0)
            strong_best_path = " -> ".join(
                match_debug.get('strong_validation_best_level2_path', [])) if match_debug else ''
            strong_coverage_ratio = selected_match.get('strong_validation_coverage_ratio',
                                                       0.0) if selected_match else 0.0
            strong_coverage = selected_match.get('strong_validation_coverage', 0) if selected_match else 0
            strong_tail_in_branch = selected_match.get('strong_validation_tail_in_branch') if selected_match else None
            tail_adjust_status = selected_match.get('strong_validation_tail_adjusted', '') if selected_match else ''
            expected_value = selected_match.get('expected_utterance', '') if selected_match else ''
            if isinstance(expected_value, dict):
                expected_value = json.dumps(expected_value, ensure_ascii=False)
            elif isinstance(expected_value, list):
                expected_value = json.dumps(expected_value, ensure_ascii=False) if expected_value else ''
            core_value = selected_match.get('core_info', '') if selected_match else ''
            if isinstance(core_value, dict):
                core_value = json.dumps(core_value, ensure_ascii=False)
            elif isinstance(core_value, list):
                core_value = json.dumps(core_value, ensure_ascii=False) if core_value else ''
            regular_tail_expected = selected_match.get('regular_tail_expected', '') if selected_match else ''
            if isinstance(regular_tail_expected, dict):
                regular_tail_expected = json.dumps(regular_tail_expected, ensure_ascii=False)
            elif isinstance(regular_tail_expected, list):
                regular_tail_expected = json.dumps(regular_tail_expected,
                                                   ensure_ascii=False) if regular_tail_expected else ''
            regular_tail_core = selected_match.get('regular_tail_core', '') if selected_match else ''
            if isinstance(regular_tail_core, dict):
                regular_tail_core = json.dumps(regular_tail_core, ensure_ascii=False)
            elif isinstance(regular_tail_core, list):
                regular_tail_core = json.dumps(regular_tail_core, ensure_ascii=False) if regular_tail_core else ''

            labeled_record = {
                "最后客户消息": last_customer_msg,
                "匹配方法": selected_match.get('match_method', selected_method) if selected_match else selected_method,
                "最近销售消息": selected_match.get('sales_message', '') if selected_match else '',
                "销售消息时间": selected_match.get('sales_time', '') if selected_match else '',
                "SOP一级节点": selected_match.get('first_node', '') if selected_match else '',
                "SOP二级节点": selected_match.get('second_node', '') if selected_match else '',
                "SOP三级节点": selected_match.get('third_node', '') if selected_match else '',
                "SOP四级节点": selected_match.get('fourth_node', '') if selected_match else '',
                "SOP五级节点": selected_match.get('fifth_node', '') if selected_match else '',
                "SOP六级节点": selected_match.get('sixth_node', '') if selected_match else '',
                "SOP七级节点": selected_match.get('seventh_node', '') if selected_match else '',
                "SOP八级节点": selected_match.get('eighth_node', '') if selected_match else '',
                "SOP九级节点": selected_match.get('ninth_node', '') if selected_match else '',
                "匹配相似度": selected_match.get('similarity', 0.0) if selected_match else 0.0,
                "匹配的参考话术": selected_match.get('matched_reference', '') if selected_match else '',
                "匹配类型": selected_match.get('match_type', '') if selected_match else '',
                "预期话术": expected_value,
                "关键信息": core_value,
                "下一步动作": selected_match.get('next_action', '') if selected_match else '',
                "SOP节点路径": selected_match.get('sop_node_path_str', '') if selected_match else '',
                "诊断最高相似度": match_debug.get('max_similarity', 0.0) if match_debug else 0.0,
                "诊断候选节点路径": " -> ".join(match_debug.get('max_similarity_path', [])) if match_debug else '',
                "诊断候选参考话术": match_debug.get('max_similarity_reference', '') if match_debug else '',
                "匹配备注": '; '.join(dict.fromkeys([r for r in reasons if r])),  # 去重并保持顺序
                "触发句级备用": "是" if sentencewise_triggered else "",
                "句级备用匹配路径": fallback_path,
                "句级备用匹配相似度": fallback_similarity,
                "句级备用匹配参考话术": fallback_reference,
                "句级备用命中句详情": fallback_trace_json,
                "句级备用备注": fallback_remark,
                "强校验触发": "是" if strong_flag else "",
                "强校验阈值": strong_threshold_value,
                "强校验默认阈值": default_threshold_value,
                "强校验二次跟进阈值": erci_threshold_value,
                "强校验是否命中二次跟进": "是" if strong_is_erci else "",
                "强校验匹配路径": strong_path,
                "强校验预期SOP路径": selected_match.get('strong_validation_expected_path',
                                                        '') if selected_match else '',
                "强校验二级节点相似度": strong_level2_similarity,
                "强校验参考话术": strong_reference,
                "强校验销售话术": strong_sales_message,
                "强校验优先节点": strong_preferred_node,
                "强校验最佳候选节点": strong_best_node,
                "强校验最佳候选相似度": strong_best_similarity,
                "强校验最佳候选路径": strong_best_path,
                "强校验覆盖层级数": strong_coverage,
                "强校验覆盖比例": strong_coverage_ratio,
                "强校验尾节点合法": (
                    "是" if strong_tail_in_branch else ("否" if strong_tail_in_branch is False else "")
                ),
                "强校验尾节点匹配结果": tail_adjust_status,
                "常规流程-预期话术": regular_tail_expected,
                "常规流程-关键信息": regular_tail_core,
            }

            level_names = ['', '一', '二', '三', '四', '五', '六', '七', '八', '九']
            all_level_matches = selected_match.get('all_level_matches', {}) if selected_match else {}
            for level in range(1, 10):
                level_name = level_names[level]
                level_match = all_level_matches.get(level)
                if level_match:
                    labeled_record[f"SOP{level_name}级节点匹配相似度"] = level_match.get('similarity', 0.0)
                    labeled_record[f"SOP{level_name}级节点匹配参考话术"] = level_match.get('matched_reference', '')
                else:
                    labeled_record[f"SOP{level_name}级节点匹配相似度"] = 0.0
                    labeled_record[f"SOP{level_name}级节点匹配参考话术"] = ''

            for level, label in STRONG_VALIDATION_NODE_LABELS.items():
                labeled_record[label] = selected_match.get(label, '') if selected_match else ''

            for col in df.columns:
                if col not in labeled_record:
                    labeled_record[col] = row.get(col, '')

            labeled_records.append(labeled_record)

            # 每处理完batch_size行保存一次
            if (idx + 1) % batch_size == 0:
                batch_count += 1
                try:
                    temp_df = pd.DataFrame(labeled_records)
                    temp_df.to_excel(temp_file_path, index=False, engine='openpyxl')
                    print(f"\n已处理 {idx + 1} 行，保存进度到临时文件（批次 {batch_count}）")
                    print(f"纯改进版匹配: {improved_matches}, 无匹配: {no_matches}")
                except Exception as save_error:
                    print(f"\n警告：保存临时文件失败 - {str(save_error)}")

        # 处理完成后保存最终结果
        if labeled_records:
            try:
                final_df = pd.DataFrame(labeled_records)
                final_df.to_excel(temp_file_path, index=False, engine='openpyxl')
                print(f"\n最终保存 {len(labeled_records)} 条记录到临时文件")
                print(f"纯改进版匹配: {improved_matches}, 无匹配: {no_matches}")
            except Exception as save_error:
                print(f"\n警告：最终保存临时文件失败 - {str(save_error)}")

        print(f"处理完成，临时文件保存在：{temp_file_path}")
        print(f"完成处理，共生成 {len(labeled_records)} 条带标签的记录")

        # 统计结果
        print(f"\n=== 纯改进版算法统计 ===")
        print(f"纯改进版匹配（话术验证成功）: {improved_matches}条")
        print(f"无匹配: {no_matches}条")
        if len(labeled_records) > 0:
            print(f"纯改进版匹配率: {improved_matches / len(labeled_records) * 100:.1f}%")

        return labeled_records

    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")
        print(traceback.format_exc())
        return []


def func_main_pure_improved(**kwargs):
    """主函数 - 纯改进版逻辑"""
    config_data = kwargs["config_data"]

    # 获取配置参数
    corpus_dir = config_data["corpus_dir"]
    pipeline_case_path = config_data["pipeline_case_path"]
    sop_logic_tree = config_data["sop_logic_tree"]

    functions = config_data["functions"]
    similarity = 0.90  # 默认阈值
    batch_size = 100
    sentencewise_options: Optional[Dict[str, Any]] = None
    parallel_workers: Optional[int] = None

    for func_config in functions:
        if func_config.get("name") == "get_sop_case.func_main":
            similarity = func_config.get("similarity", 0.90)
            batch_size = func_config.get("batch_size", 100)
        elif func_config.get("name") == "sentencewise.fallback":
            sentencewise_options = func_config.copy()
            parallel_workers = func_config.get("parallel_workers", parallel_workers)

    print(f"开始纯改进版SOP识别分析...")
    print(f"输入文件：{corpus_dir}")
    print(f"SOP逻辑树：{sop_logic_tree}")
    print(f"相似度阈值：{similarity}")
    print(f"批次大小：{batch_size}")
    print(f"纯改进算法：直接使用改进版逻辑，不使用原版备选")
    if sentencewise_options and sentencewise_matcher:
        print("句级备用匹配已启用，未命中或缺级时将提供候选路径")
    # 分析对话并打标签
    labeled_records = analyze_pure_improved_logic(
        corpus_dir,
        sop_logic_tree,
        similarity,
        batch_size,
        sentencewise_options=sentencewise_options,
        num_workers=parallel_workers
    )

    # 保存结果
    if labeled_records:
        # 修改输出文件名
        pure_improved_output_path = pipeline_case_path.replace('.xlsx', '_pure_improved.xlsx')
        success = list_of_dicts_to_xlsx(labeled_records, pure_improved_output_path)
        if success:
            print(f"成功保存纯改进版结果到：{pure_improved_output_path}")
        else:
            print("保存结果文件失败")
    else:
        print("没有生成任何标签记录")


# def test_module_matching():
#     print("\n--- Running Module Matching Test ---\n")
#     sop_logic_tree_path = "/Users/luojiatai/Documents/trae1/logictree.json"
#     sop_data = read_json_file(sop_logic_tree_path)

#     if not sop_data:
#         print("Error: Could not load SOP logic tree for testing.")
#         return

#     # Load customer_response_patterns.json to get module and province aliases
#     json_file_path = os.path.join(os.path.dirname(__file__), "customer_response_patterns.json")
#     customer_response_patterns = {}
#     if os.path.exists(json_file_path):
#         try:
#             with open(json_file_path, 'r', encoding='utf-8') as f:
#                 customer_response_patterns = json.load(f)
#         except json.JSONDecodeError:
#             print(f"Error decoding JSON from {json_file_path}. File might be malformed.")
#             return

#     module_aliases_map = customer_response_patterns.get('module_aliases', {})
#     canonical_modules = list(module_aliases_map.keys())

#     province_aliases_map = customer_response_patterns.get('province_aliases', {})
#     canonical_provinces = list(province_aliases_map.keys())

#     # Define a simplified logictree_expected_dialogue for testing purposes (Module Context)
#     example_module_expected_dialogue = {
#         "数量": "数量这个模块确实很难，很多人学了很久才能勉强做对 1-2 道，但分值逐年增加，谁拿下才有更多机会，近 2 年咱们教研重点研究出了 10 大方法，完全可以攻克，咱们可以先来认真听～",
#         "常识": "常识知识点多范围广，难记忆，但难度不大，比如时政每天关注，却只考 1 道 0.5 分；知识点多达几千页，所以老师给咱们整理了每个领域的口诀，这是最高效的拿分方法啦，咱们一定按时上课～",
#         "资料": "资料难度不大，是考试唯一有可能拿到满分的模块，用网上的公式多训练熟练度，20 道也能搞定 12 道左右，后续用老师讲的技巧，正确率至少可以稳定在 18 道，甚至全对，本次直播课你可以感受一下～<newline>咱们之前是参加的那个地区的考试呢？老师把省情也给你梳理下",
#         "言语": "言语本身难度不大，但题干长、阅读量大，导致耗时且正确率低；像最难的选词填空，用老师讲的方法可不看题干，正确率超 90%，真的很厉害，所以一定要按时上课～",
#         "判断": "判断模块难度中等，正确率必须保证 80% 以上，考试分数才能理想。这部分薄弱多是一开始学偏了，跟着经验贴学些观察图形特征的假把式，其实一张思维导图就能攻克，本次直播课你可以感受下～",
#         "申论": "主观题难易不好衡量，但是找对方向申论提升比行测快。申论其实就是考察你对机关工作和整合信息的能力，后面直播课感受清华会姐自研的申论高分技巧，按时听课，拿下 70 不成问题～",
#         "政治理论": "政治理论是新增热点，国省考都是重点，主要测查用党的创新理论分析解决问题的能力，是拉分关键，但大家对其考情比较陌生，别担心，本期直播课准备了高效记忆口诀帮你快速掌握，老师都会讲到～",
#         "时政": "时政比较考验咱们平时的积累，知识点比较琐碎，但是考试时候多以热点居多，咱们报名的课程里面每月有一次时政直播课，本期直播课老师也准备了高效记忆口诀帮你快速掌握，老师都会讲到～"
#     }

#     # Define a simplified logictree_expected_dialogue for testing purposes (Province Context)
#     example_province_expected_dialogue = {
#         "广东": "广东是省内单独省考，笔试一般在12月国考后进行，与国考时间接近，属高级难度省份，出题水平高且新题型逐年增加...",
#         "四川": "四川近2年改革，不再分上下半年考试，改为省考单独举行，笔试在12月国考后进行，国省笔试时间接近...",
#         "北京": "北京是单独举行市考，笔试一般在12月国考后进行，与国考时间接近。京考几乎都限户籍...",
#         "上海": "上海是单独举行市考，笔试一般在12月国考后考，与国考时间接近，题型上多考科学推理、数字推理...",
#         "河南": "河南是多省联考省份，笔试3月进行，预计2月出公告。河南与山东同属人口多、重教育的大省...",
#         "江苏": "江苏为省内单独省考，笔试一般在12月国考后进行，与国考时间接近。其难度全国领先...",
#         "浙江": "浙江是省内单独省考，笔试一般在12月国考后进行，与国考时间接近。近2年招录人数递增...",
#         "河北": "河北是多省联考省份，笔试在3月份考，预计2月出公告，近2年招录人数持续递增..."
#     }

#     test_cases = [
#         {
#             "description": "Module: Single, canonical, with weakness",
#             "customer_msg": "我资料分析模块比较薄弱",
#             "expected_utterance": {"资料": "资料难度不大，是考试唯一有可能拿到满分的模块，用网上的公式多训练熟练度，20 道也能搞定 12 道左右，后续用老师讲的技巧，正确率至少可以稳定在 18 道，甚至全对，本次直播课你可以感受一下～<newline>咱们之前是参加的那个地区的考试呢？老师把省情也给你梳理下"},
#             "expected_next_action": "模块讲解或询问考试地区",
#             "expected_sop_node_path": "挖需 -> 与之前参加过考试的客户沟通",
#             "target_node_expected_dialogue": example_module_expected_dialogue,
#             "target_node_path": ["挖需", "与之前参加过考试的客户沟通"]
#         },
#         {
#             "description": "Module: Multiple, mixed aliases, with weakness",
#             "customer_msg": "我的薄弱模块是政治和资料",
#             "expected_utterance": {
#                 "政治理论": example_module_expected_dialogue["政治理论"],
#                 "资料": example_module_expected_dialogue["资料"]
#             },
#             "expected_next_action": "模块讲解或询问考试地区",
#             "expected_sop_node_path": "挖需 -> 与之前参加过考试的客户沟通",
#             "target_node_expected_dialogue": example_module_expected_dialogue,
#             "target_node_path": ["挖需", "与之前参加过考试的客户沟通"]
#         },
#         {
#             "description": "Module: Single, canonical, no explicit weakness (contextual)",
#             "customer_msg": "常识",
#             "expected_utterance": {"常识": example_module_expected_dialogue["常识"]},
#             "expected_next_action": "模块讲解或询问考试地区",
#             "expected_sop_node_path": "挖需 -> 与之前参加过考试的客户沟通",
#             "target_node_expected_dialogue": example_module_expected_dialogue,
#             "target_node_path": ["挖需", "与之前参加过考试的客户沟通"]
#         },
#         {
#             "description": "Module: Multiple, mixed aliases, with weakness",
#             "customer_msg": "言语和判断有点问题",
#             "expected_utterance": {
#                 "言语": example_module_expected_dialogue["言语"],
#                 "判断": example_module_expected_dialogue["判断"]
#             },
#             "expected_next_action": "模块讲解或询问考试地区",
#             "expected_sop_node_path": "挖需 -> 与之前参加过考试的客户沟通",
#             "target_node_expected_dialogue": example_module_expected_dialogue,
#             "target_node_path": ["挖需", "与之前参加过考试的客户沟通"]
#         },
#         {
#             "description": "Module: Multiple, mixed aliases, with weakness",
#             "customer_msg": "我数量和时政都弱",
#             "expected_utterance": {
#                 "数量": example_module_expected_dialogue["数量"],
#                 "时政": example_module_expected_dialogue["时政"]
#             },
#             "expected_next_action": "模块讲解或询问考试地区",
#             "expected_sop_node_path": "挖需 -> 与之前参加过考试的客户沟通",
#             "target_node_expected_dialogue": example_module_expected_dialogue,
#             "target_node_path": ["挖需", "与之前参加过考试的客户沟通"]
#         },
#         {
#             "description": "Province: Single, canonical",
#             "customer_msg": "我想考广东",
#             "expected_utterance": {"广东": example_province_expected_dialogue["广东"]},
#             "expected_next_action": "询问时间",
#             "expected_sop_node_path": "挖需 -> 与首次备考省考的客户沟通 -> 啥也不懂并且询问省份",
#             "target_node_expected_dialogue": example_province_expected_dialogue,
#             "target_node_path": ["挖需", "与首次备考省考的客户沟通", "啥也不懂并且询问省份"]
#         },
#         {
#             "description": "Province: Multiple, mixed aliases",
#             "customer_msg": "我考粤和川",
#             "expected_utterance": {
#                 "广东": example_province_expected_dialogue["广东"],
#                 "四川": example_province_expected_dialogue["四川"]
#             },
#             "expected_next_action": "询问时间",
#             "expected_sop_node_path": "挖需 -> 与首次备考省考的客户沟通 -> 啥也不懂并且询问省份",
#             "target_node_expected_dialogue": example_province_expected_dialogue,
#             "target_node_path": ["挖需", "与首次备考省考的客户沟通", "啥也不懂并且询问省份"]
#         },
#         {
#             "description": "No match (province alias, but module context)",
#             "customer_msg": "我粤语不好", 
#             "expected_utterance": {},
#             "expected_next_action": "模块讲解或询问考试地区",
#             "expected_sop_node_path": "挖需 -> 与之前参加过考试的客户沟通",
#             "target_node_expected_dialogue": example_module_expected_dialogue,
#             "target_node_path": ["挖需", "与之前参加过考试的客户沟通"]
#         },
#         {
#             "description": "No match (module alias, but province context)",
#             "customer_msg": "我常识不好", 
#             "expected_utterance": {},
#             "expected_next_action": "询问时间",
#             "expected_sop_node_path": "挖需 -> 与首次备考省考的客户沟通 -> 啥也不懂并且询问省份",
#             "target_node_expected_dialogue": example_province_expected_dialogue,
#             "target_node_path": ["挖需", "与首次备考省考的客户沟通", "啥也不懂并且询问省份"]
#         }
#     ]

#     for i, tc in enumerate(test_cases):
#         print(f"\n--- Test Case {i+1}: {tc["description"]} ---")
#         conversations = [
#             {"role": "销售", "content": "销售消息", "time": "10:00"},
#             {"role": "客户", "content": tc["customer_msg"], "time": "10:05"}
#         ]

#         # Simulate the structure that find_best_sop_match_improved would return up to the point of '预期话术' processing
#         simulated_best_sop_match = {
#             'sales_message': "销售消息",
#             'sales_time': "10:00",
#             'expected_utterance': {},
#             'next_action': tc["expected_next_action"],
#             'sop_node_path_str': tc["expected_sop_node_path"]
#         }
#         for j, node_name in enumerate(tc["target_node_path"]):
#             node_key = ['first_node', 'second_node', 'third_node', 'fourth_node', 'fifth_node', 'sixth_node', 'seventh_node', 'eighth_node', 'ninth_node'][j]
#             simulated_best_sop_match[node_key] = node_name

#         # Directly call the logic that processes '预期话术' from find_best_sop_match_improved
#         # This is a simplified call for testing the specific part of the logic.
#         # In a real run, find_best_sop_match_improved would handle the full flow.

#         # Re-loading customer_response_patterns inside the test for consistency with the main function
#         # (This is already loaded at the beginning of test_module_matching)

#         # Determine context and use appropriate alias map
#         current_alias_to_canonical = {}
#         potential_utterances_keys = list(tc["target_node_expected_dialogue"].keys())
#         module_key_count = sum(1 for k in potential_utterances_keys if k in canonical_modules)
#         province_key_count = sum(1 for k in potential_utterances_keys if k in canonical_provinces)

#         if module_key_count > province_key_count:
#             current_alias_to_canonical = {alias.lower(): canonical for canonical, aliases in module_aliases_map.items() for alias in ([canonical] + aliases)}
#         elif province_key_count > module_key_count:
#             current_alias_to_canonical = {alias.lower(): canonical for canonical, aliases in province_aliases_map.items() for alias in ([canonical] + aliases)}
#         else: # Ambiguous or no keys, default to module context for this test
#             current_alias_to_canonical = {alias.lower(): canonical for canonical, aliases in module_aliases_map.items() for alias in ([canonical] + aliases)}

#         identified_canonical_items = set()
#         customer_msg_lower = tc["customer_msg"].lower()
#         sorted_aliases = sorted(current_alias_to_canonical.keys(), key=len, reverse=True)

#         for alias_term in sorted_aliases:
#             if alias_term in customer_msg_lower:
#                 identified_canonical_items.add(current_alias_to_canonical[alias_term])

#         matched_dialogues = {}
#         for item_name in identified_canonical_items:
#             if item_name in tc["target_node_expected_dialogue"]:
#                 matched_dialogues[item_name] = tc["target_node_expected_dialogue"][item_name]

#         simulated_best_sop_match['expected_utterance'] = matched_dialogues

#         # Compare results
#         print(f"  Customer Message: {tc["customer_msg"]}")
#         print(f"  Expected Utterance (Actual): {json.dumps(simulated_best_sop_match.get('expected_utterance'), ensure_ascii=False, indent=4)}")
#         print(f"  Expected Utterance (Test Case): {json.dumps(tc["expected_utterance"], ensure_ascii=False, indent=4)}")
#         print(f"  Next Action (Actual): {simulated_best_sop_match.get('next_action')}")
#         print(f"  Next Action (Test Case): {tc["expected_next_action"]}")
#         print(f"  SOP Node Path (Actual): {simulated_best_sop_match.get('sop_node_path_str')}")
#         print(f"  SOP Node Path (Test Case): {tc["expected_sop_node_path"]}")

#         if simulated_best_sop_match.get('expected_utterance') == tc["expected_utterance"] and \
#            simulated_best_sop_match.get('next_action') == tc["expected_next_action"] and \
#            simulated_best_sop_match.get('sop_node_path_str') == tc["expected_sop_node_path"]:
#             print("  Test Passed")
#         else:
#             print("  Test Failed")


if __name__ == "__main__":
    # 测试配置
    config_data = {
        "corpus_dir": "/Users/luojiatai/Documents/trae1/1009-222.xlsx",
        "pipeline_case_path": "/Users/luojiatai/Documents/trae1/节点匹配22-yyds.xlsx",
        "sop_logic_tree": "/Users/luojiatai/Documents/trae1/chengla_wx.json",
        "functions": [
            {
                "name": "get_sop_case.func_main",
                "similarity": 0.90,
                "batch_size": 20
            },
            {
                "name": "sentencewise.fallback",
                "enabled": True,
                "similarity": 0.80,
                "seq_ratio": 0.72,
                "level_thresholds": SENTENCEWISE_DEFAULT_THRESHOLDS,
                "min_level_matches": 1,
                "trigger_on_partial": True,
                "parallel_workers": 4
            }
        ]
    }

    func_main_pure_improved(config_data=config_data)
    # test_module_matching()
