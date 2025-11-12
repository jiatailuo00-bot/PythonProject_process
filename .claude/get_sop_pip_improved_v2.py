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

from typing import Dict, Any, Tuple, List
from difflib import SequenceMatcher
from tqdm import tqdm

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


def _collect_matches_by_path(sales_message: str, sop_data: Dict, similarity_threshold: float) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    """收集满足阈值的节点匹配，按路径聚合"""
    matches: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    def _traverse(node: Dict, current_path: List[str], depth: int):
        if not isinstance(node, dict):
            return

        for key in ("参考话术", "相似话术"):
            scripts = node.get(key, [])
            if isinstance(scripts, list):
                iterable = scripts
            elif isinstance(scripts, str):
                iterable = [scripts]
            else:
                iterable = []

            for script in iterable:
                if not script or not script.strip():
                    continue
                similarity = calculate_sentence_similarity(sales_message, script)
                seq_ratio = SequenceMatcher(None, sales_message, script).ratio()
                if (similarity >= similarity_threshold or seq_ratio >= 0.4) and current_path:
                    path_tuple = tuple(current_path)
                    existing = matches.get(path_tuple)
                    effective_similarity = max(similarity, seq_ratio)
                    if existing is None or effective_similarity > existing["similarity"]:
                        matches[path_tuple] = {
                            "similarity": effective_similarity,
                            "matched_reference": script,
                            "match_type": key,
                            "depth": depth,
                            "path": list(current_path)
                        }

        for child_key, child_value in node.items():
            if isinstance(child_value, dict) and child_key not in ("参考话术", "相似话术", "下一步动作", "预期话术"):
                _traverse(child_value, current_path + [child_key], depth + 1)

    _traverse(sop_data, [], 0)
    return matches


def _max_similarity_to_scripts(sales_messages, scripts):
    """计算所有销售话术与节点脚本的最大相似度"""
    if not scripts:
        return 0.0
    best_sim = 0.0
    for msg in sales_messages:
        if not msg:
            continue
        for script in scripts:
            sim = calculate_sentence_similarity(msg, script)
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


def _build_chain_for_path(path_tuple: Tuple[str, ...], entries: List[Dict[str, Any]], upto_index: int) -> Dict[int, Dict[str, Any]]:
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

def find_best_sop_match_improved(conversations, sop_data, similarity_threshold=0.90):
    """
    纯改进版逻辑：结合原版深层级匹配 + 改进版"从客户消息往上查找"逻辑

    参数:
        conversations: 对话列表
        sop_data: SOP逻辑树数据
        similarity_threshold: 相似度阈值

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

    for idx, conv in enumerate(conversations[:last_customer_idx + 1]):
        if conv.get('role') != '销售':
            continue
        sales_message = conv.get('content', '')
        debug_info["checked_sales_messages"] += 1
        last_sales_idx = idx if last_sales_idx is None or idx > last_sales_idx else last_sales_idx

        matches_by_path = _collect_matches_by_path(sales_message, sop_data, similarity_threshold)
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
                debug_info['candidate_sales_message'] = sales_message

    debug_info["sales_messages"] = [entry["sales_message"] for entry in sales_entries]
    if last_matched_idx is None:
        last_matched_idx = last_sales_idx

    best_path = None
    best_chain = {}
    best_score = None
    best_entry_idx = -1

    for entry_idx, entry in enumerate(sales_entries):
        for path_tuple, match_info in entry['matches_by_path'].items():
            if len(path_tuple) < 2:
                continue
            chain = _build_chain_for_path(path_tuple, sales_entries, entry_idx)
            if not chain:
                continue
            depth = len(path_tuple)
            coverage = len(chain)
            min_sim = min(info['similarity'] for info in chain.values())
            sum_sim = sum(info['similarity'] for info in chain.values())
            max_idx = max(info['entry_index'] for info in chain.values())
            contains_last = int(any(info['entry_index'] == last_matched_idx for info in chain.values()))
            score = (contains_last, coverage, min_sim, max_idx, -depth, sum_sim)

            if best_score is None or score > best_score:
                best_score = score
                best_path = path_tuple
                best_chain = chain
                best_entry_idx = entry_idx

    if best_path is None:
        if single_best is None:
            if not sales_entries:
                debug_info["reason"] = "客户消息之前没有销售话术"
            elif debug_info["max_similarity"] == 0.0:
                debug_info["reason"] = "未找到与SOP节点相似的销售话术"
            else:
                debug_info["reason"] = (
                    f"所有销售话术与SOP的最高相似度仅 {debug_info['max_similarity']:.2f} "
                    f"(阈值 {similarity_threshold})"
                )
            return None, debug_info
        best_path = single_best[1]
        best_entry_idx = single_best[2]
        fallback_entry = single_best[4]
        chain = {}
        # 允许部分层级命中
        for level in range(2, len(best_path) + 1):
            prefix = tuple(best_path[:level])
            best_match = None
            for entry in reversed(sales_entries[:best_entry_idx + 1]):
                info = entry['matches_by_path'].get(prefix)
                if not info:
                    continue
                candidate = {
                    'similarity': info['similarity'],
                    'matched_reference': info['matched_reference'],
                    'match_type': info['match_type'],
                    'sales_message': entry['sales_message'],
                    'sales_time': entry['sales_time'],
                    'entry_index': entry['index'],
                    'path': list(prefix)
                }
            if (best_match is None
                    or candidate['similarity'] > best_match['similarity']
                    or (candidate['similarity'] == best_match['similarity']
                        and candidate['entry_index'] > best_match['entry_index'])):
                best_match = candidate
        if best_match:
            chain[level] = best_match
        # 确保至少最深一层被记录
        deepest_level = len(best_path)
        if deepest_level not in chain:
            info = single_best[3]
            chain[deepest_level] = {
                'similarity': info['similarity'],
                'matched_reference': info['matched_reference'],
                'match_type': info['match_type'],
                'sales_message': fallback_entry['sales_message'],
                'sales_time': fallback_entry['sales_time'],
                'entry_index': fallback_entry['index'],
                'path': list(best_path)
            }
        best_chain = chain
    else:
        # best_chain already populated by _build_chain_for_path
        pass

    node_path = list(best_path)
    aggregated_public = {
        level: {k: v for k, v in info.items() if k != 'entry_index'}
        for level, info in best_chain.items()
    }
    node_keys_in_sop_match = ['first_node', 'second_node', 'third_node', 'fourth_node',
                              'fifth_node', 'sixth_node', 'seventh_node', 'eighth_node', 'ninth_node']

    deepest_level = max(best_chain.keys()) if best_chain else 1
    deepest_info = best_chain.get(deepest_level, {})

    final_match = {
        'best_path': node_path,
        'sales_message': deepest_info.get('sales_message', ''),
        'sales_time': deepest_info.get('sales_time'),
        'all_level_matches': aggregated_public,
        'matched_levels': sorted(aggregated_public.keys()),
        'missing_levels': [lvl for lvl in range(2, len(node_path) + 1) if lvl not in aggregated_public],
        'sop_node_path_str': ' -> '.join(node_path) if node_path else ''
    }

    for idx, node_name in enumerate(node_path):
        if idx >= len(node_keys_in_sop_match):
            break
        level = idx + 1
        if level == 1 or level in aggregated_public:
            final_match[node_keys_in_sop_match[idx]] = node_name
        else:
            final_match[node_keys_in_sop_match[idx]] = ''

    if deepest_info:
        final_match['similarity'] = deepest_info.get('similarity', 0.0)
        final_match['matched_reference'] = deepest_info.get('matched_reference', '')
        final_match['match_type'] = deepest_info.get('match_type', '')
    else:
        final_match['similarity'] = 0.0
        final_match['matched_reference'] = ''
        final_match['match_type'] = ''

    # 计算预期话术和下一步动作
    customer_patterns_path = os.path.join(os.path.dirname(__file__), "customer_response_patterns.json")
    customer_response_patterns = {}
    if os.path.exists(customer_patterns_path):
        try:
            with open(customer_patterns_path, 'r', encoding='utf-8') as f:
                customer_response_patterns = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {customer_patterns_path}. File might be malformed.")

    module_aliases_map = customer_response_patterns.get('module_aliases', {})
    alias_to_canonical_module = {}
    for canonical, aliases in module_aliases_map.items():
        alias_to_canonical_module[canonical.lower()] = canonical
        for alias in aliases:
            alias_to_canonical_module[alias.lower()] = canonical

    province_aliases_map = customer_response_patterns.get('province_aliases', {})
    alias_to_canonical_province = {}
    for canonical, aliases in province_aliases_map.items():
        alias_to_canonical_province[canonical.lower()] = canonical
        for alias in aliases:
            alias_to_canonical_province[alias.lower()] = canonical

    current_sop_node = _get_nested_value(sop_data, node_path)
    expected_utterance = {}
    next_action = ""
    if current_sop_node and '预期话术' in current_sop_node:
        potential_utterances = current_sop_node['预期话术']
        if isinstance(potential_utterances, dict):
            last_customer_msg_lower = last_customer_msg.lower()
            identified_items = set()

            sorted_module_aliases = sorted(alias_to_canonical_module.keys(), key=len, reverse=True)
            for alias_term in sorted_module_aliases:
                if alias_term in last_customer_msg_lower:
                    identified_items.add(alias_to_canonical_module[alias_term])

            if not identified_items:
                sorted_province_aliases = sorted(alias_to_canonical_province.keys(), key=len, reverse=True)
                for alias_term in sorted_province_aliases:
                    if alias_term in last_customer_msg_lower:
                        identified_items.add(alias_to_canonical_province[alias_term])

            matched_dialogues = {}
            for item_name in identified_items:
                if item_name in potential_utterances:
                    matched_dialogues[item_name] = potential_utterances[item_name]

            if matched_dialogues:
                expected_utterance = matched_dialogues
                next_action = current_sop_node.get('下一步动作', '')
            else:
                expected_utterance = potential_utterances
                next_action = current_sop_node.get('下一步动作', '')
        elif isinstance(potential_utterances, list) and potential_utterances:
            expected_utterance = potential_utterances[0]
            next_action = current_sop_node.get('下一步动作', '')
        elif isinstance(potential_utterances, str):
            expected_utterance = potential_utterances
            next_action = current_sop_node.get('下一步动作', '')

    final_match['expected_utterance'] = expected_utterance
    final_match['next_action'] = next_action

    debug_info["reason"] = ""
    debug_info["matched_sales_message"] = deepest_info.get('sales_message', '')
    debug_info["aggregated_levels"] = aggregated_public
    debug_info["depth"] = len(node_path)

    return final_match, debug_info


def analyze_pure_improved_logic(sales_corpus_xlsx, sop_logic_tree_path, similarity_threshold=0.90, batch_size=100):
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

        improved_matches = 0  # 改进版匹配计数
        no_matches = 0       # 无匹配计数

        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="分析对话记录")):
            history_str = str(row.get(conversation_column, '')).strip()
            reasons = []
            best_sop_match = None
            match_debug = {}
            last_customer_msg = ''

            if not history_str or history_str == 'nan':
                reasons.append("对话历史为空或缺失")
                no_matches += 1
            else:
                conversations = parse_conversation_history(history_str)
                if not conversations:
                    reasons.append("对话历史解析失败")
                    no_matches += 1
                else:
                    # 记录最后一个客户消息
                    for conv in reversed(conversations):
                        if conv.get('role') == '客户':
                            last_customer_msg = conv.get('content', '')
                            break
                    if not last_customer_msg:
                        reasons.append("没有找到客户消息，无法定位SOP节点")

                    if conversations[-1]['role'] != '客户':
                        reasons.append("最后一条消息不是客户消息，建议确认对话截断位置")

                    best_sop_match, match_debug = find_best_sop_match_improved(conversations, sop_data, similarity_threshold)

                    if best_sop_match:
                        match_method = "纯改进版匹配"
                        improved_matches += 1
                        node_path = best_sop_match.get('best_path') or [
                            best_sop_match[key] for key in ['first_node', 'second_node', 'third_node', 'fourth_node',
                                                            'fifth_node', 'sixth_node', 'seventh_node', 'eighth_node', 'ninth_node']
                            if best_sop_match.get(key)
                        ]
                        level_matches = best_sop_match.get('all_level_matches', {})
                        sales_texts = match_debug.get('sales_messages', [])
                        level_reasons = _describe_unmatched_levels(node_path, level_matches, sop_data, similarity_threshold, sales_texts)
                        reasons.extend(level_reasons)
                    else:
                        match_method = "无匹配"
                        no_matches += 1
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
            match_method = "纯改进版匹配" if best_sop_match else "无匹配"

            labeled_record = {
                "最后客户消息": last_customer_msg,
                "匹配方法": match_method,
                "最近销售消息": best_sop_match.get('sales_message', '') if best_sop_match else '',
                "销售消息时间": best_sop_match.get('sales_time', '') if best_sop_match else '',
                "SOP一级节点": best_sop_match.get('first_node', '') if best_sop_match else '',
                "SOP二级节点": best_sop_match.get('second_node', '') if best_sop_match else '',
                "SOP三级节点": best_sop_match.get('third_node', '') if best_sop_match else '',
                "SOP四级节点": best_sop_match.get('fourth_node', '') if best_sop_match else '',
                "SOP五级节点": best_sop_match.get('fifth_node', '') if best_sop_match else '',
                "SOP六级节点": best_sop_match.get('sixth_node', '') if best_sop_match else '',
                "SOP七级节点": best_sop_match.get('seventh_node', '') if best_sop_match else '',
                "SOP八级节点": best_sop_match.get('eighth_node', '') if best_sop_match else '',
                "SOP九级节点": best_sop_match.get('ninth_node', '') if best_sop_match else '',
                "匹配相似度": best_sop_match.get('similarity', 0.0) if best_sop_match else 0.0,
                "匹配的参考话术": best_sop_match.get('matched_reference', '') if best_sop_match else '',
                "匹配类型": best_sop_match.get('match_type', '') if best_sop_match else '',
                "预期话术": best_sop_match.get('expected_utterance', '') if best_sop_match else '',
                "下一步动作": best_sop_match.get('next_action', '') if best_sop_match else '',
                "SOP节点路径": best_sop_match.get('sop_node_path_str', '') if best_sop_match else '',
                "诊断最高相似度": match_debug.get('max_similarity', 0.0) if match_debug else 0.0,
                "诊断候选节点路径": " -> ".join(match_debug.get('max_similarity_path', [])) if match_debug else '',
                "诊断候选参考话术": match_debug.get('max_similarity_reference', '') if match_debug else '',
                "匹配备注": '; '.join(dict.fromkeys([r for r in reasons if r]))  # 去重并保持顺序
            }

            level_names = ['', '一', '二', '三', '四', '五', '六', '七', '八', '九']
            all_level_matches = best_sop_match.get('all_level_matches', {}) if best_sop_match else {}
            for level in range(1, 10):
                level_name = level_names[level]
                level_match = all_level_matches.get(level)
                if level_match:
                    labeled_record[f"SOP{level_name}级节点匹配相似度"] = level_match.get('similarity', 0.0)
                    labeled_record[f"SOP{level_name}级节点匹配参考话术"] = level_match.get('matched_reference', '')
                else:
                    labeled_record[f"SOP{level_name}级节点匹配相似度"] = 0.0
                    labeled_record[f"SOP{level_name}级节点匹配参考话术"] = ''

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
            print(f"纯改进版匹配率: {improved_matches/len(labeled_records)*100:.1f}%")

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

    for func_config in functions:
        if func_config.get("name") == "get_sop_case.func_main":
            similarity = func_config.get("similarity", 0.90)
            batch_size = func_config.get("batch_size", 100)

    print(f"开始纯改进版SOP识别分析...")
    print(f"输入文件：{corpus_dir}")
    print(f"SOP逻辑树：{sop_logic_tree}")
    print(f"相似度阈值：{similarity}")
    print(f"批次大小：{batch_size}")
    print(f"纯改进算法：直接使用改进版逻辑，不使用原版备选")

    # 分析对话并打标签
    labeled_records = analyze_pure_improved_logic(
        corpus_dir,
        sop_logic_tree,
        similarity,
        batch_size
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
        "corpus_dir": "/Users/luojiatai/Documents/trae1/badcase9.xlsx",
        "pipeline_case_path": "/Users/luojiatai/Documents/trae1/badcase9-pure.xlsx",
        "sop_logic_tree": "/Users/luojiatai/Documents/trae1/logictree.json",
        "functions": [
            {
                "name": "get_sop_case.func_main",
                "similarity": 0.90,
                "batch_size": 100
            }
        ]
    }

    func_main_pure_improved(config_data=config_data)
    # test_module_matching()
