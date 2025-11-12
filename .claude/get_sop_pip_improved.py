#!/usr/bin/env python3
"""
改进版SOP识别脚本
修改算法逻辑：从客户消息往上查找符合期望SOP节点的话术
"""

import traceback
import jieba
import pandas as pd
import os
import json
import re

from typing import Dict
from tqdm import tqdm

# 从原文件导入基础函数
from get_sop_pip import (
    preprocess_sentence,
    calculate_sentence_similarity,
    parse_conversation_history,
    find_all_level_sop_matches,
    read_json_file,
    list_of_dicts_to_xlsx
)


def find_targeted_sop_match(conversations, target_sop_l1, target_sop_l2, sop_data, similarity_threshold=0.90):
    """
    从客户消息往上查找符合特定期望SOP节点的销售话术

    新的匹配逻辑：
    1. 从对话历史中找到最后一条客户消息
    2. 从该客户消息往上遍历，查找销售消息
    3. 检查每个销售消息是否匹配目标SOP节点的话术
    4. 找到匹配的就返回，否则返回None

    参数:
        conversations: 对话列表（已按时间顺序排列）
        target_sop_l1: 期望的SOP一级节点
        target_sop_l2: 期望的SOP二级节点
        sop_data: SOP逻辑树数据
        similarity_threshold: 相似度阈值

    返回:
        dict: 匹配结果，如果找到匹配的话术则返回匹配信息，否则返回None
    """
    if not conversations or not target_sop_l1 or not target_sop_l2:
        return None

    # 检查目标SOP节点是否存在
    if target_sop_l1 not in sop_data or target_sop_l2 not in sop_data[target_sop_l1]:
        return None

    target_sop_content = sop_data[target_sop_l1][target_sop_l2]
    target_scripts = []

    # 收集目标SOP节点的所有话术（参考话术 + 相似话术）
    target_scripts.extend(target_sop_content.get('参考话术', []))
    target_scripts.extend(target_sop_content.get('相似话术', []))

    if not target_scripts:
        return None

    # 从后往前遍历对话，查找销售消息
    for conv in reversed(conversations):
        if conv['role'] == '销售':
            sales_message = conv['content']

            # 检查是否匹配目标SOP节点的任何话术
            best_similarity = 0
            best_matched_script = ""
            match_type = ""

            # 检查参考话术
            for script in target_sop_content.get('参考话术', []):
                if script and script.strip():
                    similarity = calculate_sentence_similarity(sales_message, script)
                    if similarity >= similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_matched_script = script
                        match_type = "参考话术"

            # 检查相似话术
            for script in target_sop_content.get('相似话术', []):
                if script and script.strip():
                    similarity = calculate_sentence_similarity(sales_message, script)
                    if similarity >= similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_matched_script = script
                        match_type = "相似话术"

            # 如果找到匹配，返回结果
            if best_similarity >= similarity_threshold:
                return {
                    'first_node': target_sop_l1,
                    'second_node': target_sop_l2,
                    'similarity': best_similarity,
                    'matched_reference': best_matched_script,
                    'match_type': match_type,
                    'sales_message': sales_message,
                    'sales_time': conv['time']
                }

    # 没找到匹配
    return None


def analyze_conversation_sop_labels_improved(sales_corpus_xlsx, sop_logic_tree_path, similarity_threshold=0.90, batch_size=100):
    """
    改进版SOP标签分析函数

    新逻辑：
    1. 对每条记录，读取期望的SOP标签
    2. 从客户消息往上查找是否有匹配期望SOP节点的销售话术
    3. 如果找到匹配，标记为正确；否则使用原有的最近匹配逻辑作为备选

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

        # 检查必要的列
        required_columns = ['最终传参上下文', 'SOP一级节点', 'SOP二级节点']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"错误：缺少必要的列 {', '.join(missing_columns)}")
            return []

        print("使用对话列：最终传参上下文")
        print("使用期望SOP标签列：SOP一级节点, SOP二级节点")

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
        temp_file_path = os.path.join(temp_dir, f"{temp_file_base}_improved_temp_progress.xlsx")

        print(f"每处理 {batch_size} 行将保存进度到：{temp_file_path}")
        print(f"使用改进算法：从客户消息往上查找符合期望SOP节点的话术")

        targeted_matches = 0  # 目标匹配计数
        fallback_matches = 0  # 备选匹配计数

        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="分析对话记录")):
            # 解析历史对话
            history_str = str(row.get('最终传参上下文', '')).strip()
            if not history_str or history_str == 'nan':
                continue

            conversations = parse_conversation_history(history_str)
            if not conversations:
                continue

            # 检查最后一条消息是否为客户消息
            if conversations and conversations[-1]['role'] != '客户':
                print(f"警告：第{row.name}行的最后一条消息不是客户消息，跳过")
                continue

            # 获取期望的SOP标签
            expected_l1 = str(row.get('SOP一级节点', '')).strip()
            expected_l2 = str(row.get('SOP二级节点', '')).strip()

            # 获取最后一条客户消息
            last_customer_msg = conversations[-1]['content'] if conversations else ''

            sop_match = None
            match_method = "无匹配"

            # 1. 首先尝试目标匹配：查找符合期望SOP节点的话术
            if expected_l1 and expected_l2:
                sop_match = find_targeted_sop_match(
                    conversations, expected_l1, expected_l2, sop_data, similarity_threshold
                )
                if sop_match:
                    match_method = "目标匹配"
                    targeted_matches += 1

            # 2. 如果目标匹配失败，使用原有的最近匹配逻辑作为备选
            if not sop_match:
                # 导入原有函数
                from get_sop_pip import find_nearest_sales_sop_match
                sop_match = find_nearest_sales_sop_match(conversations, sop_data, similarity_threshold)
                if sop_match:
                    match_method = "最近匹配"
                    fallback_matches += 1

            # 创建带标签的记录
            labeled_record = {
                "最后客户消息": last_customer_msg,
                "期望SOP一级节点": expected_l1,
                "期望SOP二级节点": expected_l2,
                "匹配方法": match_method,
                "最近销售消息": sop_match.get('sales_message', '') if sop_match else '',
                "销售消息时间": sop_match.get('sales_time', '') if sop_match else '',
                "SOP一级节点": sop_match.get('first_node', '') if sop_match else '',
                "SOP二级节点": sop_match.get('second_node', '') if sop_match else '',
                "匹配相似度": sop_match.get('similarity', 0.0) if sop_match else 0.0,
                "匹配的参考话术": sop_match.get('matched_reference', '') if sop_match else '',
                "匹配类型": sop_match.get('match_type', '') if sop_match else ''
            }

            # 保留原始文件的所有列
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
                    print(f"目标匹配: {targeted_matches}, 备选匹配: {fallback_matches}")
                except Exception as save_error:
                    print(f"\n警告：保存临时文件失败 - {str(save_error)}")

        # 处理完成后保存最终结果
        if labeled_records:
            try:
                final_df = pd.DataFrame(labeled_records)
                final_df.to_excel(temp_file_path, index=False, engine='openpyxl')
                print(f"\n最终保存 {len(labeled_records)} 条记录到临时文件")
                print(f"目标匹配: {targeted_matches}, 备选匹配: {fallback_matches}")
            except Exception as save_error:
                print(f"\n警告：最终保存临时文件失败 - {str(save_error)}")

        print(f"处理完成，临时文件保存在：{temp_file_path}")
        print(f"完成处理，共生成 {len(labeled_records)} 条带标签的记录")
        return labeled_records

    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")
        print(traceback.format_exc())
        return []


def func_main_improved(**kwargs):
    """改进版主函数"""
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

    print(f"开始改进版SOP识别分析...")
    print(f"输入文件：{corpus_dir}")
    print(f"SOP逻辑树：{sop_logic_tree}")
    print(f"相似度阈值：{similarity}")
    print(f"批次大小：{batch_size}")
    print(f"改进算法：从客户消息往上查找符合期望SOP节点的话术")

    # 分析对话并打标签
    labeled_records = analyze_conversation_sop_labels_improved(
        corpus_dir,
        sop_logic_tree,
        similarity,
        batch_size
    )

    # 保存结果
    if labeled_records:
        # 修改输出文件名以区分改进版本
        improved_output_path = pipeline_case_path.replace('.xlsx', '_improved.xlsx')
        success = list_of_dicts_to_xlsx(labeled_records, improved_output_path)
        if success:
            print(f"成功保存改进版结果到：{improved_output_path}")

            # 统计结果
            targeted_count = sum(1 for r in labeled_records if r['匹配方法'] == '目标匹配')
            fallback_count = sum(1 for r in labeled_records if r['匹配方法'] == '最近匹配')
            total_matched = targeted_count + fallback_count

            print(f"\n=== 改进版算法统计 ===")
            print(f"目标匹配（符合期望SOP）: {targeted_count}条")
            print(f"备选匹配（最近话术）: {fallback_count}条")
            print(f"总匹配数: {total_matched}条")
            print(f"匹配率: {total_matched/len(labeled_records)*100:.1f}%")

            if targeted_count > 0:
                print(f"目标匹配成功率: {targeted_count/len(labeled_records)*100:.1f}%")
        else:
            print("保存结果文件失败")
    else:
        print("没有生成任何标签记录")


if __name__ == "__main__":
    # 测试配置
    config_data = {
        "corpus_dir": "/Users/luojiatai/Documents/trae1/正确案例.xlsx",
        "pipeline_case_path": "/Users/luojiatai/Documents/trae1/正确案例2.xlsx",
        "sop_logic_tree": "/Users/luojiatai/Documents/trae1/logic_tree_v2.json",
        "functions": [
            {
                "name": "get_sop_case.func_main",
                "similarity": 0.90,
                "batch_size": 100
            }
        ]
    }

    func_main_improved(config_data=config_data)