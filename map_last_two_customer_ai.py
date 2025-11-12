import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


MESSAGE_PATTERN = re.compile(r"^\[(客户|销售)\]\[(.*?)\]:\s*(.*)$")


@dataclass
class Message:
    speaker: str
    timestamp: str
    content: str


def normalize_timestamp(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        text = str(value).strip()
        return text if text else None


def parse_history(history: Any) -> List[Message]:
    if not isinstance(history, str) or not history.strip():
        return []

    messages: List[Message] = []
    current: Optional[Message] = None
    for raw_line in history.splitlines():
        line = raw_line.rstrip()
        match = MESSAGE_PATTERN.match(line)
        if match:
            if current:
                current.content = current.content.strip()
                messages.append(current)
            speaker, timestamp, content = match.groups()
            current = Message(speaker, timestamp.strip(), content.strip())
        else:
            if current:
                addition = raw_line.strip()
                if addition:
                    if current.content:
                        current.content += "\n" + addition
                    else:
                        current.content = addition
    if current:
        current.content = current.content.strip()
        messages.append(current)
    return messages


def last_two_customer_messages(messages: List[Message]) -> List[Message]:
    if not messages:
        return []

    idx = len(messages) - 1
    if messages[idx].speaker != "客户":
        return []

    customer_block: List[Message] = []
    while idx >= 0 and messages[idx].speaker == "客户":
        customer_block.append(messages[idx])
        idx -= 1
    customer_block.reverse()
    return customer_block[-2:] if len(customer_block) >= 2 else customer_block


class ProgressBar:
    def __init__(self, total: int, bar_length: int = 30) -> None:
        self.total = max(total, 1)
        self.bar_length = bar_length
        self.last_percent = -1

    def update(self, processed: int) -> None:
        percent = processed / self.total
        percent_int = int(percent * 100)
        if percent_int == self.last_percent:
            return
        self.last_percent = percent_int
        filled = int(self.bar_length * percent)
        bar = "#" * filled + "-" * (self.bar_length - filled)
        sys.stdout.write(f"\r进度 [{bar}] {percent*100:5.1f}% ({processed}/{self.total})")
        sys.stdout.flush()

    def finish(self) -> None:
        self.update(self.total)
        sys.stdout.write("\n")
        sys.stdout.flush()


def build_ai_lookup(ai_df: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
    lookup: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for _, row in ai_df.iterrows():
        customer_id = row.get("客户ID")
        if not isinstance(customer_id, str) or not customer_id:
            continue

        if str(row.get("消息生成主体", "")).strip().upper() != "SA":
            continue

        timestamp = normalize_timestamp(row.get("客户消息时间"))
        if not timestamp:
            continue

        value = row.get("AI生成消息")
        candidates: List[str] = []
        if isinstance(value, str):
            text = value.strip()
            if text:
                candidates.append(text)

        if candidates:
            key = (customer_id, timestamp)
            existing = lookup[key]
            for text in candidates:
                if text not in existing:
                    existing.append(text)
    return lookup


def process_rows(
    turn_df: pd.DataFrame,
    lookup: Dict[Tuple[str, str], List[str]],
) -> pd.DataFrame:
    result_column = "倒数两条客户消息AI映射"
    flag_all_empty_column = "倒数两条AI全空"
    flag_partial_empty_column = "倒数两条AI含空"
    turn_df[result_column] = ""
    turn_df[flag_all_empty_column] = ""
    turn_df[flag_partial_empty_column] = ""

    progress = ProgressBar(len(turn_df))
    processed = 0

    for idx, row in turn_df.iterrows():
        customer_id = row.get("客户ID", "")
        history = row.get("历史对话", "")
        messages = last_two_customer_messages(parse_history(history))

        mapping: Dict[str, str] = {}
        for message in messages:
            timestamp = normalize_timestamp(message.timestamp)
            key = (customer_id, timestamp if timestamp else "")
            ai_messages = lookup.get(key, []) if timestamp else []
            if not ai_messages:
                mapping[message.content] = ""
            else:
                mapping[message.content] = " || ".join(ai_messages)

        if not mapping and isinstance(row.get("连续客户消息列表"), str):
            try:
                message_list = json.loads(row["连续客户消息列表"])
                if isinstance(message_list, list):
                    mapping = {str(msg): "" for msg in message_list[-2:]}
            except json.JSONDecodeError:
                pass

        turn_df.at[idx, result_column] = json.dumps(mapping, ensure_ascii=False)
        if mapping:
            values = list(mapping.values())
            all_empty = all(value == "" for value in values)
            any_empty = any(value == "" for value in values)
            turn_df.at[idx, flag_all_empty_column] = "TRUE" if all_empty else "FALSE"

            if len(values) < 2:
                partial_flag = "TRUE"
            else:
                partial_flag = "TRUE" if any_empty and not all_empty else "FALSE"
            turn_df.at[idx, flag_partial_empty_column] = partial_flag
        else:
            turn_df.at[idx, flag_all_empty_column] = ""
            turn_df.at[idx, flag_partial_empty_column] = ""
        processed += 1
        progress.update(processed)

    progress.finish()
    return turn_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="仅映射倒数两条连续客户消息的AI生成内容"
    )
    parser.add_argument(
        "--turn-file",
        default="转向客户消息.xlsx",
        help="包含连续客户消息的Excel文件",
    )
    parser.add_argument(
        "--ai-file",
        default="chengla_all_2025-10-12(1-3).xlsx",
        help="包含AI消息数据的Excel文件",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "outputs", "倒数两条客户消息_AI映射.xlsx"),
        help="输出Excel路径",
    )
    args = parser.parse_args()

    print(">>> 正在加载数据...")
    turn_df = pd.read_excel(args.turn_file)
    ai_df = pd.read_excel(args.ai_file)

    print(">>> 构建AI消息索引...")
    lookup = build_ai_lookup(ai_df)
    print(f">>> 已索引AI消息条目: {len(lookup)}")

    print(">>> 开始映射倒数两条客户消息")
    processed_df = process_rows(turn_df.copy(), lookup)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    processed_df.to_excel(args.output, index=False)

    summary = processed_df["倒数两条客户消息AI映射"].apply(
        lambda x: len(json.loads(x)) if x else 0
    )
    empty_flags = processed_df["倒数两条AI全空"].value_counts(dropna=False).to_dict()
    partial_flags = processed_df["倒数两条AI含空"].value_counts(dropna=False).to_dict()
    print(
        f">>> 处理完成，存在映射的记录 {summary.gt(0).sum()} 条；"
        f"无映射记录 {summary.eq(0).sum()} 条"
    )
    true_count = empty_flags.get("TRUE", 0)
    false_count = empty_flags.get("FALSE", 0)
    blank_count = empty_flags.get("", 0)
    print(
        f">>> 倒数两条AI全空：TRUE={true_count}, FALSE={false_count}, 未判定={blank_count}"
    )
    partial_true = partial_flags.get("TRUE", 0)
    partial_false = partial_flags.get("FALSE", 0)
    partial_blank = partial_flags.get("", 0)
    print(
        f">>> 倒数两条AI含空：TRUE={partial_true}, FALSE={partial_false}, 未判定={partial_blank}"
    )
    print(f">>> 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
