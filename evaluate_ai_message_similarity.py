import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROMPT_TEMPLATE = """/no_think
# 角色
你是一个汉语言专家，擅长识别语句之间的内容与语义相似性


# 任务
检查如下两句话是否存在重复的内容或相似的语义，如果存在则输出"是"，否则输出"否"
1. {{ reply }}

2. {{ latest_reply }}


# 输出规范
返回以下结构化响应，并且不能使用```包裹，
{{output_schema}}
"""

OUTPUT_SCHEMA = '{"result": "是" | "否"}'


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value)


def load_model():
    try:
        from qwen_32b_inner import llm_qwen3_32b

        return llm_qwen3_32b
    except Exception as exc:
        print(f"⚠️ 无法加载Qwen32B模型，将使用启发式判断。原因: {exc}")
        return None


def call_model(llm, reply: str, latest_reply: str, temperature: float = 0.1) -> Tuple[str, str]:
    prompt = PROMPT_TEMPLATE.replace("{{ reply }}", reply).replace(
        "{{ latest_reply }}", latest_reply
    ).replace("{{output_schema}}", OUTPUT_SCHEMA)
    try:
        response = llm(prompt, temperature=temperature, timeout=60)
        return response.strip(), prompt
    except Exception as exc:
        return f'{{"result": "错误"}}#error:{exc}', prompt


def heuristic_judge(reply: str, latest_reply: str) -> str:
    if normalize_text(reply) == normalize_text(latest_reply):
        return "是"
    return "否"


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


def extract_ai_pairs(mapping_str: str) -> List[str]:
    if not isinstance(mapping_str, str) or not mapping_str.strip():
        return []
    try:
        mapping = json.loads(mapping_str)
    except json.JSONDecodeError:
        return []
    if not isinstance(mapping, dict):
        return []
    values = list(mapping.values())
    return values[:2]


def evaluate_rows(df: pd.DataFrame, llm) -> pd.DataFrame:
    result_column = "AI生成消息重复判定"
    raw_column = "AI生成消息判定原始响应"
    prompt_column = "AI生成消息判定Prompt"
    first_ai_column = "AI生成消息_第一条"
    second_ai_column = "AI生成消息_第二条"
    df[result_column] = ""
    df[raw_column] = ""
    df[prompt_column] = ""
    df[first_ai_column] = ""
    df[second_ai_column] = ""

    processed = 0
    progress = ProgressBar(len(df))

    for idx, row in df.iterrows():
        ai_map = row.get("倒数两条客户消息AI映射", "")
        values = extract_ai_pairs(ai_map)
        reply = values[0] if len(values) > 0 else ""
        latest_reply = values[1] if len(values) > 1 else ""

        df.at[idx, first_ai_column] = reply
        df.at[idx, second_ai_column] = latest_reply

        if not reply and not latest_reply:
            df.at[idx, result_column] = "无AI消息"
            df.at[idx, raw_column] = ""
            df.at[idx, prompt_column] = ""
        elif not reply or not latest_reply:
            df.at[idx, result_column] = "缺少一条AI消息"
            df.at[idx, raw_column] = ""
            df.at[idx, prompt_column] = ""
        else:
            if llm:
                raw_response, prompt = call_model(llm, reply, latest_reply)
                df.at[idx, raw_column] = raw_response
                df.at[idx, prompt_column] = prompt
                match = re.search(r'"result"\s*:\s*"(?P<res>[是否])"', raw_response)
                if match:
                    df.at[idx, result_column] = match.group("res")
                else:
                    df.at[idx, result_column] = "解析失败"
            else:
                df.at[idx, raw_column] = ""
                df.at[idx, prompt_column] = ""
                df.at[idx, result_column] = heuristic_judge(reply, latest_reply)

        processed += 1
        progress.update(processed)

    progress.finish()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对倒数两条AI生成消息进行语义重复判定"
    )
    parser.add_argument(
        "--input",
        default="用于对AI生成消息进行评判.xlsx",
        help="输入Excel文件路径",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "outputs", "AI生成消息重复判定结果.xlsx"),
        help="输出Excel文件路径",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="模型调用温度参数",
    )
    args = parser.parse_args()

    print(">>> 正在读取数据...")
    df = pd.read_excel(args.input)

    print(">>> 尝试加载模型...")
    llm = load_model()

    print(">>> 开始判定AI生成消息是否重复")
    evaluated_df = evaluate_rows(df.copy(), llm)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    evaluated_df.to_excel(args.output, index=False)

    summary = evaluated_df["AI生成消息重复判定"].value_counts(dropna=False).to_dict()
    print(f">>> 判定结果统计: {summary}")
    print(f">>> 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
