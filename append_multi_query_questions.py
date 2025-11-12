"""
Append context-engaged product/industry questions to positive customer replies.

Usage:
    python append_multi_query_questions.py \
        --input 珍酒正面回复+问题.xlsx \
        --output 珍酒正面回复+问题_多query.xlsx \
        --variants 2 --workers 4
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from threading import Lock
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from qwen_32b_inner import Qwen32BClient


QUESTION_CATEGORIES = {
    "产品相关": [
        "产品介绍",
        "基酒年份",
        "调味酒年份",
        "酿造工艺",
        "酒款定位",
        "外形包装",
        "产品对比",
        "竞品对比",
    ],
    "行业相关": [
        "香型区别",
        "香型代表",
        "香型特点",
        "饮用场景",
    ],
}

PROMPT_TEMPLATE = """请根据以下信息，输出若干条客户追加提问（中文）：
- 客户身份：四五十岁的珍酒老客户，语气自然朴实。
- 当前对话场景：
```
{context}
```
- 客户刚回应销售说：“{base_sentence}”
- 客户想追加提问，问题类型：{category}（示例：{examples}）

输出要求：
1. 每条为一句完整问题，口吻自然，不过度公式化。
2. 问题需在上述回复基础上顺势追问，让对话自然连贯。
3. 问题须与“{category}”主题相关，可结合自身体验或好奇点。
4. 用不同提问角度，避免重复。
5. 长度不超过20个汉字。
6. 输出JSON数组，例如 ["问题1", "问题2", ...]，数组长度为 {variant_count}。
"""


MAX_LEN = 20
DISALLOWED_PHRASES = [
    "作为AI",
    "很抱歉",
    "抱歉",
    "请您",
    "敬请",
    "烦请",
]


@dataclass
class AugmentResult:
    new_message: str
    question_category: str
    question_subtype: str


_prompt_cache: Dict[str, List[str]] = {}
_cache_lock = Lock()


def generate_questions(
    client: Qwen32BClient,
    context: str,
    base_sentence: str,
    category: str,
    examples: str,
    variant_count: int,
) -> List[str]:
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        base_sentence=base_sentence,
        category=category,
        examples=examples,
        variant_count=variant_count,
    )

    temperatures = [0.2, 0.35]
    results: List[str] = []
    seen: set[str] = set()

    for temp in temperatures:
        response = client.generate(prompt=prompt, temperature=temp).strip()
        candidates: List[str] = []
        parsed = False
        for parser in (lambda text: json.loads(text), lambda text: ast.literal_eval(text)):
            if parsed:
                break
            try:
                data = parser(response)
                if isinstance(data, str):
                    data = [data]
                if isinstance(data, list):
                    candidates = [str(item) for item in data]
                    parsed = True
            except Exception:
                continue

        if not parsed:
            match = re.search(r"\[[^\]]+\]", response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if isinstance(data, list):
                        candidates = [str(item) for item in data]
                        parsed = True
                except Exception:
                    try:
                        data = ast.literal_eval(match.group(0))
                        if isinstance(data, list):
                            candidates = [str(item) for item in data]
                            parsed = True
                    except Exception:
                        pass

        if not parsed:
            fallback = [frag.strip() for frag in response.split("\n") if frag.strip()]
            if fallback:
                candidates = fallback
                parsed = True

        if not parsed or not isinstance(candidates, list):
            continue

        for question in candidates:
            question = str(question).strip()
            if not question or question in seen:
                continue
            question = re.sub(r"<think>.*?</think>", "", question, flags=re.DOTALL).strip()
            if "?" not in question and "？" not in question:
                question += "？"
            if any(bad in question for bad in DISALLOWED_PHRASES):
                continue
            if len(question) > MAX_LEN:
                question = question[:MAX_LEN].rstrip("，,。.!？?")
                question += "？"
            if len(question) < 3:
                continue
            seen.add(question)
            results.append(question)
            if len(results) >= variant_count:
                break
        if len(results) >= variant_count:
            break

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="为正面回复追加产品/行业问题")
    parser.add_argument("--input", default="珍酒正面回复+问题.xlsx", help="输入Excel，包含正面客户回复")
    parser.add_argument(
        "--output",
        default="珍酒正面回复+问题_多query.xlsx",
        help="输出Excel，附加问题后的结果",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="每条客户消息追加多少个问题（默认1）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发线程数，默认4；设为1则串行执行",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")

    df = pd.read_excel(input_path)
    if "上文-不带img路径" not in df.columns or "最新客户消息" not in df.columns:
        raise KeyError("输入表缺少必须列：上文-不带img路径 或 最新客户消息")

    client = Qwen32BClient()

    rows: List[dict] = []
    tasks: Dict[object, Dict[str, object]] = {}
    total_tasks = len(df)

    submit_bar = tqdm(total=total_tasks, desc="提交改写任务", unit="条")
    product_cycle = cycle(QUESTION_CATEGORIES["产品相关"])
    industry_cycle = cycle(QUESTION_CATEGORIES["行业相关"])
    category_cycle = cycle(["产品相关", "行业相关"])
    row_question: Dict[int, str] = {idx: "" for idx in range(len(df))}
    row_cat: Dict[int, str] = {idx: "" for idx in range(len(df))}
    row_sub: Dict[int, str] = {idx: "" for idx in range(len(df))}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for idx, row in df.iterrows():
            context = str(row["上文-不带img路径"])
            base_sentence = str(row["最新客户消息"]).strip()
            if not base_sentence:
                submit_bar.update(1)
                continue
            cat = next(category_cycle)
            subtype = next(product_cycle) if cat == "产品相关" else next(industry_cycle)

            prompt_key = json.dumps(
                {
                    "context": context,
                    "base": base_sentence,
                    "category": cat,
                    "subtype": subtype,
                    "variants": 1,
                },
                ensure_ascii=False,
            )
            with _cache_lock:
                cached = _prompt_cache.get(prompt_key)
            if cached:
                question = cached[0]
                row_question[idx] = question
                row_cat[idx] = cat
                row_sub[idx] = subtype
                submit_bar.update(1)
                continue

            future = executor.submit(
                generate_questions,
                client,
                context,
                base_sentence,
                subtype,
                "、".join(QUESTION_CATEGORIES[cat][:3]),
                1,
            )
            tasks[future] = {
                "row_index": idx,
                "category": cat,
                "subtype": subtype,
                "prompt_key": prompt_key,
            }
            submit_bar.update(1)
    submit_bar.close()

    process_bar = tqdm(total=len(tasks), desc="等待模型生成问题", unit="条")
    for future in as_completed(tasks):
        process_bar.update(1)
        payload = tasks[future]
        row_idx = payload["row_index"]
        cat = payload["category"]
        subtype = payload["subtype"]
        prompt_key = payload["prompt_key"]
        try:
            questions = future.result()
        except Exception as exc:
            print(f"[警告] 行 {row_idx} 追加问题失败：{exc}")
            continue

        if not questions:
            continue

        with _cache_lock:
            _prompt_cache[prompt_key] = questions

        question = questions[0]
        row_question[row_idx] = question
        row_cat[row_idx] = cat
        row_sub[row_idx] = subtype
    process_bar.close()

    for idx, row in df.iterrows():
        question = row_question[idx]
        if not question:
            continue
        base_sentence = str(row["最新客户消息"]).strip()
        augmented = base_sentence + " " + question

        new_row = row.to_dict().copy()
        new_row["最新客户消息"] = augmented
        new_row["追加问题类别"] = row_cat[idx]
        new_row["追加问题子类"] = row_sub[idx]
        rows.append(new_row)

    if not rows:
        raise RuntimeError("未生成任何多query案例，请检查输出或模型返回")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_path, index=False)
    print(f"生成案例共 {len(out_df)} 条，已保存至 {out_path}")


if __name__ == "__main__":
    main()
