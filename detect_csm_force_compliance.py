#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSM 强遵循识别脚本

根据 CSM 回复内容识别是否命中强遵循话术:
1. 读取独立维护的 `force_patterns_config.json` 中的强遵循话术模式
2. 对输入数据逐条匹配，输出强遵循标签及推荐话术

用法示例:
    python detect_csm_force_compliance.py 输入文件.xlsx --text-column reply
    python detect_csm_force_compliance.py 输入文件.xlsx --config my_patterns.json

输出:
    - 原始数据附加以下字段:
        * 强遵循识别结果       (是 / 否)
        * 强遵循识别话术       (命中的标准话术文本)
        * 强遵循识别分类       (course_overview / module_explanation 等)
        * 强遵循识别模式       (pattern 名称)
        * 强遵循识别关键词     (命中的核心关键词列表)
        * 强遵循识别置信度     (匹配得分, 0-1)
    - 结果默认保存为 `<输入文件名>_force_detected.xlsx`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import pandas as pd

ATTACHMENT_JSON_RE = re.compile(r'\{ *"bucketName"[^{}]*\}', re.IGNORECASE)
MARKDOWN_MEDIA_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
SQUARE_MEDIA_RE = re.compile(r"\[[^\]]+\.(?:png|jpg|jpeg|gif|pdf|docx?)\]")
COLOR_CODE_RE = re.compile(r"\b\d{1,3};rgb:[0-9a-fA-F/]+\b")

CATEGORY_PRIORITY = {
    "province_analysis": 0,
    "module_explanation": 1,
    "course_overview": 2,
    "follow_up": 3,
    "homework_feedback": 4,
    "general_force_compliance": 5,
    "class_invitation": 6,
}

MODULE_STRICT_TERMS = [
    "数量", "数量关系", "常识", "资料", "资料分析", "判断", "图判", "图形推理",
    "言语", "申论", "主观题", "政治理论", "数学运算", "时政"
]

MODULE_PHRASE_WHITELIST = [
    "数量这个模块", "数量关系确实", "数量关系是个难点", "常识知识点", "资料难度不大",
    "言语本身难度不大", "判断模块难度", "主观题难易不好衡量", "申论其实就是",
    "政治理论是新增热点", "时政比较考验", "这几个模块", "模块不太好"
]


def strip_attachment_tokens(text: str) -> str:
    """移除图片/附件链接、色值等噪声内容"""
    cleaned = MARKDOWN_MEDIA_RE.sub(" ", text)
    cleaned = SQUARE_MEDIA_RE.sub(" ", cleaned)
    cleaned = ATTACHMENT_JSON_RE.sub(" ", cleaned)
    cleaned = COLOR_CODE_RE.sub(" ", cleaned)
    return cleaned


def normalize_text(text: str) -> str:
    """统一处理文本中的换行符、附件标记与空白字符"""
    normalized = text.replace("<newline>", " ")
    normalized = strip_attachment_tokens(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_candidate_keywords(text: str) -> List[str]:
    """
    基于标点拆分片段，筛选长度>=2的片段作为关键词。
    关键词用于后续的子串匹配。
    """
    cleaned = strip_attachment_tokens(text.replace("<newline>", " "))
    parts = re.split(r"[，,。！？!?；;：:、\n\r\t\s()（）\[\]{}<>《》“”\"'·…~～❤-]+", cleaned)
    keywords = [p.strip() for p in parts if len(p.strip()) >= 2]
    return keywords


@dataclass
class ScriptPattern:
    """单条强遵循话术模式"""

    name: str
    category: str
    text: str
    source: str
    keywords: List[str] = field(default_factory=list)
    min_matches: int = 2
    min_ratio: float = 0.45
    template: str = ""
    fuzzy_threshold: float = 0.88
    mandatory_keywords: List[str] = field(default_factory=list)

    def _keyword_match(self, reply_text: str) -> Tuple[float, List[str]]:
        """关键词匹配"""
        if not reply_text or not self.keywords:
            return 0.0, []

        matched = [kw for kw in self.keywords if kw and kw in reply_text]
        total = len(self.keywords) or 1
        ratio = len(matched) / total
        return ratio, matched

    def _normalize_for_template(self, text: str) -> str:
        """模板相似度归一化：去非字母数字汉字、统一大小写、数字→x"""
        if not text:
            return ""
        # 替换<newline>等
        normalized = text.replace("<newline>", " ")
        # 只保留字母数字汉字
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", normalized, flags=re.UNICODE)
        # 把数字替换为x
        normalized = re.sub(r"\d", "x", normalized)
        # 多余空格
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        return normalized

    def _fuzzy_match(self, reply_text: str) -> float:
        """模板相似度"""
        if not self.template:
            return 0.0

        normalized_reply = self._normalize_for_template(reply_text)
        normalized_template = self._normalize_for_template(self.template)
        if not normalized_reply or not normalized_template:
            return 0.0

        # 快速包含判断
        if normalized_template in normalized_reply:
            return 1.0
        if normalized_reply in normalized_template:
            return len(normalized_reply) / len(normalized_template)

        return SequenceMatcher(None, normalized_reply, normalized_template).ratio()

    def is_hit(self, reply_text: str) -> Tuple[bool, float, List[str]]:
        """综合关键词与模板相似度判定"""
        keyword_ratio, matched_keywords = self._keyword_match(reply_text)

        keyword_pass = (
            len(matched_keywords) >= self.min_matches or keyword_ratio >= self.min_ratio
        )
        if self.mandatory_keywords:
            keyword_pass = keyword_pass and all(
                mk in matched_keywords for mk in self.mandatory_keywords if mk
            )

        fuzzy_ratio = self._fuzzy_match(reply_text)
        fuzzy_pass = not self.template or fuzzy_ratio >= self.fuzzy_threshold

        if keyword_pass and fuzzy_pass:
            return True, max(keyword_ratio, fuzzy_ratio), matched_keywords

        return False, max(keyword_ratio, fuzzy_ratio), matched_keywords


class CSMForceComplianceDetector:
    """强遵循识别核心逻辑"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.patterns: List[ScriptPattern] = []
        self._load_patterns_from_config()

    def _load_patterns_from_config(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"缺少强遵循话术配置文件: {self.config_path}，"
                "请创建独立的模式维护文件(例如 force_patterns_config.json)。"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        raw_patterns = config.get("patterns", [])
        if not raw_patterns:
            raise ValueError(f"{self.config_path} 中未找到 patterns 列表")

        loaded: List[ScriptPattern] = []
        for idx, pattern_conf in enumerate(raw_patterns, 1):
            text = normalize_text(pattern_conf.get("text", ""))
            if not text:
                continue

            category = pattern_conf.get("category", "").strip()
            if not category:
                continue
            if category == "module_explanation":
                if not any(term in text for term in MODULE_STRICT_TERMS):
                    continue
                if not any(phrase in text for phrase in MODULE_PHRASE_WHITELIST):
                    continue

            min_matches = pattern_conf.get("min_matches")
            min_ratio = pattern_conf.get("min_ratio")

            if "keywords" in pattern_conf:
                keywords = [kw.strip() for kw in pattern_conf["keywords"] if kw.strip()]
            else:
                keywords = sorted(set(extract_candidate_keywords(text)))
            if not keywords:
                continue

            min_matches = int(min_matches) if isinstance(min_matches, (int, float)) else max(2, math.ceil(len(keywords) * 0.35))
            min_ratio = float(min_ratio) if isinstance(min_ratio, (int, float)) else (0.5 if category == "province_analysis" else 0.45)
            template = pattern_conf.get("template", "")
            fuzzy_threshold = float(pattern_conf.get("fuzzy_threshold", 0.88))
            mandatory_keywords = [kw.strip() for kw in pattern_conf.get("mandatory_keywords", []) if kw.strip()]

            loaded.append(
                ScriptPattern(
                    name=pattern_conf.get("name", f"pattern_{idx}"),
                    category=category,
                    text=text,
                    source=pattern_conf.get("source", "custom"),
                    keywords=keywords,
                    min_matches=min_matches,
                    min_ratio=min_ratio,
                    template=template,
                    fuzzy_threshold=fuzzy_threshold,
                    mandatory_keywords=mandatory_keywords,
                )
            )

        if not loaded:
            raise ValueError(f"{self.config_path} 中未成功加载任何模式，请检查配置内容。")

        # 去重 (category + text)
        unique: Dict[Tuple[str, str], ScriptPattern] = {}
        for pattern in loaded:
            unique[(pattern.category, pattern.text)] = pattern

        self.patterns = sorted(unique.values(), key=lambda p: (p.category, p.name))
        print(f"✅ 从配置加载强遵循模式 {len(self.patterns)} 条 (来源: {self.config_path})")

    # ------------------------------------------------------------------ #
    # 匹配逻辑
    # ------------------------------------------------------------------ #
    def detect_reply(self, reply_text: str) -> Dict[str, object]:
        """
        对单条 CSM 回复进行识别
        返回: 包含是否命中、命中话术、分类、匹配关键词、得分等信息的字典
        """
        if not isinstance(reply_text, str) or not reply_text.strip():
            return self._empty_result()

        normalized = normalize_text(reply_text)
        best_match: Optional[Tuple[ScriptPattern, float, List[str]]] = None

        for pattern in self.patterns:
            is_hit, ratio, matched_keywords = pattern.is_hit(normalized)
            if not is_hit:
                continue

            candidate = (pattern, ratio, matched_keywords)
            if best_match is None:
                best_match = candidate
            else:
                best_pattern, best_ratio, best_keywords = best_match
                if ratio > best_ratio:
                    best_match = candidate
                elif math.isclose(ratio, best_ratio):
                    if len(matched_keywords) > len(best_keywords):
                        best_match = candidate
                    elif len(matched_keywords) == len(best_keywords):
                        current_priority = CATEGORY_PRIORITY.get(pattern.category, 99)
                        best_priority = CATEGORY_PRIORITY.get(best_pattern.category, 99)
                        if current_priority < best_priority:
                            best_match = candidate

        if best_match is None:
            return self._empty_result()

        pattern, ratio, matched_keywords = best_match
        return {
            "is_force": True,
            "force_script": pattern.text,
            "force_category": pattern.category,
            "force_pattern": pattern.name,
            "force_keywords": ", ".join(matched_keywords),
            "force_score": round(ratio, 4),
            "force_source": pattern.source,
        }

    @staticmethod
    def _empty_result() -> Dict[str, object]:
        return {
            "is_force": False,
            "force_script": "",
            "force_category": "",
            "force_pattern": "",
            "force_keywords": "",
            "force_score": 0.0,
            "force_source": "",
        }

    # ------------------------------------------------------------------ #
    # 数据处理
    # ------------------------------------------------------------------ #
    def annotate_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """为 DataFrame 附加强遵循识别结果"""
        if text_column not in df.columns:
            raise KeyError(f"未找到回复列 '{text_column}'，当前列: {list(df.columns)}")

        results = df[text_column].apply(self.detect_reply)
        result_df = pd.json_normalize(results)

        df = df.copy()
        df["强遵循识别结果"] = result_df["is_force"].map(lambda x: "是" if x else "否")
        df["强遵循识别话术"] = result_df["force_script"]
        df["强遵循识别分类"] = result_df["force_category"]
        df["强遵循识别模式"] = result_df["force_pattern"]
        df["强遵循识别关键词"] = result_df["force_keywords"]
        df["强遵循识别置信度"] = result_df["force_score"]
        df["强遵循识别来源"] = result_df["force_source"]

        return df


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 CSM 回复识别强遵循话术")
    parser.add_argument("input_file",default="1104挖需修改数据-yyds_pure_improved.xlsx", help="待识别的 Excel/CSV 文件路径")
    parser.add_argument(
        "--text-column",
        default="发送消息内容",
        help="CSM 回复所在列名 (默认: reply)",
    )
    parser.add_argument(
        "--output-file",
        default="1104挖需修改数据-yyds_force_detected.xlsx",
        help="结果输出路径 (默认: <输入文件名>_force_detected.xlsx)",
    )
    parser.add_argument(
        "--config",
        default="force_patterns_config.json",
        help="强遵循话术配置文件路径 (默认: force_patterns_config.json)",
    )
    return parser.parse_args()


def load_dataframe(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(file_path)
    if ext == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"不支持的文件格式: {ext}")


def guess_default_output(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return f"{base}_force_detected.xlsx"


def main() -> None:
    args = parse_args()

    print("🚀 启动 CSM 强遵循识别")
    print(f"   输入文件: {args.input_file}")
    print(f"   回复列名: {args.text_column}")

    detector = CSMForceComplianceDetector(config_path=args.config)

    df = load_dataframe(args.input_file)
    print(f"📖 读取数据成功，共 {len(df)} 条记录，列: {list(df.columns)}")

    annotated_df = detector.annotate_dataframe(df, args.text_column)
    total_force = (annotated_df["强遵循识别结果"] == "是").sum()
    print(f"✅ 强遵循识别完成，命中 {total_force} 条 / {len(annotated_df)}")

    output_file = args.output_file or guess_default_output(args.input_file)
    annotated_df.to_excel(output_file, index=False, engine="openpyxl")
    print(f"💾 结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
