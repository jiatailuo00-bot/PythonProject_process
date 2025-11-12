#!/usr/bin/env python3
"""
SOP reference matcher for ZLKT conversations.

Given an Excel dataset that contains full conversation histories plus SOP level
labels, and a SOP logic tree JSON that stores canonical reference scripts
down to level-three nodes, this tool checks whether any sales-side reply in the
history aligns with the reference scripts of the labeled SOP node.

Usage examples (if you omit every flag, the script defaults to mode=both,
`zlkt_all_2025-10-31_to_2025-11-04_combined_customer_dataset_sampled.xlsx`,
and `logic_tree_zlkt_origin2.json`):
    # 校验是否遵循既定 SOP 节点
    python zlkt_sop_reference_matcher.py \
        --mode verify \
        --excel zlkt_all_2025-10-31_to_2025-11-04_combined_preprocessed_sop.xlsx \
        --sop-json logic_tree_zlkt_origin2.json \
        --output zlkt_sop_matched.xlsx

    # 自动识别最可能的 SOP 节点（从最新销售消息向前匹配）
    python zlkt_sop_reference_matcher.py \
        --mode infer \
        --excel zlkt_all_2025-10-31_to_2025-11-04_combined_customer_dataset_sampled.xlsx \
        --sop-json logic_tree_zlkt_origin2.json \
        --output zlkt_sop_inferred.xlsx \
        --history-column 完整历史对话

    # 一次输出校验+自动定位两类字段
    python zlkt_sop_reference_matcher.py \
        --mode both \
        --excel zlkt_all_2025-10-31_to_2025-11-04_combined_customer_dataset_sampled.xlsx \
        --sop-json logic_tree_zlkt_origin2.json \
        --output zlkt_sop_combined.xlsx \
        --history-column 完整历史对话
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_EXCEL_PATH = Path("badcase2233.xlsx")
DEFAULT_SOP_JSON_PATH = Path("logic_tree_zlkt_origin2.json")
DEFAULT_HISTORY_FALLBACKS = ["历史对话", "完整历史对话"]
VERIFY_COLUMN_MAP = {
    "match_status": "校验-是否命中",
    "match_reason": "校验-未命中原因",
    "matched_sales_message": "校验-命中销售话术",
    "matched_sales_time": "校验-命中销售时间",
    "matched_reference_text": "校验-参考话术",
    "matched_reference_source": "校验-参考来源",
    "matched_reference_level3": "校验-三级节点",
    "matched_similarity": "校验-匹配相似度",
    "matched_turn_from_last": "校验-倒数第几条销售",
    "matched_round_from_last": "校验-倒数第几轮销售",
    "latest_sales_message": "尾部销售话术",
    "latest_sales_time": "尾部销售时间",
    "match_hit": "校验-命中标记",
}
INFER_COLUMN_MAP = {
    "sop_match_found": "自动定位-是否命中",
    "failure_reason": "自动定位-失败原因",
    "predicted_level1": "自动定位-SOP一级",
    "predicted_level2": "自动定位-SOP二级",
    "matched_reference_level3": "自动定位-三级节点",
    "matched_reference_text": "自动定位-参考话术",
    "matched_reference_source": "自动定位-参考来源",
    "matched_similarity": "自动定位-匹配相似度",
    "matched_sales_message": "自动定位-命中销售话术",
    "matched_sales_time": "自动定位-命中销售时间",
    "matched_turn_from_last": "自动定位-倒数第几条销售",
    "matched_round_from_last": "自动定位-倒数第几轮销售",
    "latest_sales_message": "尾部销售话术",
    "latest_sales_time": "尾部销售时间",
}

LINE_PATTERN = re.compile(r"^\[(?P<speaker>.+?)]\[(?P<timestamp>.+?)]\s*:\s*(?P<text>.*)$")
NOISE_PATTERN = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def _split_reference_text(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    candidate = text.strip()
    if not candidate:
        return []
    if "<newline>" not in candidate:
        return [candidate]
    before, _ = candidate.split("<newline>", 1)
    variants: List[str] = []
    head = before.strip()
    if head:
        variants.append(head)
    variants.append(candidate)
    return variants


@dataclass(frozen=True)
class ReferenceUtterance:
    level1: str
    level2: str
    text: str
    source: str  # 参考话术 / 相似话术等
    level3: Optional[str] = None
    normalized: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalized", normalize_text(self.text))


class SOPReferenceResolver:
    """
    Converts the nested SOP JSON into a flat lookup.
    Supports finding reference utterances for (level1, level2) pairs while
    collapsing level-three entries back to their parent level-two node.
    """

    def __init__(self, tree: Dict[str, Dict[str, Dict]]) -> None:
        self._tree = tree
        self._refs: Dict[Tuple[str, str], List[ReferenceUtterance]] = {}
        self._index: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        self._all_refs: List[ReferenceUtterance] = []
        self._leading_index: Dict[str, List[ReferenceUtterance]] = {}
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._reference_tfidf = None
        self._ref_index_map: Dict[int, int] = {}
        self._flatten()
        self._build_vector_index()

    def _flatten(self) -> None:
        for level1, level2_nodes in self._tree.items():
            if not isinstance(level2_nodes, dict):
                continue
            for level2, payload in level2_nodes.items():
                if not isinstance(payload, dict):
                    continue
                refs = []
                refs.extend(self._extract_refs(payload, level1, level2, level3=None))
                for maybe_level3, nested in payload.items():
                    if isinstance(nested, dict):
                        refs.extend(self._extract_refs(nested, level1, level2, level3=maybe_level3))
                if refs:
                    key = (level1, level2)
                    self._refs[key] = refs
                    norm_key = (self._normalize_key(level1), self._normalize_key(level2))
                    self._index.setdefault(norm_key, []).append(key)
                    self._all_refs.extend(refs)
                    for ref in refs:
                        leading = ref.normalized[:1]
                        self._leading_index.setdefault(leading, []).append(ref)

    @staticmethod
    def _extract_refs(
        node: Dict,
        level1: str,
        level2: str,
        level3: Optional[str],
    ) -> List[ReferenceUtterance]:
        utterances: List[ReferenceUtterance] = []
        for field in ("参考话术", "相似话术"):
            values = node.get(field)
            if isinstance(values, str):
                for variant in _split_reference_text(values):
                    utterances.append(ReferenceUtterance(level1, level2, variant, field, level3))
            elif isinstance(values, Iterable):
                for entry in values:
                    if isinstance(entry, str):
                        for variant in _split_reference_text(entry):
                            utterances.append(
                                ReferenceUtterance(level1, level2, variant, field, level3)
                            )
        return utterances

    @staticmethod
    def _normalize_key(value: Optional[str]) -> str:
        if value is None:
            return ""
        collapsed = NOISE_PATTERN.sub("", unicodedata.normalize("NFKC", str(value)))
        return collapsed.lower()

    def get_references(self, level1: str, level2: str) -> List[ReferenceUtterance]:
        """
        Returns all reference utterances for the requested node.
        Attempts an exact lookup first, then falls back to normalized matching.
        """
        direct_key = (level1, level2)
        if direct_key in self._refs:
            return self._refs[direct_key]
        norm_key = (self._normalize_key(level1), self._normalize_key(level2))
        candidates = self._index.get(norm_key, [])
        if candidates:
            # Return refs of the first candidate; duplicates should be rare.
            return self._refs[candidates[0]]
        return []

    def all_references(self) -> List[ReferenceUtterance]:
        return list(self._all_refs)

    def reference_candidates_for(self, normalized_sales_text: str) -> List[ReferenceUtterance]:
        if not normalized_sales_text:
            return self._all_refs
        leading = normalized_sales_text[:1]
        return self._leading_index.get(leading) or self._all_refs

    def semantic_similarities(self, text: str, references: Sequence[ReferenceUtterance]) -> Optional[np.ndarray]:
        if (
            not references
            or self._tfidf_vectorizer is None
            or self._reference_tfidf is None
        ):
            return None
        indices: List[int] = []
        for ref in references:
            idx = self._ref_index_map.get(id(ref))
            if idx is None:
                return None
            indices.append(idx)
        if not indices:
            return None
        vec = self._tfidf_vectorizer.transform([text])
        subset = self._reference_tfidf[indices]
        sims = cosine_similarity(vec, subset).flatten()
        return sims

    def _build_vector_index(self) -> None:
        if not self._all_refs:
            return
        corpus = [ref.text for ref in self._all_refs]
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(corpus)
        self._tfidf_vectorizer = vectorizer
        self._reference_tfidf = matrix
        self._ref_index_map = {id(ref): idx for idx, ref in enumerate(self._all_refs)}


def normalize_text(text: str) -> str:
    token = unicodedata.normalize("NFKC", str(text))
    token = token.lower()
    token = NOISE_PATTERN.sub("", token)
    return token


def compute_similarity(text_a: str, text_b: str) -> float:
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0
    if norm_a in norm_b or norm_b in norm_a:
        shorter, longer = (norm_a, norm_b) if len(norm_a) <= len(norm_b) else (norm_b, norm_a)
        coverage = len(shorter) / len(longer) if longer else 0.0
        return coverage
    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    set_a, set_b = set(norm_a), set(norm_b)
    if not set_a or not set_b:
        return ratio
    overlap = len(set_a & set_b) / max(len(set_a), len(set_b))
    return max(ratio, overlap)


def resolve_column_name(
    df: pd.DataFrame,
    preferred: str,
    fallback_candidates: Optional[Sequence[str]] = None,
) -> str:
    fallback_candidates = fallback_candidates or []
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    for name in fallback_candidates:
        if name not in candidates:
            candidates.append(name)
    for name in candidates:
        if name in df.columns:
            if name != preferred:
                print(f"列“{preferred}”不存在，自动切换为“{name}”。")
            return name
    raise ValueError(
        f"数据表缺少列：{preferred}（以及备用列 {', '.join(fallback_candidates) or '无'}）"
    )


def parse_history(history_text: str) -> List[Dict[str, Optional[str]]]:
    """
    Parses the full history blob into ordered entries.
    Lines that do not start with `[speaker][timestamp]:` are treated as
    continuations of the latest entry.
    """
    if not history_text:
        return []
    entries: List[Dict[str, Optional[str]]] = []
    current: Optional[Dict[str, Optional[str]]] = None
    for raw_line in str(history_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_PATTERN.match(line)
        if match:
            if current:
                entries.append(current)
            current = {
                "speaker": match.group("speaker"),
                "timestamp": match.group("timestamp"),
                "text": match.group("text").strip(),
            }
        else:
            if current:
                current["text"] = f"{current.get('text', '')} {line}".strip()
            else:
                current = {"speaker": None, "timestamp": None, "text": line}
    if current:
        entries.append(current)
    return entries


def _extract_sales_entries_with_rounds(
    entries: List[Dict[str, Optional[str]]],
    keywords: Sequence[str]
) -> List[Dict[str, Optional[str]]]:
    sales_entries: List[Dict[str, Optional[str]]] = []
    current_round = 0
    in_sales_block = False
    for entry in entries:
        if is_sales_speaker(entry.get("speaker"), keywords):
            if not in_sales_block:
                current_round += 1
                in_sales_block = True
            entry["_sales_round"] = current_round
            sales_entries.append(entry)
        else:
            in_sales_block = False
    total_rounds = current_round
    for entry in sales_entries:
        round_index = entry.get("_sales_round") or 0
        entry["_sales_round_from_last"] = total_rounds - round_index + 1 if total_rounds else None
    return sales_entries


def _extract_last_sales_entry(
    history_text: str,
    keywords: Sequence[str]
) -> Tuple[str, Optional[str]]:
    entries = parse_history(history_text)
    for entry in reversed(entries):
        if is_sales_speaker(entry.get("speaker"), keywords):
            return entry.get("text") or "", entry.get("timestamp")
    return "", None


def is_sales_speaker(speaker: Optional[str], keywords: Sequence[str]) -> bool:
    if not speaker:
        return False
    return any(keyword in speaker for keyword in keywords if keyword)


def find_sales_match(
    history_text: str,
    references: Sequence[ReferenceUtterance],
    keywords: Sequence[str],
    similarity_threshold: float,
    resolver: Optional[SOPReferenceResolver] = None,
) -> Dict[str, Optional[str]]:
    """
    Returns metadata about the latest (from the end) sales reply that matches
    any reference utterance above the similarity threshold.
    """
    entries = parse_history(history_text)
    sales_entries = _extract_sales_entries_with_rounds(entries, keywords)
    if not sales_entries:
        return {
            "match_status": False,
            "match_reason": "历史对话中没有销售侧消息",
        }

    if not references:
        return {
            "match_status": False,
            "match_reason": "SOP节点未找到参考话术",
        }

    best_result: Optional[Dict[str, Optional[str]]] = None
    for reverse_idx, entry in enumerate(reversed(sales_entries), start=1):
        text = entry.get("text") or ""
        semantic_scores = resolver.semantic_similarities(text, references) if resolver else None
        best_ref: Optional[ReferenceUtterance] = None
        best_score = 0.0
        for idx, ref in enumerate(references):
            lexical_score = compute_similarity(text, ref.text)
            semantic_score = float(semantic_scores[idx]) if semantic_scores is not None else 0.0
            score = max(lexical_score, semantic_score)
            if score > best_score:
                best_score = score
                best_ref = ref
        if best_ref and best_score >= similarity_threshold:
            best_result = {
                "match_status": True,
                "match_reason": "",
                "matched_sales_message": text,
                "matched_sales_time": entry.get("timestamp"),
                "matched_reference_text": best_ref.text,
                "matched_reference_source": best_ref.source,
                "matched_reference_level3": best_ref.level3 or "",
                "matched_similarity": round(best_score, 4),
                "matched_turn_from_last": reverse_idx,
                "matched_round_from_last": entry.get("_sales_round_from_last"),
            }
            break

    if best_result:
        return best_result

    return {
        "match_status": False,
        "match_reason": "未找到满足阈值的匹配",
    }


def identify_node_from_history(
    history_text: str,
    resolver: SOPReferenceResolver,
    all_references: Sequence[ReferenceUtterance],
    keywords: Sequence[str],
    similarity_threshold: float,
) -> Dict[str, Optional[str]]:
    entries = parse_history(history_text)
    sales_entries = _extract_sales_entries_with_rounds(entries, keywords)
    if not sales_entries:
        return {
            "sop_match_found": False,
            "failure_reason": "历史对话中没有销售侧消息",
        }
    if not all_references:
        return {
            "sop_match_found": False,
            "failure_reason": "SOP逻辑树未提供参考话术",
        }

    for reverse_idx, entry in enumerate(reversed(sales_entries), start=1):
        text = entry.get("text") or ""
        normalized_text = normalize_text(text)
        candidates = resolver.reference_candidates_for(normalized_text)
        best_ref: Optional[ReferenceUtterance] = None
        best_score = 0.0
        semantic_scores = resolver.semantic_similarities(text, candidates)
        for idx, ref in enumerate(candidates):
            lexical_score = compute_similarity(text, ref.text)
            semantic_score = float(semantic_scores[idx]) if semantic_scores is not None else 0.0
            score = max(lexical_score, semantic_score)
            if score > best_score:
                best_score = score
                best_ref = ref
        if best_ref and best_score >= similarity_threshold:
            return {
                "sop_match_found": True,
                "predicted_level1": best_ref.level1,
                "predicted_level2": best_ref.level2,
                "matched_reference_level3": best_ref.level3 or "",
                "matched_reference_text": best_ref.text,
                "matched_reference_source": best_ref.source,
                "matched_similarity": round(best_score, 4),
                "matched_sales_message": text,
                "matched_sales_time": entry.get("timestamp"),
                "matched_turn_from_last": reverse_idx,
                "matched_round_from_last": entry.get("_sales_round_from_last"),
            }

    return {
        "sop_match_found": False,
        "failure_reason": "未找到满足阈值的匹配",
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match full-history sales replies against SOP reference scripts."
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "infer", "both"],
        default="both",
        help="verify: 仅校验；infer: 仅自动定位；both: 同时输出两类字段。",
    )
    parser.add_argument(
        "--excel",
        default=str(DEFAULT_EXCEL_PATH),
        help=f"Path to the source Excel dataset (default: {DEFAULT_EXCEL_PATH.name}).",
    )
    parser.add_argument(
        "--sop-json",
        default=str(DEFAULT_SOP_JSON_PATH),
        help=f"Path to the SOP logic tree JSON file (default: {DEFAULT_SOP_JSON_PATH.name}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination Excel path for augmented results. "
        "Defaults to <excel_stem>_<mode>.xlsx when omitted.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Minimum similarity score to accept a match (default: 0.78).",
    )
    parser.add_argument(
        "--sales-keywords",
        default="销售,CSM",
        help="Comma-separated keywords that denote sales-side speakers in历史对话.",
    )
    parser.add_argument(
        "--history-column",
        default="历史对话",
        help="Name of the column that holds the full conversation history.",
    )
    parser.add_argument(
        "--level1-column",
        default="SOP一级节点",
        help="Column containing SOP level-one labels.",
    )
    parser.add_argument(
        "--level2-column",
        default="SOP二级节点",
        help="Column containing SOP level-two labels.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    excel_path = Path(args.excel).expanduser().resolve()
    sop_path = Path(args.sop_json).expanduser().resolve()
    output_path = Path(args.output) if args.output else None
    if output_path is None:
        output_path = excel_path.with_name(f"{excel_path.stem}_{args.mode}.xlsx")
    else:
        output_path = Path(output_path).expanduser().resolve()

    if not excel_path.exists():
        parser.error(f"无法找到 Excel 文件：{excel_path}")
    if not sop_path.exists():
        parser.error(f"无法找到 SOP JSON 文件：{sop_path}")

    sales_keywords = [kw.strip() for kw in args.sales_keywords.split(",") if kw.strip()]
    if not sales_keywords:
        parser.error("至少提供一个销售侧关键字以识别销售消息。")

    print(f"加载对话数据：{excel_path}")
    df = pd.read_excel(excel_path)
    print(f"加载 SOP 逻辑树：{sop_path}")
    with sop_path.open("r", encoding="utf-8") as f:
        sop_tree = json.load(f)
    resolver = SOPReferenceResolver(sop_tree)

    history_col = resolve_column_name(df, args.history_column, DEFAULT_HISTORY_FALLBACKS)

    merged = df.copy()
    level1_col: Optional[str] = None
    level2_col: Optional[str] = None
    if args.mode in {"verify", "both"}:
        level1_col = resolve_column_name(df, args.level1_column, [args.level1_column])
        level2_col = resolve_column_name(df, args.level2_column, [args.level2_column])
        verify_df = run_verify_mode(
            df=df,
            resolver=resolver,
            sales_keywords=sales_keywords,
            history_col=history_col,
            level1_col=level1_col,
            level2_col=level2_col,
            similarity_threshold=args.similarity_threshold,
        )
        verify_df = verify_df.rename(columns=VERIFY_COLUMN_MAP)
        for col in verify_df.columns:
            merged[col] = verify_df[col]
    if args.mode in {"infer", "both"}:
        infer_df = run_infer_mode(
            df=df,
            resolver=resolver,
            sales_keywords=sales_keywords,
            history_col=history_col,
            similarity_threshold=args.similarity_threshold,
        )
        infer_df = infer_df.rename(columns=INFER_COLUMN_MAP)
        for col in infer_df.columns:
            merged[col] = infer_df[col]

        if args.mode == "both":
            level2_col = level2_col or resolve_column_name(df, args.level2_column, [args.level2_column])
            level1_col = level1_col or resolve_column_name(df, args.level1_column, [args.level1_column])

            def compare(orig_series: pd.Series, pred_series: pd.Series) -> pd.Series:
                pred_series = pred_series.fillna("").astype(str).str.strip()
                orig_series = orig_series.fillna("").astype(str).str.strip()
                return np.where(
                    pred_series.eq(""),
                    "未定位",
                    np.where(
                        orig_series.eq(""),
                        "无原SOP标签",
                        np.where(pred_series.eq(orig_series), "一致", "不一致")
                    )
                )

            merged["自动定位-与原SOP一级一致"] = compare(
                merged.get(level1_col, pd.Series("", index=merged.index)),
                merged.get(INFER_COLUMN_MAP["predicted_level1"], pd.Series("", index=merged.index)),
            )
            merged["自动定位-与原SOP一致"] = compare(
                merged.get(level2_col, pd.Series("", index=merged.index)),
                merged.get(INFER_COLUMN_MAP["predicted_level2"], pd.Series("", index=merged.index)),
            )

    print(f"写入结果：{output_path}")
    merged.to_excel(output_path, index=False)
    print("完成。")
    return 0


def run_verify_mode(
    df: pd.DataFrame,
    resolver: SOPReferenceResolver,
    sales_keywords: Sequence[str],
    history_col: str,
    level1_col: str,
    level2_col: str,
    similarity_threshold: float,
) -> pd.DataFrame:
    results: List[Dict[str, Optional[str]]] = []
    total_rows = len(df)
    for idx, row in df.iterrows():
        level1 = row.get(level1_col)
        level2 = row.get(level2_col)
        history_text = row.get(history_col)
        result: Dict[str, Optional[str]] = {
            "match_status": False,
            "match_reason": "",
            "matched_sales_message": "",
            "matched_sales_time": "",
            "matched_reference_text": "",
            "matched_reference_source": "",
            "matched_reference_level3": "",
            "matched_similarity": None,
            "matched_turn_from_last": None,
            "matched_round_from_last": None,
            "match_hit": "否",
            "latest_sales_message": "",
            "latest_sales_time": "",
        }

        if pd.isna(level1) or pd.isna(level2):
            result["match_reason"] = "缺少SOP节点标签"
            results.append(result)
            continue
        if pd.isna(history_text) or not str(history_text).strip():
            result["match_reason"] = "缺少历史对话"
            results.append(result)
            continue

        last_msg, last_time = _extract_last_sales_entry(str(history_text), sales_keywords)
        result["latest_sales_message"] = last_msg
        result["latest_sales_time"] = last_time

        references = resolver.get_references(str(level1), str(level2))
        matched = find_sales_match(
            str(history_text),
            references,
            sales_keywords,
            similarity_threshold=similarity_threshold,
            resolver=resolver,
        )
        result.update(matched)
        result["match_hit"] = "是" if result.get("match_status") else "否"
        results.append(result)

        if (idx + 1) % 500 == 0 or idx + 1 == total_rows:
            print(f"已处理 {idx + 1}/{total_rows} 行")

    return pd.DataFrame(results)


def run_infer_mode(
    df: pd.DataFrame,
    resolver: SOPReferenceResolver,
    sales_keywords: Sequence[str],
    history_col: str,
    similarity_threshold: float,
) -> pd.DataFrame:
    all_refs = resolver.all_references()
    if not all_refs:
        raise RuntimeError("SOP逻辑树缺少参考话术，无法执行自动识别。")

    results: List[Dict[str, Optional[str]]] = []
    total_rows = len(df)
    for idx, row in df.iterrows():
        history_text = row.get(history_col)
        result: Dict[str, Optional[str]] = {
            "sop_match_found": False,
            "failure_reason": "",
            "predicted_level1": "",
            "predicted_level2": "",
            "matched_reference_level3": "",
            "matched_reference_text": "",
            "matched_reference_source": "",
            "matched_similarity": None,
            "matched_sales_message": "",
            "matched_sales_time": "",
            "matched_turn_from_last": None,
            "matched_round_from_last": None,
            "latest_sales_message": "",
            "latest_sales_time": "",
        }

        if pd.isna(history_text) or not str(history_text).strip():
            result["failure_reason"] = "缺少历史对话"
            results.append(result)
            continue

        last_msg, last_time = _extract_last_sales_entry(str(history_text), sales_keywords)
        result["latest_sales_message"] = last_msg
        result["latest_sales_time"] = last_time

        matched = identify_node_from_history(
            str(history_text),
            resolver,
            all_refs,
            sales_keywords,
            similarity_threshold,
        )
        result.update(matched)
        results.append(result)

        if (idx + 1) % 500 == 0 or idx + 1 == total_rows:
            print(f"已处理 {idx + 1}/{total_rows} 行")

    return pd.DataFrame(results)


if __name__ == "__main__":
    sys.exit(main())
