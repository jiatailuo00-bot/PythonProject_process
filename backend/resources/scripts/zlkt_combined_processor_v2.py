import argparse
import ast
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------- 通用工具 ----------

def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return True
    if text in {"[]", "{}", "nan", "None"}:
        return True
    return False


def parse_time(value: Any) -> Optional[pd.Timestamp]:
    if is_blank(value):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0]
    if pd.isna(ts):
        return None
    return ts.floor("s")


# ---------- SOP 传播 ----------

def _safe_eval(text: str) -> Optional[Any]:
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            continue
    return None


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "nan"}:
            return ""
        return stripped
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _iterate_values(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for value in obj.values():
            yield value
            yield from _iterate_values(value)
    elif isinstance(obj, list):
        for item in obj:
            yield item
            yield from _iterate_values(item)


def _extract_from_thought_unit(raw: Any) -> Tuple[str, str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "", ""

    if isinstance(raw, str):
        candidate = raw.strip()
        if not candidate or candidate.lower() in {"nan", "none"}:
            return "", ""
        parsed = _safe_eval(candidate)
    else:
        parsed = raw

    if parsed is None:
        return "", ""

    if isinstance(parsed, dict):
        sources = [parsed]
    elif isinstance(parsed, list):
        sources = list(parsed)
    else:
        return "", ""

    task_theme = ""
    sub_task_theme = ""

    target_keys = {
        "task_theme": {"task_theme", "taskTheme", "task_theme_name"},
        "sub_task_theme": {
            "sub_task_theme",
            "sub_task_name",
            "subTaskTheme",
            "subTaskName",
        },
    }

    def lookup(container: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
        for key in keys:
            if key in container and container[key] not in (None, "", [], {}):
                return container[key]
        return None

    for item in sources:
        if not isinstance(item, dict):
            continue
        if not task_theme:
            value = lookup(item, target_keys["task_theme"])
            if value is None:
                for nested in _iterate_values(item):
                    if isinstance(nested, dict):
                        value = lookup(nested, target_keys["task_theme"])
                        if value is not None:
                            break
            if value is not None:
                task_theme = _normalize_value(value)

        if not sub_task_theme:
            value = lookup(item, target_keys["sub_task_theme"])
            if value is None:
                for nested in _iterate_values(item):
                    if isinstance(nested, dict):
                        value = lookup(nested, target_keys["sub_task_theme"])
                        if value is not None:
                            break
            if value is not None:
                sub_task_theme = _normalize_value(value)

        if task_theme and sub_task_theme:
            break

    return task_theme, sub_task_theme


# ---------- SOP Schema 补充 ----------

SCHEMA_FILE = Path("之了SOP_schema-全局任务感知_final_new.json")


def _normalize_key(value: Any) -> str:
    if is_blank(value):
        return ""
    return str(value).strip()


def _generate_key_variants(value: str) -> Iterable[str]:
    stripped = value.strip()
    yield stripped
    yield stripped.replace(" ", "")
    yield stripped.replace(" ", "").replace("．", ".")
    yield stripped.replace("．", ".")


SECOND_FOLLOWUP_KEYWORD = "二次跟进"


def _is_second_followup_node(value: Any) -> bool:
    normalized = _normalize_value(value)
    return bool(normalized) and SECOND_FOLLOWUP_KEYWORD in normalized


def _select_schema_entry(
    candidates: Iterable[Dict[str, str]], preferred_task: Optional[str] = None
) -> Optional[Dict[str, str]]:
    candidates = list(candidates)
    if not candidates:
        return None

    if preferred_task:
        preferred_task_norm = _normalize_value(preferred_task)
        for entry in candidates:
            if _normalize_value(entry.get("task_theme")) == preferred_task_norm:
                return entry

    return candidates[0]


def _get_schema_entry_by_theme(
    mapping: Dict[str, List[Dict[str, str]]], theme: Any, preferred_task: Optional[str] = None
) -> Optional[Dict[str, str]]:
    normalized = _normalize_value(theme)
    if not normalized:
        return None
    for variant in _generate_key_variants(normalized):
        entries = mapping.get(variant)
        if not entries:
            continue
        return _select_schema_entry(entries, preferred_task)
    return None


def normalize_second_followup_ids(
    df: pd.DataFrame,
    sop_col: str = "SOP二级节点",
    sop_id_col: str = "SOP二级节点ID",
) -> pd.DataFrame:
    if sop_col not in df.columns or sop_id_col not in df.columns:
        return df

    mask = df[sop_col].apply(_is_second_followup_node)
    if not mask.any():
        return df

    def adjust_id(raw: Any) -> str:
        if is_blank(raw):
            return ""
        text = str(raw).strip()
        if "-" in text:
            return text.split("-", 1)[0].strip()
        return text

    df.loc[mask, sop_id_col] = df.loc[mask, sop_id_col].apply(adjust_id)
    return df


@lru_cache(maxsize=1)
def load_sop_schema_mapping(schema_path: Path = SCHEMA_FILE) -> Dict[str, List[Dict[str, str]]]:
    if not schema_path.exists():
        raise FileNotFoundError(f"未找到SOP schema文件: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("任务树", [])
    mapping: Dict[str, List[Dict[str, str]]] = {}

    def register(sub_theme: Optional[str], info: Dict[str, str]) -> None:
        if not sub_theme:
            return
        normalized = _normalize_key(sub_theme)
        if not normalized:
            return
        for key in _generate_key_variants(normalized):
            mapping.setdefault(key, []).append(info)

    for task in tasks:
        task_theme = _normalize_key(task.get("任务主题"))
        task_id = _normalize_key(task.get("ID") or task.get("任务ID"))
        sub_tasks = task.get("子任务") or task.get("子任务列表") or []
        if not isinstance(sub_tasks, list):
            continue

        prev_entry: Optional[Dict[str, str]] = None

        for sub in sub_tasks:
            if not isinstance(sub, dict):
                continue
            sub_theme = _normalize_key(sub.get("子任务主题"))
            sub_theme_alt = _normalize_key(sub.get("子任务主题(改写)"))
            sub_id = _normalize_key(sub.get("子任务ID") or sub.get("id"))
            info = {
                "task_theme": task_theme,
                "task_id": task_id,
                "sub_task_theme": sub_theme or sub_theme_alt,
                "sub_task_id": sub_id,
                "prev_sub_task_theme": "",
                "prev_sub_task_id": "",
            }
            if prev_entry:
                info["prev_sub_task_theme"] = prev_entry.get("sub_task_theme", "")
                info["prev_sub_task_id"] = prev_entry.get("sub_task_id", "")
            register(sub_theme, info)
            register(sub_theme_alt, info)
            prev_entry = info

    return mapping


def enrich_sop_from_schema(df: pd.DataFrame) -> pd.DataFrame:
    try:
        mapping = load_sop_schema_mapping()
    except FileNotFoundError as exc:
        print(str(exc))
        return df

    if "SOP一级节点" not in df.columns:
        df["SOP一级节点"] = ""
    if "SOP二级节点" not in df.columns:
        df["SOP二级节点"] = ""

    for col in ["SOP一级节点ID", "SOP二级节点ID"]:
        if col not in df.columns:
            df[col] = ""

    for idx, row in df.iterrows():
        sub_theme_raw = row.get("SOP二级节点", "")
        sub_theme_norm = _normalize_key(sub_theme_raw)
        if not sub_theme_norm:
            continue

        schema_entry = None
        for key_variant in _generate_key_variants(sub_theme_norm):
            entries = mapping.get(key_variant, [])
            if not entries:
                continue
            schema_entry = _select_schema_entry(entries, row.get("SOP一级节点"))
            if schema_entry:
                break

        if not schema_entry:
            continue

        if is_blank(row.get("SOP一级节点")) and schema_entry.get("task_theme"):
            df.at[idx, "SOP一级节点"] = schema_entry["task_theme"]

        if schema_entry.get("task_id"):
            df.at[idx, "SOP一级节点ID"] = schema_entry["task_id"]

        if schema_entry.get("sub_task_id"):
            df.at[idx, "SOP二级节点ID"] = schema_entry["sub_task_id"]

        # Ensure二级节点文本与schema一致（优先使用schema提供的主题）
        if schema_entry.get("sub_task_theme"):
            df.at[idx, "SOP二级节点"] = schema_entry["sub_task_theme"]

    return normalize_second_followup_ids(df)


ANCHOR_SENDERS = {"CSM", "未发送"}
PROPAGATION_SENDERS = {"客户", "CSM", "未发送"}
BACKFILL_PRE_ANCHOR_SENDERS = {"客户", "CSM", "未发送"}


def filter_conversations_without_terminal_csm(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    if "发送方" not in df.columns:
        return df, 0, 0

    drop_indices: List[int] = []
    trimmed_pairs = 0
    removed_pairs = 0

    grouped = df.groupby(["销售ID", "客户ID", "外部客户ID"], dropna=False, sort=False)
    for _, sub_df in grouped:
        senders = sub_df["发送方"].astype(str).fillna("")
        indices = sub_df.index.to_list()
        csm_positions = np.where(senders == "CSM")[0]
        if csm_positions.size == 0:
            drop_indices.extend(indices)
            removed_pairs += 1
            continue
        unsent_positions = np.where(senders == "未发送")[0]
        last_csm = csm_positions[-1]
        last_unsent = unsent_positions[-1] if unsent_positions.size else -1
        last_pos = max(last_csm, last_unsent)
        tail_indices = indices[last_pos + 1 :]
        if not tail_indices:
            continue
        drop_indices.extend(tail_indices)
        trimmed_pairs += 1

    if not drop_indices:
        return df, trimmed_pairs, removed_pairs

    filtered_df = df.drop(index=drop_indices)
    return filtered_df, trimmed_pairs, removed_pairs


def _resolve_previous_sop_info(
    current_sop1: Any, current_sop2: Any, current_display: Any
) -> Optional[Dict[str, str]]:
    mapping = load_sop_schema_mapping()

    candidates = []
    if not is_blank(current_sop2):
        candidates.append(_normalize_key(current_sop2))
    if not is_blank(current_display):
        candidates.append(_normalize_key(current_display))
    if not is_blank(current_sop1):
        candidates.append(_normalize_key(current_sop1))

    for candidate in candidates:
        if not candidate:
            continue
        entry = _get_schema_entry_by_theme(mapping, candidate, preferred_task=_normalize_value(current_sop1))
        if not entry:
            continue

        prev_entry = entry
        visited: set[str] = set()
        while True:
            prev_theme = _normalize_value(prev_entry.get("prev_sub_task_theme"))
            prev_id = prev_entry.get("prev_sub_task_id", "")
            if not prev_theme:
                break

            normalized_prev = _normalize_key(prev_theme)
            if normalized_prev in visited:
                break
            visited.add(normalized_prev)

            if _is_second_followup_node(prev_theme):
                next_entry = _get_schema_entry_by_theme(
                    mapping, prev_theme, preferred_task=_normalize_value(entry.get("task_theme"))
                )
                if not next_entry:
                    prev_entry = {}
                    break
                prev_entry = next_entry
                continue

            resolved_entry = _get_schema_entry_by_theme(
                mapping, prev_theme, preferred_task=_normalize_value(entry.get("task_theme"))
            ) or {
                "task_theme": entry.get("task_theme"),
                "task_id": entry.get("task_id", ""),
                "sub_task_theme": prev_theme,
                "sub_task_id": prev_id,
            }
            return {
                "task_theme": _normalize_value(resolved_entry.get("task_theme")),
                "task_id": resolved_entry.get("task_id", ""),
                "sub_task_theme": _normalize_value(resolved_entry.get("sub_task_theme")),
                "sub_task_id": resolved_entry.get("sub_task_id", ""),
            }

    return None


def _backfill_pre_anchor_rows(
    processed: pd.DataFrame,
    indices: List[int],
    previous_info: Dict[str, str],
    sop_col_1: str,
    sop_col_2: str,
    sop_display_col: str,
) -> int:
    if not indices or not previous_info:
        return 0

    prev_task = _normalize_value(previous_info.get("task_theme"))
    prev_sub = _normalize_value(previous_info.get("sub_task_theme"))
    prev_display = prev_sub or prev_task

    updated_rows = 0

    for idx in reversed(indices):
        sender = processed.at[idx, "发送方"]
        if sender not in BACKFILL_PRE_ANCHOR_SENDERS:
            continue

        has_structured = (
            not is_blank(processed.at[idx, sop_col_1])
            or not is_blank(processed.at[idx, sop_col_2])
        )
        has_display = not is_blank(processed.at[idx, sop_display_col])
        if has_structured or has_display:
            break

        row_updated = False

        if prev_task and is_blank(processed.at[idx, sop_col_1]):
            processed.at[idx, sop_col_1] = prev_task
            row_updated = True

        if prev_sub and is_blank(processed.at[idx, sop_col_2]):
            processed.at[idx, sop_col_2] = prev_sub
            row_updated = True

        if prev_display and is_blank(processed.at[idx, sop_display_col]):
            processed.at[idx, sop_display_col] = prev_display
            row_updated = True

        if row_updated:
            updated_rows += 1

    return updated_rows


def propagate_sop_nodes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    required = {"销售ID", "客户ID", "发送方", "thought_unit"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要列: {', '.join(sorted(missing))}")

    processed = df.copy()
    sop_col_1 = "SOP一级节点"
    sop_col_2 = "SOP二级节点"
    sop_display_col = "SOP节点"

    if sop_display_col not in processed.columns:
        processed[sop_display_col] = ""

    orig_sop1 = processed[sop_col_1].copy() if sop_col_1 in processed.columns else None
    orig_sop2 = processed[sop_col_2].copy() if sop_col_2 in processed.columns else None
    orig_sop_display = (
        processed[sop_display_col].copy()
        if sop_display_col in processed.columns
        else None
    )

    extracted = processed["thought_unit"].apply(_extract_from_thought_unit)
    extracted_df = pd.DataFrame(extracted.tolist(), columns=[sop_col_1, sop_col_2])

    if sop_col_1 in processed.columns:
        processed.drop(columns=[sop_col_1], inplace=True)
    if sop_col_2 in processed.columns:
        processed.drop(columns=[sop_col_2], inplace=True)

    processed = pd.concat([processed, extracted_df], axis=1)

    if orig_sop1 is not None:
        orig_sop1_norm = orig_sop1.apply(_normalize_value)
        mask = processed[sop_col_1].apply(is_blank) & ~orig_sop1_norm.apply(is_blank)
        processed.loc[mask, sop_col_1] = orig_sop1_norm[mask]

    if orig_sop2 is not None:
        orig_sop2_norm = orig_sop2.apply(_normalize_value)
        mask = processed[sop_col_2].apply(is_blank) & ~orig_sop2_norm.apply(is_blank)
        processed.loc[mask, sop_col_2] = orig_sop2_norm[mask]

    if orig_sop_display is not None:
        orig_sop_display_norm = orig_sop_display.apply(_normalize_value)
        mask = processed[sop_col_2].apply(is_blank) & ~orig_sop_display_norm.apply(
            is_blank
        )
        processed.loc[mask, sop_col_2] = orig_sop_display_norm[mask]

    processed["_sop_anchor"] = False
    processed["_row_order"] = range(len(processed))

    sop_anchor_count = 0
    propagated_count = 0

    for (sales_id, customer_id), indices in processed.groupby(["销售ID", "客户ID"]).groups.items():
        ordered_indices = sorted(indices, key=lambda idx: processed.at[idx, "_row_order"])

        current_sop1 = ""
        current_sop2 = ""
        current_tu = None
        current_display = ""

        has_anchor = False

        for position, idx in enumerate(ordered_indices):
            sender = processed.at[idx, "发送方"]
            sop1 = processed.at[idx, sop_col_1]
            sop2 = processed.at[idx, sop_col_2]
            display_value = processed.at[idx, sop_display_col]
            has_structured = not is_blank(sop1) or not is_blank(sop2)
            has_display = not is_blank(display_value)
            has_sop = has_structured or has_display

            is_anchor = sender in ANCHOR_SENDERS and has_sop

            if is_anchor:
                if not has_anchor:
                    previous_info = _resolve_previous_sop_info(sop1, sop2, display_value)
                    if previous_info:
                        propagated_count += _backfill_pre_anchor_rows(
                            processed,
                            ordered_indices[:position],
                            previous_info,
                            sop_col_1,
                            sop_col_2,
                            sop_display_col,
                        )

                current_sop1 = sop1 if has_structured else ""
                current_sop2 = sop2 if has_structured else ""
                current_display = display_value if has_display else current_sop2
                current_tu = processed.at[idx, "thought_unit"]
                processed.at[idx, "_sop_anchor"] = True
                sop_anchor_count += 1
                has_anchor = True
                continue

            if not current_sop1 and not current_sop2 and not current_display:
                continue

            if sender in PROPAGATION_SENDERS and not has_sop:
                updated = False

                if not is_blank(current_sop1) and is_blank(processed.at[idx, sop_col_1]):
                    processed.at[idx, sop_col_1] = current_sop1
                    updated = True

                if not is_blank(current_sop2) and is_blank(processed.at[idx, sop_col_2]):
                    processed.at[idx, sop_col_2] = current_sop2
                    updated = True

                if not is_blank(current_display) and is_blank(display_value):
                    processed.at[idx, sop_display_col] = current_display
                    updated = True

                if updated:
                    propagated_count += 1

    processed.drop(columns=["_sop_anchor", "_row_order"], inplace=True)

    # reorder columns
    cols = list(processed.columns)
    for col in (sop_col_1, sop_col_2):
        if col in cols:
            cols.remove(col)
    if sop_display_col in cols:
        insert_at = cols.index(sop_display_col) + 1
    else:
        insert_at = len(cols)
    cols[insert_at:insert_at] = [sop_col_1, sop_col_2]
    processed = processed[cols]
    processed = normalize_second_followup_ids(processed, sop_col_2, "SOP二级节点ID")

    stats = {
        "sop_anchor_rows": sop_anchor_count,
        "propagated_rows": propagated_count,
    }

    return processed, stats


# ---------- 客户测试集生成 ----------

CUSTOMER_SENDER = "客户"


def standardize_conversation(conversation_text: Any) -> str:
    if is_blank(conversation_text):
        return ""

    conversation = str(conversation_text)
    lines = conversation.split("\n")
    normalized = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("[客户]"):
            normalized.append(line)
            continue

        if line.startswith("["):
            bracket_end = line.find("]")
            if bracket_end != -1:
                remaining = line[bracket_end + 1 :]
                if remaining.startswith("[") and "]: " in remaining:
                    time_end = remaining.find("]:")
                    if time_end != -1:
                        time_part = remaining[: time_end + 1]
                        content = remaining[time_end + 2 :]
                        normalized.append(f"[销售]{time_part}: {content}")
                        continue
                if remaining.startswith(":"):
                    normalized.append(f"[销售]: {remaining[1:].strip()}")
                    continue
                normalized.append(f"[销售]: {remaining.strip()}")
                continue

        normalized.append(line)

    return "\n".join(normalized)


def build_complete_conversation(row: pd.Series) -> str:
    history = standardize_conversation(row.get("历史对话", ""))
    customer_msg = row.get("客户消息", "")
    timestamp = row.get("客户消息时间", "")

    if is_blank(customer_msg):
        return history

    customer_msg = str(customer_msg).strip()
    if is_blank(timestamp):
        customer_line = f"[客户]: {customer_msg}"
    else:
        customer_line = f"[客户][{timestamp}]: {customer_msg}"

    if history:
        return f"{history}\n{customer_line}"
    return customer_line


QUESTION_INDICATORS = [
    "？",
    "?",
    "什么",
    "怎么",
    "如何",
    "为什么",
    "哪个",
    "哪里",
    "哪些",
    "谁",
    "何时",
    "多少",
    "啥",
    "咋",
    "咋样",
    "怎样",
    "几",
    "多久",
    "多长",
    "是什么",
    "是谁",
    "是哪",
    "怎么样",
    "好吗",
    "行吗",
    "可以吗",
    "不是吗",
    "对吗",
    "吗",
    "呢",
    "不",
    "吧",
]

QUOTE_SEPARATOR = "- - - - - - - - - - - - - - -"


def is_question(message: Any) -> str:
    if is_blank(message):
        return "否"
    text = str(message).strip()

    candidate = text
    if QUOTE_SEPARATOR in text:
        trailing = text.split(QUOTE_SEPARATOR)[-1].strip()
        candidate = trailing

    if len(candidate) <= 1:
        return "否"

    if "？" in candidate or "?" in candidate:
        return "是"

    for indicator in QUESTION_INDICATORS:
        if candidate.endswith(indicator) or indicator in candidate:
            return "是"

    return "否"


TARGET_SOP_LEVEL1_NODES = [
    "进一步了解客户的背景信息(进一步沟通)",
    "提供基础服务",
    "基础信息了解",
    "获取客户备考目的(挖需/扩需)\n同时强化客户的考证需求(强需)",
]
SAMPLE_SIZE_TOTAL = 400


def compute_sop_level1_distribution(df: pd.DataFrame) -> Counter:
    if "SOP一级节点" not in df.columns:
        return Counter()
    normalized = df["SOP一级节点"].apply(_normalize_value)
    valid = [value for value in normalized if value]
    return Counter(valid)


def allocate_samples_by_distribution(
    distribution: Counter,
    target_nodes: List[str],
    total_samples: int,
) -> Dict[str, int]:
    available_counts = {node: distribution.get(node, 0) for node in target_nodes}
    filtered_counts = {node: count for node, count in available_counts.items() if count > 0}

    if not filtered_counts or total_samples <= 0:
        return {node: 0 for node in target_nodes}

    total_pool = sum(filtered_counts.values())
    if total_pool == 0:
        return {node: 0 for node in target_nodes}

    allocations: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []

    assigned = 0

    for node, count in filtered_counts.items():
        ideal = total_samples * (count / total_pool)
        base = int(ideal)
        base = min(base, count)
        allocations[node] = base
        assigned += base
        residual_capacity = count - base
        remainder = ideal - base if residual_capacity > 0 else -1.0
        remainders.append((remainder, node))

    remaining = total_samples - assigned

    if remaining > 0:
        remainders.sort(reverse=True)
        for remainder, node in remainders:
            if remaining <= 0:
                break
            if remainder <= 0:
                continue
            capacity = available_counts[node] - allocations[node]
            if capacity <= 0:
                continue
            take = min(capacity, remaining)
            allocations[node] += take
            remaining -= take

    for node in target_nodes:
        allocations.setdefault(node, 0)
        allocations[node] = min(allocations[node], available_counts.get(node, 0))

    return allocations


def sample_cases_by_sop(
    df: pd.DataFrame,
    allocations: Dict[str, int],
    sample_seed: int = 42,
) -> pd.DataFrame:
    sampled_frames = []

    for node, quota in allocations.items():
        if quota <= 0:
            continue
        subset = df[df["SOP一级节点"] == node]
        if subset.empty:
            continue
        take = min(quota, len(subset))
        sampled = subset.sample(n=take, random_state=sample_seed)
        sampled_frames.append(sampled)

    if not sampled_frames:
        return pd.DataFrame(columns=df.columns)

    combined = pd.concat(sampled_frames, ignore_index=True)
    combined.sort_values(["销售ID", "客户ID", "客户消息时间", "发送时间"], inplace=True, ignore_index=True)
    return combined


def propagate_rag_and_thought(df: pd.DataFrame) -> pd.DataFrame:
    propagated = df.copy()

    if "rag传播来源" not in propagated.columns:
        propagated["rag传播来源"] = ""
    if "thought_unit_source" not in propagated.columns:
        propagated["thought_unit_source"] = ""

    orig_rag_mask = ~propagated["rag"].apply(is_blank)
    propagated.loc[orig_rag_mask & propagated["rag传播来源"].apply(is_blank), "rag传播来源"] = "原始"

    orig_thought_mask = ~propagated["thought_unit"].apply(is_blank)
    propagated.loc[
        orig_thought_mask & propagated["thought_unit_source"].apply(is_blank), "thought_unit_source"
    ] = "原始"

    propagated["_row_order"] = range(len(propagated))

    for (_, _), group_df in propagated.groupby(["销售ID", "客户ID"]):
        group_df = group_df.copy()
        group_df["_timestamp"] = group_df["客户消息时间"].apply(parse_time)

        producer_series = (
            group_df["消息生成主体"].astype(str).str.strip().str.upper()
            if "消息生成主体" in group_df.columns
            else pd.Series("", index=group_df.index)
        )

        csm_rows = group_df[(group_df["发送方"] == "CSM") & (producer_series == "SA")]
        unsent_rows = group_df[(group_df["发送方"] == "未发送") & (producer_series == "SA")]

        for idx in group_df[group_df["发送方"] == CUSTOMER_SENDER].index:
            timestamp = group_df.at[idx, "_timestamp"]
            if timestamp is None:
                continue

            anchor_row = None
            anchor_source = None

            csm_matches = csm_rows[csm_rows["_timestamp"] == timestamp]
            if not csm_matches.empty:
                anchor_row = csm_matches.sort_values("_row_order").iloc[0]
                anchor_source = "CSM"
            else:
                unsent_matches = unsent_rows[unsent_rows["_timestamp"] == timestamp]
                if not unsent_matches.empty:
                    anchor_row = unsent_matches.sort_values("_row_order").iloc[0]
                    anchor_source = "未发送"

            if anchor_row is None:
                continue

            anchor_rag = anchor_row["rag"]
            anchor_thought = anchor_row["thought_unit"]

            if is_blank(propagated.at[idx, "rag"]) and not is_blank(anchor_rag):
                propagated.at[idx, "rag"] = anchor_rag
                propagated.at[idx, "rag传播来源"] = anchor_source

            if not is_blank(anchor_thought):
                propagated.at[idx, "thought_unit"] = anchor_thought
                propagated.at[idx, "thought_unit_source"] = anchor_source

    propagated.drop(columns=["_row_order"], inplace=True)
    return propagated


def propagate_ai_fields(df: pd.DataFrame) -> pd.DataFrame:
    propagated = df.copy()
    ai_fields = ["AI生成消息", "发送消息内容", "AI与审核结果是否一致"]
    for field in ai_fields:
        if field not in propagated.columns:
            propagated[field] = ""

    grouped = propagated.groupby(
        ["销售ID", "客户ID", "外部客户ID"], dropna=False, sort=False
    )
    for _, group_df in grouped:
        csm_rows = group_df[group_df["发送方"] == "CSM"]
        if csm_rows.empty:
            continue

        for idx in group_df[group_df["发送方"] == CUSTOMER_SENDER].index:
            customer_time = propagated.at[idx, "客户消息时间"]
            if pd.isna(customer_time) or customer_time == "":
                continue
            matches = csm_rows[csm_rows["客户消息时间"] == customer_time]
            if matches.empty:
                continue
            merged_values: Dict[str, List[str]] = {field: [] for field in ai_fields}
            for _, source_row in matches.iterrows():
                for field in ai_fields:
                    value = source_row.get(field, "")
                    if pd.isna(value):
                        continue
                    value_str = str(value).strip()
                    if value_str:
                        merged_values[field].append(value_str)
            for field in ai_fields:
                if merged_values[field]:
                    propagated.at[idx, field] = "；".join(merged_values[field])

    return propagated


def safe_parse_thought_unit(raw: Any) -> Any:
    if is_blank(raw):
        return ""
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            continue
    return text


def build_customer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = propagate_rag_and_thought(df)
    df = propagate_ai_fields(df)

    df["完整历史对话"] = df.apply(build_complete_conversation, axis=1)
    df["最新客户消息"] = df["客户消息"].apply(lambda x: "" if is_blank(x) else str(x).strip())
    df["是否是问句"] = df["最新客户消息"].apply(is_question)
    df["解析thought_unit"] = df["thought_unit"].apply(safe_parse_thought_unit)

    columns = [
        "销售ID",
        "销售名称",
        "客户ID",
        "客户名称",
        "source_file",
        "客户消息时间",
        "发送时间",
        "最新客户消息",
        "完整历史对话",
        "SOP一级节点",
        "SOP二级节点",
        "SOP一级节点ID",
        "SOP二级节点ID",
        "SOP节点",
        "是否是问句",
        "rag",
        "rag传播来源",
        "thought_unit",
        "解析thought_unit",
        "thought_unit_source",
        "AI生成消息",
        "发送消息内容",
        "AI与审核结果是否一致",
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = ""

    result = df[df["发送方"] == CUSTOMER_SENDER].copy()
    if "历史对话" in result.columns:
        result = result[~result["历史对话"].apply(is_blank)]
    result = result[result["最新客户消息"] != ""]
    result = result[~result["SOP二级节点"].apply(is_blank)]
    result = result[columns]
    result.sort_values(["销售ID", "客户ID", "客户消息时间", "发送时间"], inplace=True, ignore_index=True)
    return result


# ---------- 主流程 ----------

def run_pipeline(input_file: Path) -> None:
    print(f"读取原始数据: {input_file}")
    original_df = pd.read_excel(input_file)

    original_df, trimmed_pairs, removed_pairs = filter_conversations_without_terminal_csm(original_df)
    if removed_pairs > 0:
        print(f"移除 {removed_pairs} 个完全没有 CSM 的会话。")
    if trimmed_pairs > 0:
        print(f"对 {trimmed_pairs} 个会话删除了尾部仅客户/销售的片段。")
    print(f"过滤完成，剩余 {len(original_df)} 条记录。")

    sop_output = input_file.with_name(f"{input_file.stem}_preprocessed_sop.xlsx")
    customer_output = input_file.with_name(f"{input_file.stem}_customer_dataset.xlsx")

    print("步骤 1: 传播 SOP 节点...")
    sop_df, sop_stats = propagate_sop_nodes(original_df)
    sop_df = enrich_sop_from_schema(sop_df)
    sop_df.to_excel(sop_output, index=False)
    print(f"SOP 传播完成，结果保存至: {sop_output}")
    print(
        f"  锚点行数: {sop_stats.get('sop_anchor_rows', 0)}, "
        f"传播补全行数: {sop_stats.get('propagated_rows', 0)}"
    )

    print("步骤 2: 生成客户测试集...")
    customer_df = build_customer_dataset(sop_df)
    customer_df.to_excel(customer_output, index=False)
    print(f"客户测试集完成，结果保存至: {customer_output}")
    print(f"  客户记录数: {len(customer_df)}")
    sop_count = customer_df["SOP节点"].replace({None: "", float("nan"): ""}).astype(str).str.strip()
    rag_count = customer_df["rag"].replace({None: "", float("nan"): ""}).astype(str).str.strip()
    print(f"  含 SOP 节点客户数: {(sop_count != '').sum()}")
    print(f"  含 rag 客户数: {(rag_count != '').sum()}")

    print("步骤 3: 统计并抽取重点 SOP 案例...")
    level1_distribution = compute_sop_level1_distribution(customer_df)
    print("  SOP 一级节点分布（客户测试集）:")
    for node, count in level1_distribution.most_common():
        print(f"    {node}: {count} 条")

    target_allocations = allocate_samples_by_distribution(
        level1_distribution,
        TARGET_SOP_LEVEL1_NODES,
        SAMPLE_SIZE_TOTAL,
    )

    focus_df = customer_df[customer_df["SOP一级节点"].isin(TARGET_SOP_LEVEL1_NODES)].copy()
    sampled_df = sample_cases_by_sop(focus_df, target_allocations)

    sample_output = input_file.with_name(f"{input_file.stem}_customer_dataset_sampled.xlsx")
    sampled_df.to_excel(sample_output, index=False)

    planned_total = sum(target_allocations.values())
    actual_total = len(sampled_df)
    print(f"  目标抽取总数: {SAMPLE_SIZE_TOTAL}")
    print(f"  计划配额总和: {planned_total}")
    print(f"  实际抽取总数: {actual_total}")
    for node in TARGET_SOP_LEVEL1_NODES:
        print(
            f"    {node}: 计划 {target_allocations.get(node, 0)} 条, 实际 {len(sampled_df[sampled_df['SOP一级节点'] == node])} 条"
        )
    print(f"重点案例已保存至: {sample_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SOP 传播 + 客户测试集生成 一体化脚本")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="zlkt_all_2025-10-31_to_2025-11-04_combined.xlsx",
        help="原始合并后的 zlkt Excel 文件路径（可选，默认使用同目录示例文件）",
    )

    args = parser.parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")

    run_pipeline(input_path)


if __name__ == "__main__":
    main()
