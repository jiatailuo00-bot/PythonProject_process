# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python data processing project focused on sales conversation analysis and SOP (Standard Operating Procedure) matching. The system analyzes sales dialogues from Excel files and matches them against predefined SOP logic trees to categorize and label conversations.

## Core Scripts and Dependencies

### Main Scripts
- `get_sop_pip.py` - Core SOP matching engine with conversation analysis functions
- `get_sop_improved_la_new.py` - Improved SOP identification using enhanced validation logic
- `get_sop_pip_improved.py` - Enhanced version with targeted SOP matching
- `get_sop_pip_improved_v2.py` - Second iteration of improved matching algorithm
- `main.py` - Basic template script (minimal functionality)

### Key Functions
- `calculate_sentence_similarity()` - Chinese text similarity using jieba tokenization
- `parse_conversation_history()` - Extracts conversation data from various formats
- `find_nearest_sales_sop_match()` - Locates matching SOP nodes for sales messages
- `analyze_conversation_sop_labels()` - Main analysis pipeline for conversation labeling
- `list_of_dicts_to_xlsx()` - Converts structured data to Excel format

### Dependencies
Core libraries: pandas, jieba, tqdm, openpyxl
All scripts are interconnected with shared utility functions imported from `get_sop_pip.py`

## Data Flow

1. Input: Excel files containing sales conversations with columns like 'ID', '发送消息角色', '处理后对话内容', '对话时间'
2. Processing: Scripts analyze conversations and match against SOP logic tree (JSON format)
3. Output: Labeled Excel files with SOP node classifications and similarity scores

## SOP Logic Tree Structure

The `chengla_wx.json` file contains hierarchical SOP definitions with:
- Reference scripts (参考话术)
- Similar scripts (相似话术)
- Expected responses (预期话术)
- Next actions (下一步动作)
- Multi-level nested categories

## Running Scripts

Most scripts follow this pattern:
```python
# Example execution pattern
config_data = {
    "corpus_dir": "path/to/input.xlsx",
    "pipeline_case_path": "path/to/output.xlsx",
    "sop_logic_tree": "path/to/chengla_wx.json",
    "functions": [{"name": "script_name.func_main", "similarity": 0.90}]
}

script.func_main(config_data=config_data)
```

## File Processing Notes

- Scripts expect Excel files with specific Chinese column headers
- Similarity thresholds typically range from 0.90-0.95
- Batch processing is supported for large datasets
- Output includes detailed matching statistics and confidence scores