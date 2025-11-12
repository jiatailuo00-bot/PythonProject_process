import pandas as pd
import re
import json
import ast
from collections import defaultdict
import sys
import os

# 添加zlkt_action目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'zlkt_action'))

# 取消模型调用，统一使用关键字方法
MODEL_AVAILABLE = False
llm_qwen3_32b = None
print("?? 已禁用模型问句识别，统一使用关键字方法")

def _normalize_user_info_content(content):
    """将用户信息库content统一转换为列表[dict]形式"""
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        return [content]
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return []
        # 优先尝试JSON解析
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        # 回退到安全的字面量解析（处理单引号格式）
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []

def read_and_process_excel(file_path):
    """读取并处理Excel文件"""

    # 读取Excel文件
    print("正在读取Excel文件...")
    df = pd.read_excel(file_path)

    # 显示数据基本信息
    print(f"数据总行数: {len(df)}")
    print(f"数据列名: {list(df.columns)}")
    print("\n前5行数据:")
    print(df.head())

    return df

def is_question_keyword_based(message):
    """基于关键字判断消息是否是问句（fallback方法）"""
    if pd.isna(message) or message == '':
        return '否'

    message_str = str(message).strip()

    # 问句的判断规则
    question_indicators = [
        # 直接的问号
        '？', '?',
        # 疑问词开头
        '什么', '怎么', '如何', '为什么', '哪个', '哪里', '哪些', '谁', '何时', '多少',
        '啥', '咋', '咋样', '怎样', '几', '多久', '多长',
        # 疑问词在中间或结尾
        '是什么', '是谁', '是哪', '怎么样', '如何',
        # 常见问句模式
        '吗', '呢', '不是吗', '对吗', '好吗', '行吗', '可以吗'
    ]

    # 检查是否包含问号
    if '？' in message_str or '?' in message_str:
        return '是'

    # 检查是否包含疑问词
    for indicator in question_indicators:
        if indicator in message_str:
            return '是'

    # 检查是否以疑问语气结尾
    question_endings = ['吗', '呢', '不', '吧']
    for ending in question_endings:
        if message_str.endswith(ending):
            return '是'

    return '否'

def is_question(message):
    """使用模型或关键字判断消息是否是问句"""
    if pd.isna(message) or message == '':
        return '否'

    message_str = str(message).strip()

    # 如果消息太短（1个字符），直接返回否
    if len(message_str) <= 1:
        return '否'

    # 直接使用关键字方法（模型调用已禁用）
    return is_question_keyword_based(message)

def extract_cycle_label(rag_content):
    """从rag内容中提取周期标签"""

    if pd.isna(rag_content) or rag_content == '':
        return ''

    try:
        rag_str = str(rag_content)

        # 寻找周期标签的模式
        pattern = r"'周期标签':\s*'([^']+)'"
        match = re.search(pattern, rag_str)

        if match:
            cycle_label_str = match.group(1)
            # 处理转义字符，去掉反斜杠
            cycle_label_str = cycle_label_str.replace('\\\"', '"')

            try:
                # 尝试解析JSON
                cycle_label_json = json.loads(cycle_label_str)
                return cycle_label_json
            except json.JSONDecodeError:
                # 如果JSON解析失败，返回原始字符串（去掉转义字符）
                return cycle_label_str
        else:
            return ''

    except Exception as e:
        print(f"提取周期标签时出错: {e}")
        return ''

def extract_batch_id(rag_content):
    """从rag内容中提取批次ID"""

    if pd.isna(rag_content) or rag_content == '':
        return ''

    try:
        rag_str = str(rag_content)

        # 首先尝试解析JSON结构
        try:
            rag_json = json.loads(rag_str)
            recall_items = rag_json.get('data', {}).get('recall', [])

            for item in recall_items:
                if item.get('db_name') == '用户信息库':
                    content = item.get('content', '')
                    if content:
                        # content是字符串形式的列表，需要解析
                        try:
                            content_list = _normalize_user_info_content(content)
                            for user_info in content_list:
                                if isinstance(user_info, dict) and user_info.get('批次ID'):
                                    return user_info['批次ID']
                        except:
                            pass
        except:
            pass

        # 备用正则表达式方法
        pattern = r"'批次ID':\s*'([^']+)'"
        match = re.search(pattern, rag_str)
        if match:
            return match.group(1)

        return ''

    except Exception as e:
        print(f"提取批次ID时出错: {e}")
        return ''

def extract_questionnaire_content(rag_content):
    """从rag内容中提取调查问卷库内容"""

    if pd.isna(rag_content) or rag_content == '':
        return ''

    try:
        rag_str = str(rag_content)

        # 首先尝试解析JSON结构
        try:
            rag_json = json.loads(rag_str)
            recall_items = rag_json.get('data', {}).get('recall', [])

            for item in recall_items:
                if item.get('db_name') == '调查问卷库':
                    content = item.get('content', '')
                    if content:
                        try:
                            content_list = eval(content)
                            # 查找问卷内容字段
                            for questionnaire_data in content_list:
                                if isinstance(questionnaire_data, dict) and '问卷内容' in questionnaire_data:
                                    questionnaire_content_str = questionnaire_data['问卷内容']
                                    # 尝试解析问卷内容JSON
                                    try:
                                        questionnaire_json = json.loads(questionnaire_content_str)
                                        return json.dumps(questionnaire_json, ensure_ascii=False)
                                    except:
                                        return questionnaire_content_str
                        except:
                            pass
        except:
            pass

        return ''

    except Exception as e:
        print(f"提取问卷内容时出错: {e}")
        return ''

def extract_questionnaire_title(rag_content):
    """从rag内容中提取问卷标题"""

    if pd.isna(rag_content) or rag_content == '':
        return ''

    try:
        rag_str = str(rag_content)

        # 首先尝试解析JSON结构
        try:
            rag_json = json.loads(rag_str)
            recall_items = rag_json.get('data', {}).get('recall', [])

            for item in recall_items:
                if item.get('db_name') == '调查问卷库':
                    content = item.get('content', '')
                    if content:
                        try:
                            content_list = eval(content)
                            # 查找问卷标题字段
                            for questionnaire_data in content_list:
                                if isinstance(questionnaire_data, dict) and '问卷标题' in questionnaire_data:
                                    return questionnaire_data['问卷标题']
                        except:
                            # 备用正则表达式方法
                            title_pattern = r'"问卷标题":\s*"([^"]+)"'
                            title_match = re.search(title_pattern, content)
                            if title_match:
                                return title_match.group(1)
        except:
            pass

        return ''

    except Exception as e:
        print(f"提取问卷标题时出错: {e}")
        return ''

def extract_questionnaire_id(rag_content):
    """从rag内容中提取问卷ID"""

    if pd.isna(rag_content) or rag_content == '':
        return ''

    try:
        rag_str = str(rag_content)

        # 首先尝试解析JSON结构
        try:
            rag_json = json.loads(rag_str)
            recall_items = rag_json.get('data', {}).get('recall', [])

            for item in recall_items:
                if item.get('db_name') == '调查问卷库':
                    content = item.get('content', '')
                    if content:
                        try:
                            content_list = eval(content)
                            # 查找问卷ID字段
                            for questionnaire_data in content_list:
                                if isinstance(questionnaire_data, dict) and '问卷ID' in questionnaire_data:
                                    return questionnaire_data['问卷ID']
                        except:
                            # 备用正则表达式方法
                            id_pattern = r'"问卷ID":\s*"([^"]+)"'
                            id_match = re.search(id_pattern, content)
                            if id_match:
                                return id_match.group(1)
        except:
            pass

        return ''

    except Exception as e:
        print(f"提取问卷ID时出错: {e}")
        return ''

def extract_exam_paper_info(rag_content):
    """从rag内容中提取试卷库信息"""

    if pd.isna(rag_content) or rag_content == '':
        return {'试卷名称': '', '试卷ID': '', '分数': '', '天': '', '正确率': ''}

    try:
        rag_str = str(rag_content)

        exam_info = {
            '试卷名称': '',
            '试卷ID': '',
            '分数': '',
            '天': '',
            '正确率': ''
        }

        # 首先尝试解析JSON结构
        try:
            rag_json = json.loads(rag_str)
            recall_items = rag_json.get('data', {}).get('recall', [])

            for item in recall_items:
                if item.get('db_name') == '试卷库':
                    content = item.get('content', '')
                    if content:
                        try:
                            # content可能是字符串形式的列表
                            content_list = eval(content)
                            if isinstance(content_list, list) and len(content_list) > 0:
                                exam_data = content_list[0]
                                if isinstance(exam_data, dict):
                                    exam_info['试卷名称'] = exam_data.get('试卷名称', '')
                                    exam_info['试卷ID'] = exam_data.get('试卷ID', '')
                                    exam_info['分数'] = str(exam_data.get('分数', ''))
                                    exam_info['天'] = str(exam_data.get('天', ''))
                                    exam_info['正确率'] = str(exam_data.get('正确率', ''))
                                    return exam_info
                        except:
                            # 如果content是字符串，直接在其中搜索
                            if '试卷名称' in content:
                                patterns = {
                                    '试卷名称': r'"试卷名称":\s*"([^"]+)"',
                                    '试卷ID': r'"试卷ID":\s*"([^"]+)"',
                                    '分数': r'"分数":\s*"?([^",}]+)"?',
                                    '天': r'"天":\s*"?([^",}]+)"?',
                                    '正确率': r'"正确率":\s*"?([^",}]+)"?'
                                }

                                for key, pattern in patterns.items():
                                    match = re.search(pattern, content)
                                    if match:
                                        if key == '天':
                                            exam_info['天'] = match.group(1).strip('"')
                                        else:
                                            exam_info[key] = match.group(1).strip('"')
                                return exam_info
        except:
            pass

        # 备用正则表达式方法
        if '"db_name": "试卷库"' in rag_str or '试卷名称' in rag_str:
            patterns = {
                '试卷名称': r'"试卷名称":\s*"([^"]+)"',
                '试卷ID': r'"试卷ID":\s*"([^"]+)"',
                '分数': r'"分数":\s*"?([^",}]+)"?',
                '天': r'"天":\s*"?([^",}]+)"?',
                '正确率': r'"正确率":\s*"?([^",}]+)"?'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, rag_str)
                if match:
                    if key == '天':
                        exam_info['天'] = match.group(1).strip('"')
                    else:
                        exam_info[key] = match.group(1).strip('"')

        return exam_info

    except Exception as e:
        print(f"提取试卷信息时出错: {e}")
        return {'试卷名称': '', '试卷ID': '', '分数': '', '天': '', '正确率': ''}

def extract_cycle_batch_id(rag_content):
    """从rag内容中提取周期标签对应的批次ID"""

    if pd.isna(rag_content) or rag_content == '':
        return ''

    try:
        rag_str = str(rag_content)

        # 首先尝试解析JSON结构，从用户信息库中提取批次ID
        try:
            rag_json = json.loads(rag_str)
            recall_items = rag_json.get('data', {}).get('recall', [])

            for item in recall_items:
                if item.get('db_name') == '用户信息库':
                    content = item.get('content', '')
                    if content:
                        try:
                            content_list = _normalize_user_info_content(content)
                            for user_info in content_list:
                                if isinstance(user_info, dict) and '周期标签' in user_info and '批次ID' in user_info:
                                    return user_info['批次ID']
                        except:
                            pass
        except:
            pass

        # 备用方法：直接提取批次ID
        return extract_batch_id(rag_content)

    except Exception as e:
        print(f"提取周期批次ID时出错: {e}")
        return ''


def extract_user_tags(rag_content):
    """从rag内容中提取用户标签信息"""

    if pd.isna(rag_content) or rag_content == '':
        return ''

    try:
        rag_str = str(rag_content)

        # 优先解析JSON结构
        try:
            rag_json = json.loads(rag_str)
            recall_items = rag_json.get('data', {}).get('recall', [])

            for item in recall_items:
                if item.get('db_name') == '用户信息库':
                    content = item.get('content', '')
                    user_records = _normalize_user_info_content(content)
                    for user_info in user_records:
                        if not isinstance(user_info, dict):
                            continue
                        user_tags = user_info.get('标签') or user_info.get('用户标签')
                        if user_tags:
                            if isinstance(user_tags, (list, dict)):
                                try:
                                    return json.dumps(user_tags, ensure_ascii=False)
                                except Exception:
                                    return str(user_tags)
                            return str(user_tags)
        except Exception:
            pass

        # 回退到正则提取
        json_pattern = r'"标签":\s*(\[.*?\]|\{.*?\}|".*?")'
        match = re.search(json_pattern, rag_str, re.DOTALL)
        if match:
            value = match.group(1).strip()
            if value.startswith('"') and value.endswith('"'):
                return value[1:-1]
            return value

        single_quote_pattern = r"'标签':\s*'([^']+)'"
        match = re.search(single_quote_pattern, rag_str)
        if match:
            return match.group(1)

        return ''

    except Exception as e:
        print(f"提取用户标签时出错: {e}")
        return ''


def extract_user_nickname(rag_content):
    """从rag内容中提取用户昵称"""

    if pd.isna(rag_content) or rag_content == '':
        return ''

    try:
        rag_str = str(rag_content)

        # 优先解析JSON结构
        try:
            rag_json = json.loads(rag_str)
            recall_items = rag_json.get('data', {}).get('recall', [])

            for item in recall_items:
                if item.get('db_name') == '用户信息库':
                    content = item.get('content', '')
                    user_records = _normalize_user_info_content(content)
                    for user_info in user_records:
                        if not isinstance(user_info, dict):
                            continue
                        for key in ['用户昵称', '昵称', '用户名称']:
                            nickname = user_info.get(key)
                            if nickname:
                                return str(nickname)
        except Exception:
            pass

        # 回退到正则提取
        for key in ['用户昵称', '昵称', '用户名称']:
            pattern_double = rf'"{key}":\s*"([^"]+)"'
            match = re.search(pattern_double, rag_str)
            if match:
                return match.group(1)
            pattern_single = rf"'{key}':\s*'([^']+)'"
            match = re.search(pattern_single, rag_str)
            if match:
                return match.group(1)

        return ''

    except Exception as e:
        print(f"提取用户昵称时出错: {e}")
        return ''


def extract_last_consecutive_sales_messages(conversation):
    """提取历史对话中最后的连续销售消息

    改进版：同时提取最后客户消息块之前和之后的销售消息
    """

    if pd.isna(conversation) or conversation == '':
        return []

    try:
        conversation_str = str(conversation).strip()
        if not conversation_str:
            return []

        # 按行分割对话
        lines = conversation_str.split('\n')

        # 解析对话消息，识别每条消息的发送者和内容
        messages = []
        current_message = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否是新消息的开始（包含 [客户] 或 [销售] 标记）
            if line.startswith('[客户]') or line.startswith('[销售]'):
                # 如果有之前的消息，先保存
                if current_message:
                    messages.append(current_message)

                # 解析消息头
                if line.startswith('[客户]'):
                    sender = '客户'
                    # 提取消息内容 - 格式：[客户][时间戳]: 内容
                    colon_index = line.find(']: ')
                    if colon_index != -1:
                        content = line[colon_index + 3:]
                    else:
                        content = line[line.find(']') + 1:] if ']' in line else line
                elif line.startswith('[销售]'):
                    sender = '销售'
                    # 提取消息内容 - 格式：[销售][时间戳]: 内容
                    colon_index = line.find(']: ')
                    if colon_index != -1:
                        content = line[colon_index + 3:]
                    else:
                        content = line[line.find(']') + 1:] if ']' in line else line

                current_message = {
                    'sender': sender,
                    'content': content.strip()
                }
            else:
                # 这是多行消息的续行
                if current_message:
                    current_message['content'] += '\n' + line

        # 添加最后一条消息
        if current_message:
            messages.append(current_message)

        # 找到最后连续客户消息块的范围
        last_customer_block_start = -1
        last_customer_block_end = -1

        # 从后往前遍历，找到最后一个连续客户消息块
        in_customer_block = False
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['sender'] == '客户':
                if not in_customer_block:
                    # 刚进入客户消息块，记录结束位置
                    in_customer_block = True
                    last_customer_block_end = i
                # 继续记录客户消息块的开始位置
                last_customer_block_start = i
            else:
                # 遇到非客户消息
                if in_customer_block:
                    # 已经找到了完整的最后客户消息块
                    break

        last_sales_messages = []

        # 情况1：提取最后客户消息块之前的连续销售消息
        if last_customer_block_start > 0:
            for i in range(last_customer_block_start - 1, -1, -1):
                message = messages[i]
                if message['sender'] == '销售':
                    last_sales_messages.insert(0, message['content'].strip())
                elif message['sender'] == '客户':
                    break

        # 情况2：提取最后客户消息块之后的连续销售消息
        if last_customer_block_end != -1 and last_customer_block_end < len(messages) - 1:
            for i in range(last_customer_block_end + 1, len(messages)):
                message = messages[i]
                if message['sender'] == '销售':
                    last_sales_messages.append(message['content'].strip())
                elif message['sender'] == '客户':
                    break

        return last_sales_messages

    except Exception as e:
        print(f"提取最后销售消息时出错: {e}")
        return []


def extract_expected_sales_replies_for_customer(customer_data):
    """
    为每个客户提取预期的销售回复作为参考答案

    逻辑：
    1. 对每条客户消息，查找它下一条记录的"最后销售消息"
    2. 如果下一条记录有销售消息，则第一条作为参考答案，其余作为备选答案
    3. 这样可以构建客户消息 -> 销售回复的映射关系

    Args:
        customer_data: 同一客户的所有记录（按时间排序）

    Returns:
        为每条记录添加参考答案和备选答案的数据
    """

    if customer_data.empty:
        return customer_data

    # 确保按时间排序
    customer_data_sorted = customer_data.sort_values('客户消息时间').reset_index(drop=True)

    # 为每条记录添加参考答案列
    expected_replies = []
    alternative_replies = []

    for i in range(len(customer_data_sorted)):
        current_record = customer_data_sorted.iloc[i]
        current_customer_msg = current_record.get('最新客户消息', '')

        # 初始化
        expected_reply = ''
        alternative_reply_list = []

        # 如果当前记录有客户消息，查找下一条记录的销售回复
        if current_customer_msg and str(current_customer_msg).strip():
            # 查找下一条记录
            if i + 1 < len(customer_data_sorted):
                next_record = customer_data_sorted.iloc[i + 1]
                next_sales_messages = next_record.get('最后销售消息', [])

                # 如果下一条记录有销售消息
                if isinstance(next_sales_messages, list) and len(next_sales_messages) > 0:
                    # 第一条作为参考答案
                    expected_reply = next_sales_messages[0]
                    # 其余作为备选答案
                    if len(next_sales_messages) > 1:
                        alternative_reply_list = next_sales_messages[1:]
                elif isinstance(next_sales_messages, str) and next_sales_messages.strip():
                    # 如果是字符串形式，尝试解析
                    try:
                        sales_msg_list = eval(next_sales_messages)
                        if isinstance(sales_msg_list, list) and len(sales_msg_list) > 0:
                            expected_reply = sales_msg_list[0]
                            if len(sales_msg_list) > 1:
                                alternative_reply_list = sales_msg_list[1:]
                    except:
                        expected_reply = next_sales_messages

        expected_replies.append(expected_reply)
        alternative_replies.append(alternative_reply_list)

    # 添加新列到数据中
    customer_data_sorted['预期销售回复'] = expected_replies
    customer_data_sorted['备选销售回复'] = alternative_replies

    return customer_data_sorted


def build_customer_sales_reply_mapping(test_df):
    """
    为整个测试集构建客户消息到销售回复的映射

    这个函数会：
    1. 按客户分组处理数据
    2. 为每个客户的消息序列提取对应的销售回复
    3. 生成用于AI训练/测试的参考答案
    """

    print("\n=== 构建客户消息-销售回复映射 ===")

    all_processed_data = []

    # 按客户分组处理
    for customer_id in test_df['客户ID'].unique():
        customer_data = test_df[test_df['客户ID'] == customer_id].copy()

        # 为该客户提取预期回复
        processed_customer_data = extract_expected_sales_replies_for_customer(customer_data)

        all_processed_data.append(processed_customer_data)

    # 合并所有处理后的数据
    enhanced_test_df = pd.concat(all_processed_data, ignore_index=True)

    # 统计信息
    total_records = len(enhanced_test_df)
    has_expected_reply = len(enhanced_test_df[enhanced_test_df['预期销售回复'] != ''])
    has_alternative_reply = len(enhanced_test_df[enhanced_test_df['备选销售回复'].astype(str) != '[]'])

    print(f"  总记录数: {total_records}")
    print(f"  有预期回复的记录: {has_expected_reply} ({has_expected_reply/total_records*100:.1f}%)")
    print(f"  有备选回复的记录: {has_alternative_reply} ({has_alternative_reply/total_records*100:.1f}%)")

    # 显示一些样本
    print(f"\n=== 预期回复样本 ===")
    sample_with_reply = enhanced_test_df[enhanced_test_df['预期销售回复'] != ''].head(3)
    for i, (_, row) in enumerate(sample_with_reply.iterrows()):
        print(f"样本 {i+1}:")
        print(f"  客户消息: {row['最新客户消息'][:50]}...")
        print(f"  预期回复: {row['预期销售回复'][:100]}...")
        if row['备选销售回复'] and str(row['备选销售回复']) != '[]':
            print(f"  备选回复: {len(eval(str(row['备选销售回复'])))}条")
        print()

    return enhanced_test_df


def extract_sales_info(rag_content):
    """从rag内容中提取销售信息库的所有字段"""
    if pd.isna(rag_content) or rag_content == '':
        return {
            '直播课时间': '',
            '激活直播课链接': '',
            '学习档案链接': '',
            '手机平板上课链接': '',
            '电脑上课链接': '',
            '秒题技巧链接': '',
            '报课链接': '',
            '试学链接': ''
        }

    try:
        rag_str = str(rag_content)

        def normalize_content(content):
            """将content统一转换为列表[dict]形式，方便后续提取"""
            if isinstance(content, list):
                return content
            if isinstance(content, dict):
                return [content]
            if isinstance(content, str):
                text = content.strip()
                if not text:
                    return []
                # 优先尝试JSON解析
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return [parsed]
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
                # 回退到安全的字面量解析（处理单引号格式）
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, dict):
                        return [parsed]
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    return []
            return []

        def extract_from_dict(sales_data, info_dict):
            """从字典中提取目标字段"""
            fields = ['直播课时间', '激活直播课链接', '学习档案链接', '手机平板上课链接',
                      '电脑上课链接', '秒题技巧链接', '报课链接', '试学链接']
            for field in fields:
                info_dict[field] = sales_data.get(field, info_dict[field])
            return info_dict

        sales_info = {
            '直播课时间': '',
            '激活直播课链接': '',
            '学习档案链接': '',
            '手机平板上课链接': '',
            '电脑上课链接': '',
            '秒题技巧链接': '',
            '报课链接': '',
            '试学链接': ''
        }

        # 首先尝试解析JSON结构
        try:
            rag_json = json.loads(rag_str)
            recall_items = rag_json.get('data', {}).get('recall', [])

            for item in recall_items:
                if item.get('db_name') == '销售信息库':
                    content = item.get('content', '')
                    if content:
                        try:
                            content_list = normalize_content(content)
                            for sales_data in content_list:
                                if isinstance(sales_data, dict):
                                    return extract_from_dict(sales_data, sales_info)
                        except Exception:
                            # 备用正则表达式方法（将content转换为字符串再匹配）
                            content_str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                            if '直播课时间' in content_str:
                                patterns = {
                                    '直播课时间': r'"直播课时间":\s*"([^"]+)"',
                                    '激活直播课链接': r'"激活直播课链接":\s*"([^"]+)"',
                                    '学习档案链接': r'"学习档案链接":\s*"([^"]+)"',
                                    '手机平板上课链接': r'"手机平板上课链接":\s*"([^"]+)"',
                                    '电脑上课链接': r'"电脑上课链接":\s*"([^"]+)"',
                                    '秒题技巧链接': r'"秒题技巧链接":\s*"([^"]+)"',
                                    '报课链接': r'"报课链接":\s*"([^"]+)"',
                                    '试学链接': r'"试学链接":\s*"([^"]+)"'
                                }

                                for key, pattern in patterns.items():
                                    match = re.search(pattern, content_str)
                                    if match:
                                        sales_info[key] = match.group(1).strip('"')
                                return sales_info
        except:
            pass

        # 备用全局正则表达式方法
        if '"db_name": "销售信息库"' in rag_str:
            patterns = {
                '直播课时间': r'"直播课时间":\s*"([^"]+)"',
                '激活直播课链接': r'"激活直播课链接":\s*"([^"]+)"',
                '学习档案链接': r'"学习档案链接":\s*"([^"]+)"',
                '手机平板上课链接': r'"手机平板上课链接":\s*"([^"]+)"',
                '电脑上课链接': r'"电脑上课链接":\s*"([^"]+)"',
                '秒题技巧链接': r'"秒题技巧链接":\s*"([^"]+)"',
                '报课链接': r'"报课链接":\s*"([^"]+)"',
                '试学链接': r'"试学链接":\s*"([^"]+)"'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, rag_str)
                if match:
                    sales_info[key] = match.group(1).strip('"')

        return sales_info

    except Exception as e:
        print(f"提取销售信息时出错: {e}")
        return {
            '直播课时间': '',
            '激活直播课链接': '',
            '学习档案链接': '',
            '手机平板上课链接': '',
            '电脑上课链接': '',
            '秒题技巧链接': '',
            '报课链接': '',
            '试学链接': ''
        }


def is_media_only_sales_messages(last_sales_messages):
    """判断销售消息列表是否只包含媒体文件"""
    try:
        # 处理numpy数组或其他类型，先转换为原生Python对象
        if hasattr(last_sales_messages, 'tolist'):
            # 如果是numpy数组，转换为list
            last_sales_messages = last_sales_messages.tolist()
        elif hasattr(last_sales_messages, '__iter__') and not isinstance(last_sales_messages, (str, list)):
            # 如果是其他可迭代对象（除了字符串和列表），转换为list
            try:
                last_sales_messages = list(last_sales_messages)
            except:
                return False

        # 检查空值
        if last_sales_messages is None or (hasattr(last_sales_messages, '__len__') and len(str(last_sales_messages).strip()) == 0):
            return False

        # 检查pandas的NaN
        try:
            if pd.isna(last_sales_messages):
                return False
        except (ValueError, TypeError):
            # 如果pd.isna出现错误，说明可能是数组类型，继续处理
            pass

        # 检查空列表
        if isinstance(last_sales_messages, list) and len(last_sales_messages) == 0:
            return False

        # 检查字符串形式的空列表
        if isinstance(last_sales_messages, str) and last_sales_messages.strip() == '[]':
            return False

        # 如果是字符串，尝试解析为列表
        if isinstance(last_sales_messages, str):
            if last_sales_messages.strip() == '[]':
                return False
            messages_list = eval(last_sales_messages)
        elif isinstance(last_sales_messages, list):
            messages_list = last_sales_messages
        else:
            # 处理其他类型，如numpy数组等
            try:
                messages_list = list(last_sales_messages)
            except:
                return False

        # 如果列表为空，不过滤
        if not messages_list or len(messages_list) == 0:
            return False

        # 检查每条销售消息是否都只包含媒体文件
        for message in messages_list:
            message_str = str(message).strip()

            # 如果消息为空，跳过
            if not message_str:
                continue

            # 检查是否只包含图片、语音、视频等媒体文件标签
            # 媒体文件格式：<image_xxx>, <voice_xxx>, <video_xxx>, <emotion_xxx>
            if not (message_str.startswith('<image_') or
                   message_str.startswith('<voice_') or
                   message_str.startswith('<video_') or
                   message_str.startswith('<emotion_')):
                # 如果有任何一条消息不是纯媒体文件，则保留这条记录
                return False

        # 如果所有销售消息都只是媒体文件，则过滤掉
        return True

    except Exception as e:
        # 解析失败的情况，保留记录
        return False


def extract_history_until_last_customer(conversation_text):
    """截取历史对话到最后一句客户消息"""

    if pd.isna(conversation_text) or conversation_text == '':
        return ''

    conversation = str(conversation_text)
    lines = conversation.split('\n')

    # 找到最后一个[客户]消息的位置
    last_customer_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('[客户]'):
            last_customer_index = i

    if last_customer_index >= 0:
        # 返回到最后一句客户消息的所有对话
        return '\n'.join(lines[:last_customer_index + 1])
    else:
        return conversation

def extract_last_customer_message(conversation_text):
    """从对话历史中提取最后一句客户消息"""

    if pd.isna(conversation_text) or conversation_text == '':
        return ''

    # 将对话文本转换为字符串
    conversation = str(conversation_text)

    # 按行分割对话，找到所有[客户]开头的消息
    lines = conversation.split('\n')
    customer_messages = []

    for line in lines:
        line = line.strip()
        if line.startswith('[客户]'):
            # 提取消息内容，格式: [客户][时间戳]: 消息内容
            if ']:' in line:
                message_content = line.split(']:')[1].strip()
                if message_content:
                    customer_messages.append(message_content)

    # 返回最后一句客户消息
    return customer_messages[-1] if customer_messages else ''

def select_best_thought_unit_for_merged_message(processed_customer_records, merged_msg, original_msg):
    """
    为合并后的客户消息选择最合适的thought_unit
    优先级：未发送记录 > CSM记录
    """
    if processed_customer_records.empty:
        return None

    # 解析合并消息中包含的原始消息
    original_messages = merged_msg.split(' ')

    # 收集所有可能的thought_unit候选项
    candidates = []

    for _, record in processed_customer_records.iterrows():
        record_msg = record.get('客户消息', '')
        record_tu = record.get('thought_unit', '')
        record_source = record.get('thought_unit_source', '')

        # 检查当前记录的消息是否在合并消息中
        if (record_msg and pd.notna(record_tu) and record_tu != '' and
            record_msg in merged_msg):
            candidates.append({
                'message': record_msg,
                'thought_unit': record_tu,
                'source': record_source,
                'priority': 1 if record_source == '未发送' else 2  # 未发送优先级更高
            })

    if not candidates:
        return None

    # 按优先级排序（优先级数字越小越优先）
    candidates.sort(key=lambda x: x['priority'])

    # 返回优先级最高的thought_unit
    return candidates[0]


def merge_consecutive_customer_messages(conversation_text):
    """合并连续的客户消息为一个完整的问题

    返回:
        str: 合并后的客户消息内容
    """

    if pd.isna(conversation_text) or conversation_text == '':
        return ''

    try:
        conversation = str(conversation_text).strip()
        if not conversation:
            return ''

        # 按行分割对话
        lines = conversation.split('\n')

        # 解析对话消息
        messages = []
        current_message = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否是新消息的开始
            if line.startswith('[客户]') or line.startswith('[销售]'):
                # 如果有之前的消息，先保存
                if current_message:
                    messages.append(current_message)

                # 解析消息头
                if line.startswith('[客户]'):
                    sender = '客户'
                    colon_index = line.find(']: ')
                    if colon_index != -1:
                        content = line[colon_index + 3:]
                    else:
                        content = line[line.find(']') + 1:] if ']' in line else line
                elif line.startswith('[销售]'):
                    sender = '销售'
                    colon_index = line.find(']: ')
                    if colon_index != -1:
                        content = line[colon_index + 3:]
                    else:
                        content = line[line.find(']') + 1:] if ']' in line else line

                current_message = {
                    'sender': sender,
                    'content': content.strip()
                }
            else:
                # 多行消息的续行
                if current_message:
                    current_message['content'] += '\n' + line

        # 添加最后一条消息
        if current_message:
            messages.append(current_message)

        # 找到最后连续客户消息块并合并
        last_customer_messages = []

        # 从后往前找连续的客户消息
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['sender'] == '客户':
                last_customer_messages.insert(0, messages[i]['content'].strip())
            else:
                # 遇到非客户消息就停止
                break

        # 合并连续客户消息
        if last_customer_messages:
            # 用空格或换行符连接多条客户消息
            merged_message = ' '.join(last_customer_messages)
            return merged_message
        else:
            return ''

    except Exception as e:
        print(f"合并连续客户消息时出错: {e}")
        return ''

def get_date_from_timestamp(timestamp):
    """从时间戳中提取日期（年月日）"""

    if pd.isna(timestamp) or timestamp == '':
        return None

    try:
        if isinstance(timestamp, str):
            dt = pd.to_datetime(timestamp)
        else:
            dt = timestamp
        return dt.date()
    except:
        return None

def is_valid_rag(rag_content):
    """判断rag内容是否有效（不为空、不为空列表、不为空字符串）"""

    if pd.isna(rag_content) or rag_content == '':
        return False

    # 检查是否为空列表的字符串表示
    rag_str = str(rag_content).strip()
    if rag_str == '[]' or rag_str == 'nan':
        return False

    return True

def find_same_day_rag_info(customer_data):
    """为同一个客户的记录，根据同一天的CSM和未发送记录匹配rag信息"""

    # 筛选客户、CSM和未发送记录
    customer_records = customer_data[customer_data['发送方'] == '客户'].copy()
    csm_records = customer_data[customer_data['发送方'] == 'CSM'].copy()
    unsent_records = customer_data[customer_data['发送方'] == '未发送'].copy()

    # 如果没有客户记录，返回空
    if customer_records.empty:
        return pd.DataFrame()

    # 如果既没有CSM记录也没有未发送记录，返回空
    if csm_records.empty and unsent_records.empty:
        return pd.DataFrame()

    # 为客户记录添加日期字段
    customer_records['customer_date'] = customer_records['客户消息时间'].apply(get_date_from_timestamp)

    # 创建日期到rag信息的映射，优先选择有效的rag
    date_rag_map = {}

    # 处理CSM记录
    if not csm_records.empty:
        csm_records['csm_date'] = csm_records['发送时间'].apply(get_date_from_timestamp)

        # 按日期分组处理CSM记录
        for date, date_group in csm_records.groupby('csm_date'):
            if pd.isna(date):
                continue

            # 在同一天的CSM记录中寻找最好的rag信息
            for _, csm_record in date_group.iterrows():
                rag_content = csm_record['rag']

                # 如果找到有效的rag，使用它
                if is_valid_rag(rag_content):
                    cycle_label = extract_cycle_label(rag_content)
                    date_rag_map[date] = {
                        'rag': rag_content,
                        'thought_unit': csm_record.get('thought_unit', ''),
                        'cycle_label': cycle_label,
                        'source': 'CSM',
                        'ai_generated_message': csm_record.get('AI生成消息', ''),
                        'sent_message_content': csm_record.get('发送消息内容', '')
                    }
                    break  # 找到第一个有效的就使用

    # 处理未发送记录（补充CSM记录中没有的日期）
    if not unsent_records.empty:
        # 对于未发送记录，先尝试发送时间，如果无效则使用客户消息时间
        def get_unsent_date(record):
            send_time = record['发送时间']
            if pd.notna(send_time):
                return get_date_from_timestamp(send_time)
            else:
                # 如果发送时间无效，尝试使用客户消息时间
                customer_time = record['客户消息时间']
                return get_date_from_timestamp(customer_time)

        unsent_records['unsent_date'] = unsent_records.apply(get_unsent_date, axis=1)

        # 按日期分组处理未发送记录
        for date, date_group in unsent_records.groupby('unsent_date'):
            if pd.isna(date):
                continue

            # 如果该日期已经有CSM记录的rag信息，跳过
            if date in date_rag_map:
                continue

            # 在同一天的未发送记录中寻找最好的rag信息
            for _, unsent_record in date_group.iterrows():
                rag_content = unsent_record['rag']

                # 如果找到有效的rag，使用它
                if is_valid_rag(rag_content):
                    cycle_label = extract_cycle_label(rag_content)
                    # 只有当rag中确实包含周期标签时才使用
                    if cycle_label:
                        date_rag_map[date] = {
                            'rag': rag_content,
                            'thought_unit': unsent_record.get('thought_unit', ''),
                            'cycle_label': cycle_label,
                            'source': '未发送'
                        }
                        break  # 找到第一个有效的就使用

    print(f"    同一天rag映射(CSM+未发送): {len(date_rag_map)}个有效日期")

    # 为客户记录匹配同一天的rag信息
    processed_records = []

    for _, customer_record in customer_records.iterrows():
        customer_date = customer_record['customer_date']

        # 查找同一天的rag信息
        if customer_date and customer_date in date_rag_map:
            rag_info = date_rag_map[customer_date]

            # 检查是否应该过滤掉（包含行课期2的记录）
            if should_filter_cycle_label(rag_info['cycle_label']):
                continue

            # 创建处理后的记录
            processed_record = customer_record.copy()
            processed_record['rag'] = rag_info['rag']

            # 完全禁用RAG阶段的thought_unit分配，严格保留传播阶段的结果
            # 传播算法已经确保了精确的内容匹配和一对一映射
            # RAG信息中的thought_unit可能包含API调用数据，不应该用于客户记录
            # 只保留客户记录原有的thought_unit（来自传播算法或原始数据）
            pass  # 不进行任何thought_unit赋值，保持原始状态

            processed_record['周期标签'] = rag_info['cycle_label']
            processed_record['rag来源'] = rag_info['source']  # 标记rag来源
            # 添加CSM相关字段
            processed_record['ai_generated_message'] = rag_info.get('ai_generated_message', '')
            processed_record['sent_message_content'] = rag_info.get('sent_message_content', '')

            processed_records.append(processed_record)
        else:
            # 如果没有找到同一天的有效rag记录，跳过该客户记录
            continue

    return pd.DataFrame(processed_records)

def should_filter_cycle_label(cycle_label):
    """判断是否应该过滤掉包含行课期2的周期标签"""

    if not cycle_label:
        return False

    try:
        if isinstance(cycle_label, dict):
            cycle_dict = cycle_label
        elif isinstance(cycle_label, str):
            try:
                cycle_dict = json.loads(cycle_label)
            except:
                return False
        else:
            return False

        # 检查是否包含行课期2
        if '行课期' in cycle_dict and cycle_dict['行课期'] == 2:
            return True

        return False

    except Exception as e:
        print(f"检查周期标签时出错: {e}")
        return False

def standardize_conversation_format(conversation_text):
    """标准化对话格式，将所有非客户消息的发送者统一改为[销售]"""

    if pd.isna(conversation_text) or conversation_text == '':
        return ''

    conversation = str(conversation_text)
    lines = conversation.split('\n')
    standardized_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 如果是客户消息，保持不变
        if line.startswith('[客户]'):
            standardized_lines.append(line)
        # 如果是其他发送者的消息，统一改为[销售]
        elif line.startswith('['):
            # 查找消息内容开始的位置（]之后的第一个:后面）
            bracket_end = line.find(']')
            if bracket_end != -1:
                remaining = line[bracket_end + 1:]
                # 查找时间戳结束位置（如果有的话）
                if remaining.startswith('[') and ']:' in remaining:
                    # 有时间戳的格式: [发送者][时间戳]: 内容
                    time_end = remaining.find(']:')
                    if time_end != -1:
                        time_part = remaining[:time_end + 1]
                        content = remaining[time_end + 2:]
                        standardized_line = f"[销售]{time_part}: {content}"
                    else:
                        standardized_line = f"[销售]: {remaining}"
                elif remaining.startswith(':'):
                    # 无时间戳的格式: [发送者]: 内容
                    content = remaining[1:].strip()
                    standardized_line = f"[销售]: {content}"
                else:
                    # 其他格式，保持原样但改为销售
                    standardized_line = f"[销售]: {remaining}"

                standardized_lines.append(standardized_line)
            else:
                # 如果没找到]，说明格式异常，保持原样
                standardized_lines.append(line)
        else:
            # 不是以[开头的行，可能是普通文本，保持原样
            standardized_lines.append(line)

    return '\n'.join(standardized_lines)

def build_complete_conversation(customer_record, history_conversation):
    """构建包含客户消息的完整历史对话"""

    # 首先标准化历史对话格式
    standardized_history = standardize_conversation_format(history_conversation)

    # 获取客户消息和时间
    customer_msg = customer_record.get('客户消息', '') if pd.notna(customer_record.get('客户消息', '')) else ""
    customer_time = customer_record.get('客户消息时间', '') if pd.notna(customer_record.get('客户消息时间', '')) else ""

    # 如果没有客户消息，返回标准化后的历史对话
    if not customer_msg:
        return standardized_history

    # 构建客户消息行
    if customer_time:
        customer_line = f"[客户][{customer_time}]: {customer_msg}"
    else:
        customer_line = f"[客户]: {customer_msg}"

    # 将客户消息添加到标准化历史对话末尾
    if standardized_history:
        complete_conversation = standardized_history + "\n" + customer_line
    else:
        complete_conversation = customer_line

    return complete_conversation

def process_customer_records_with_rag(customer_data):
    """处理单个客户的记录，将同一天的CSM rag信息传播到客户记录中"""

    # 使用新的同一天匹配逻辑
    return find_same_day_rag_info(customer_data)

def process_data_by_sales_id(df):
    """按销售ID分组处理数据，生成标准格式的测试集"""

    print(f"\n=== 按销售ID分组处理数据，生成测试集 ===")

    # 获取所有唯一的销售ID
    unique_sales_ids = df['销售ID'].unique()
    print(f"共发现 {len(unique_sales_ids)} 个销售ID")

    all_test_data = []
    sales_summary = []
    cycle_label_stats = defaultdict(int)

    for sales_id in unique_sales_ids:
        print(f"\n处理销售ID: {sales_id[:30]}...")

        # 筛选当前销售的数据
        sales_data = df[df['销售ID'] == sales_id].copy()
        print(f"  原始记录数: {len(sales_data)}")

        print(f"  过滤后记录数: {len(sales_data)}")

        # 获取销售名称
        sales_name = sales_data['销售名称'].iloc[0] if not sales_data.empty and '销售名称' in sales_data.columns else "未知"

        # 按客户ID分组处理
        customers_in_sales = sales_data['客户ID'].unique()
        print(f"  客户数量: {len(customers_in_sales)}")

        customer_order = 1
        valid_customer_count = 0

        for customer_id in customers_in_sales:
            # 获取该客户的所有记录
            customer_data = sales_data[sales_data['客户ID'] == customer_id].copy()

            # 处理客户记录，传播rag信息并调整周期标签
            processed_customer_records = process_customer_records_with_rag(customer_data)

            # 如果没有有效的处理记录（无CSM记录或无rag），跳过该客户
            if processed_customer_records.empty:
                continue

            valid_customer_count += 1

            # 按时间排序
            time_columns = ['客户消息时间', '发送时间', '对话日期']
            for time_col in time_columns:
                if time_col in processed_customer_records.columns and processed_customer_records[time_col].notna().any():
                    processed_customer_records = processed_customer_records.sort_values(time_col)
                    break

            # 为每个客户记录创建测试数据
            for _, record in processed_customer_records.iterrows():
                # 获取客户消息
                customer_msg = record.get('客户消息', '') if pd.notna(record.get('客户消息', '')) else ""

                # 获取原始历史对话
                history_conversation = record.get('历史对话', '') if pd.notna(record.get('历史对话', '')) else ""

                # 构建包含客户消息的完整历史对话
                complete_conversation = build_complete_conversation(record, history_conversation)

                # 尝试合并连续客户消息以获得更完整的问题描述
                merged_customer_msg = merge_consecutive_customer_messages(complete_conversation)
                if merged_customer_msg and len(merged_customer_msg) > len(customer_msg):
                    customer_msg = merged_customer_msg

                    # 当消息被合并时，完全保留传播阶段分配的thought_unit
                    # 禁用消息合并阶段的thought_unit重新分配，避免重复使用
                    # 传播阶段已经确保了正确的一对一匹配，这里不应该再次分配
                    pass  # 保持原始record不变，不进行任何thought_unit重新分配

                # 提取周期标签统计
                cycle_label = record.get('周期标签', '')
                cycle_label_stats[str(cycle_label)] += 1

                # 判断是否为问句
                is_question_flag = is_question(customer_msg)

                # 提取用户信息库相关字段
                rag_content = record.get('rag', '')
                batch_id = extract_batch_id(rag_content)
                questionnaire_content = extract_questionnaire_content(rag_content)
                questionnaire_title = extract_questionnaire_title(rag_content)
                questionnaire_id = extract_questionnaire_id(rag_content)
                exam_paper_info = extract_exam_paper_info(rag_content)
                cycle_batch_id = extract_cycle_batch_id(rag_content)
                user_tags = extract_user_tags(rag_content)
                user_nickname = extract_user_nickname(rag_content)

                # 提取销售信息库相关字段
                sales_info = extract_sales_info(rag_content)

                # 注：移除了AI生成消息和发送消息内容的对比功能，因为客户消息没有这些字段

                # 提取最后的连续销售消息
                last_sales_messages = extract_last_consecutive_sales_messages(complete_conversation)

                # 创建标准格式的测试记录
                test_record = {
                    '销售ID': sales_id,
                    '客户ID': customer_id,
                    '发送方': '客户',  # 只保留发送方为客户的记录
                    '历史对话': complete_conversation,  # 包含客户消息的完整对话
                    '最新客户消息': customer_msg,
                    'rag': record.get('rag', ''),
                    'thought_unit': record.get('thought_unit', ''),
                    'thought_unit_source': record.get('thought_unit_source', ''),
                    # 添加提取的新字段
                    '强遵循标签': record.get('强遵循标签', 'False'),
                    'FAQ判断': record.get('FAQ判断', 'False'),
                    '知识问答判断': record.get('知识问答判断', 'False'),
                    '销售一级节点': record.get('销售一级节点', ''),
                    '销售二级节点': record.get('销售二级节点', ''),
                    'reference_script': record.get('reference_script', ''),
                    '周期标签': cycle_label,
                    '是否是问句': is_question_flag,
                    '客户消息时间': record.get('客户消息时间', ''),  # 保留客户消息时间列
                    # 额外信息用于分析
                    '销售名称': sales_name,
                    '销售内客户顺序': customer_order,
                    '原始记录数': len(customer_data),
                    '发送时间': record.get('发送时间', ''),
                    'rag来源': record.get('rag来源', ''),  # 标记rag来源（CSM或未发送）
                    # 用户信息库字段
                    '批次ID': batch_id,
                    '问卷内容': questionnaire_content,
                    '问卷标题': questionnaire_title,
                    '问卷ID': questionnaire_id,
                    '标签': user_tags,
                    '用户昵称': user_nickname,
                    '试卷名称': exam_paper_info['试卷名称'],
                    '试卷ID': exam_paper_info['试卷ID'],
                    '分数': exam_paper_info['分数'],
                    '天': exam_paper_info['天'],
                    '正确率': exam_paper_info['正确率'],
                    '周期批次ID': cycle_batch_id,
                    # 销售信息库字段
                    '直播课时间': sales_info['直播课时间'],
                    '激活直播课链接': sales_info['激活直播课链接'],
                    '学习档案链接': sales_info['学习档案链接'],
                    '手机平板上课链接': sales_info['手机平板上课链接'],
                    '电脑上课链接': sales_info['电脑上课链接'],
                    '秒题技巧链接': sales_info['秒题技巧链接'],
                    '报课链接': sales_info['报课链接'],
                    '试学链接': sales_info['试学链接'],
                    # 最后销售消息
                    '最后销售消息': last_sales_messages,
                    # CSM回复信息
                    'CSM_AI生成消息': record.get('CSM_AI生成消息', ''),
                    'CSM_发送消息内容': record.get('CSM_发送消息内容', ''),
                    'CSM_AI与审核结果是否一致': record.get('CSM_AI与审核结果是否一致', '')
                }

                all_test_data.append(test_record)

            customer_order += 1

        print(f"  有效客户数量（有rag信息）: {valid_customer_count}")

        # 记录该销售的汇总信息
        sales_summary.append({
            '销售ID': sales_id,
            '销售名称': sales_name,
            '原始客户数量': len(customers_in_sales),
            '有效客户数量': valid_customer_count,
            '总记录数': len(sales_data),
            '平均每客户记录数': len(sales_data) / len(customers_in_sales) if len(customers_in_sales) > 0 else 0
        })

    # 创建测试集DataFrame
    test_df = pd.DataFrame(all_test_data)
    sales_summary_df = pd.DataFrame(sales_summary)

    print(f"\n生成测试集记录数: {len(test_df)}")

    # 过滤纯媒体文件的销售消息记录
    original_count = len(test_df)
    media_only_mask = test_df['最后销售消息'].apply(is_media_only_sales_messages)
    test_df = test_df[~media_only_mask]
    filtered_count = original_count - len(test_df)

    if filtered_count > 0:
        print(f"过滤掉 {filtered_count} 条最后销售消息只包含媒体文件的记录")
        print(f"过滤后测试集记录数: {len(test_df)}")

    print(f"有效客户消息数: {len(test_df[test_df['最新客户消息'] != ''])}")
    print(f"问句数量: {len(test_df[test_df['是否是问句'] == '是'])}")
    print(f"有周期标签的记录: {len(test_df[test_df['周期标签'] != ''])}")

    # 构建客户消息到销售回复的映射
    enhanced_test_df = build_customer_sales_reply_mapping(test_df)

    return enhanced_test_df, sales_summary_df, cycle_label_stats

def analyze_cycle_labels(cycle_label_stats, test_df):
    """分析周期标签分布"""

    print("\n=== 周期标签分布分析 ===")

    # 周期标签总体分布
    total_records = len(test_df)
    print(f"总测试记录数: {total_records}")

    cycle_analysis = []
    for label, count in sorted(cycle_label_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_records) * 100 if total_records > 0 else 0
        cycle_analysis.append({
            '周期标签': label,
            '记录数': count,
            '占比': f"{percentage:.1f}%"
        })
        print(f"  {label}: {count}条 ({percentage:.1f}%)")

    # 按销售分组的周期标签分布
    sales_cycle_analysis = []
    for sales_id in test_df['销售ID'].unique():
        sales_data = test_df[test_df['销售ID'] == sales_id]
        sales_name = sales_data['销售名称'].iloc[0]

        sales_cycle_count = sales_data['周期标签'].value_counts()
        for label, count in sales_cycle_count.items():
            percentage = (count / len(sales_data)) * 100
            sales_cycle_analysis.append({
                '销售ID': sales_id,
                '销售名称': sales_name,
                '周期标签': str(label),
                '记录数': count,
                '该销售占比': f"{percentage:.1f}%"
            })

    return pd.DataFrame(cycle_analysis), pd.DataFrame(sales_cycle_analysis)

def create_rag_process_flow_table(test_df):
    """创建Rag处理流程分析表"""

    flow_data = []

    # 1. 数据处理流程说明
    flow_data.append({
        '处理步骤': '步骤1',
        '流程描述': '原始数据筛选',
        '处理内容': '筛选切分次数1和2的记录',
        '数据变化': f'保留符合条件的记录用于后续处理',
        '关键逻辑': '基于切分次数字段进行初步过滤'
    })

    flow_data.append({
        '处理步骤': '步骤2',
        '流程描述': '按销售ID和客户ID分组',
        '处理内容': '将数据按销售ID分组，再按客户ID子分组',
        '数据变化': f'共处理{test_df["销售ID"].nunique()}个销售，{test_df["客户ID"].nunique()}个客户',
        '关键逻辑': '保证同一销售下客户按顺序处理'
    })

    flow_data.append({
        '处理步骤': '步骤3',
        '流程描述': '客户-CSM/未发送记录匹配',
        '处理内容': '根据同一天日期匹配客户记录和CSM/未发送记录',
        '数据变化': '只保留有CSM或未发送记录的客户数据',
        '关键逻辑': '同一客户ID + 同一日期 = 匹配成功，优先CSM记录'
    })

    flow_data.append({
        '处理步骤': '步骤4',
        '流程描述': 'Rag内容验证和传播',
        '处理内容': '验证CSM/未发送记录的rag内容有效性，传播到客户记录',
        '数据变化': '过滤掉rag为空列表[]或无效的记录，标记rag来源',
        '关键逻辑': 'is_valid_rag()函数排除空值、空列表、NaN'
    })

    flow_data.append({
        '处理步骤': '步骤5',
        '流程描述': '周期标签提取和过滤',
        '处理内容': '从rag中提取周期标签，过滤掉行课期2的记录',
        '数据变化': '移除包含行课期2的记录',
        '关键逻辑': 'extract_cycle_label()解析JSON + should_filter_cycle_label()过滤'
    })

    flow_data.append({
        '处理步骤': '步骤6',
        '流程描述': '对话格式标准化',
        '处理内容': '将历史对话中所有非客户消息的发送者统一改为[销售]',
        '数据变化': '统一对话格式，[CSM]、[销售xxx]等都改为[销售]',
        '关键逻辑': 'standardize_conversation_format()函数标准化格式'
    })

    flow_data.append({
        '处理步骤': '步骤7',
        '流程描述': '历史对话重构',
        '处理内容': '将客户消息和时间追加到标准化后的历史对话末尾',
        '数据变化': '生成包含客户消息的完整对话历史',
        '关键逻辑': 'build_complete_conversation()函数拼接对话'
    })

    flow_data.append({
        '处理步骤': '步骤8',
        '流程描述': '最终测试集生成',
        '处理内容': '创建标准格式的测试记录，只保留客户记录',
        '数据变化': f'最终生成{len(test_df)}条测试记录',
        '关键逻辑': '发送方统一为"客户"，包含完整rag和周期标签信息'
    })

    return pd.DataFrame(flow_data)

def create_data_quality_stats_table(test_df):
    """创建数据质量统计表"""

    quality_stats = []

    # 基础数据统计
    total_records = len(test_df)
    quality_stats.append({
        '质量指标': '总记录数',
        '数值': total_records,
        '占比': '100.0%',
        '说明': '最终测试集的总记录数量'
    })

    # 客户消息完整性
    valid_messages = len(test_df[test_df['最新客户消息'] != ''])
    message_rate = (valid_messages / total_records) * 100 if total_records > 0 else 0
    quality_stats.append({
        '质量指标': '有效客户消息数',
        '数值': valid_messages,
        '占比': f'{message_rate:.1f}%',
        '说明': '包含非空客户消息的记录数'
    })

    # Rag内容完整性
    valid_rags = len(test_df[test_df['rag'] != ''])
    rag_rate = (valid_rags / total_records) * 100 if total_records > 0 else 0
    quality_stats.append({
        '质量指标': '有效Rag内容数',
        '数值': valid_rags,
        '占比': f'{rag_rate:.1f}%',
        '说明': '包含有效rag内容的记录数'
    })

    # 周期标签覆盖率
    valid_cycle_labels = len(test_df[test_df['周期标签'] != ''])
    cycle_rate = (valid_cycle_labels / total_records) * 100 if total_records > 0 else 0
    quality_stats.append({
        '质量指标': '周期标签覆盖数',
        '数值': valid_cycle_labels,
        '占比': f'{cycle_rate:.1f}%',
        '说明': '成功提取周期标签的记录数'
    })

    # 问句识别率
    questions = len(test_df[test_df['是否是问句'] == '是'])
    question_rate = (questions / total_records) * 100 if total_records > 0 else 0
    quality_stats.append({
        '质量指标': '问句数量',
        '数值': questions,
        '占比': f'{question_rate:.1f}%',
        '说明': '被识别为问句的客户消息数'
    })

    # 历史对话完整性
    valid_history = len(test_df[test_df['历史对话'] != ''])
    history_rate = (valid_history / total_records) * 100 if total_records > 0 else 0
    quality_stats.append({
        '质量指标': '完整历史对话数',
        '数值': valid_history,
        '占比': f'{history_rate:.1f}%',
        '说明': '包含历史对话内容的记录数'
    })

    # 客户消息时间完整性
    valid_time = len(test_df[test_df['客户消息时间'] != ''])
    time_rate = (valid_time / total_records) * 100 if total_records > 0 else 0
    quality_stats.append({
        '质量指标': '客户消息时间数',
        '数值': valid_time,
        '占比': f'{time_rate:.1f}%',
        '说明': '包含客户消息时间的记录数'
    })

    # 销售分布情况
    unique_sales = test_df['销售ID'].nunique()
    avg_per_sales = total_records / unique_sales if unique_sales > 0 else 0
    quality_stats.append({
        '质量指标': '销售人员数量',
        '数值': unique_sales,
        '占比': f'平均{avg_per_sales:.1f}条/人',
        '说明': '涉及的销售人员数量和平均记录数'
    })

    # 客户分布情况
    unique_customers = test_df['客户ID'].nunique()
    avg_per_customer = total_records / unique_customers if unique_customers > 0 else 0
    quality_stats.append({
        '质量指标': '客户数量',
        '数值': unique_customers,
        '占比': f'平均{avg_per_customer:.1f}条/客户',
        '说明': '涉及的客户数量和平均记录数'
    })

    # Rag来源分布
    if 'rag来源' in test_df.columns:
        csm_source = len(test_df[test_df['rag来源'] == 'CSM'])
        unsent_source = len(test_df[test_df['rag来源'] == '未发送'])
        csm_rate = (csm_source / total_records) * 100 if total_records > 0 else 0
        unsent_rate = (unsent_source / total_records) * 100 if total_records > 0 else 0

        quality_stats.append({
            '质量指标': 'CSM来源Rag数',
            '数值': csm_source,
            '占比': f'{csm_rate:.1f}%',
            '说明': '来自CSM记录的rag内容数量'
        })

        quality_stats.append({
            '质量指标': '未发送来源Rag数',
            '数值': unsent_source,
            '占比': f'{unsent_rate:.1f}%',
            '说明': '来自未发送记录的rag内容数量'
        })

    # 用户信息库字段统计
    if '批次ID' in test_df.columns:
        valid_batch_id = len(test_df[test_df['批次ID'] != ''])
        batch_rate = (valid_batch_id / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效批次ID数',
            '数值': valid_batch_id,
            '占比': f'{batch_rate:.1f}%',
            '说明': '包含有效批次ID的记录数'
        })

    if '问卷内容' in test_df.columns:
        valid_questionnaire = len(test_df[test_df['问卷内容'] != ''])
        questionnaire_rate = (valid_questionnaire / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效问卷内容数',
            '数值': valid_questionnaire,
            '占比': f'{questionnaire_rate:.1f}%',
            '说明': '包含有效问卷内容的记录数'
        })

    if '标签' in test_df.columns:
        def has_valid_tags(value):
            if pd.isna(value):
                return False
            if isinstance(value, (list, dict)):
                return len(value) > 0
            text = str(value).strip()
            return text not in ('', '[]', '{}', 'nan', 'None')

        valid_tags = test_df['标签'].apply(has_valid_tags).sum()
        tags_rate = (valid_tags / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效用户标签数',
            '数值': valid_tags,
            '占比': f'{tags_rate:.1f}%',
            '说明': '包含有效用户标签信息的记录数'
        })

    if '用户昵称' in test_df.columns:
        def has_valid_nickname(value):
            if pd.isna(value):
                return False
            text = str(value).strip()
            return text not in ('', 'nan', 'None', '')

        valid_nickname = test_df['用户昵称'].apply(has_valid_nickname).sum()
        nickname_rate = (valid_nickname / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效用户昵称数',
            '数值': valid_nickname,
            '占比': f'{nickname_rate:.1f}%',
            '说明': '包含有效用户昵称信息的记录数'
        })

    if '试卷名称' in test_df.columns:
        valid_exam_paper = len(test_df[test_df['试卷名称'] != ''])
        exam_rate = (valid_exam_paper / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效试卷信息数',
            '数值': valid_exam_paper,
            '占比': f'{exam_rate:.1f}%',
            '说明': '包含有效试卷信息的记录数'
        })

    if '周期批次ID' in test_df.columns:
        valid_cycle_batch = len(test_df[test_df['周期批次ID'] != ''])
        cycle_batch_rate = (valid_cycle_batch / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效周期批次ID数',
            '数值': valid_cycle_batch,
            '占比': f'{cycle_batch_rate:.1f}%',
            '说明': '包含有效周期批次ID的记录数'
        })

    # 销售信息库字段统计
    if '直播课时间' in test_df.columns:
        valid_live_time = len(test_df[test_df['直播课时间'] != ''])
        live_time_rate = (valid_live_time / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效直播课时间数',
            '数值': valid_live_time,
            '占比': f'{live_time_rate:.1f}%',
            '说明': '包含有效直播课时间的记录数'
        })

    if '激活直播课链接' in test_df.columns:
        valid_activate_link = len(test_df[test_df['激活直播课链接'] != ''])
        activate_link_rate = (valid_activate_link / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效激活直播课链接数',
            '数值': valid_activate_link,
            '占比': f'{activate_link_rate:.1f}%',
            '说明': '包含有效激活直播课链接的记录数'
        })

    if '学习档案链接' in test_df.columns:
        valid_study_link = len(test_df[test_df['学习档案链接'] != ''])
        study_link_rate = (valid_study_link / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效学习档案链接数',
            '数值': valid_study_link,
            '占比': f'{study_link_rate:.1f}%',
            '说明': '包含有效学习档案链接的记录数'
        })

    if '手机平板上课链接' in test_df.columns:
        valid_mobile_link = len(test_df[test_df['手机平板上课链接'] != ''])
        mobile_link_rate = (valid_mobile_link / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效手机平板上课链接数',
            '数值': valid_mobile_link,
            '占比': f'{mobile_link_rate:.1f}%',
            '说明': '包含有效手机平板上课链接的记录数'
        })

    if '电脑上课链接' in test_df.columns:
        valid_pc_link = len(test_df[test_df['电脑上课链接'] != ''])
        pc_link_rate = (valid_pc_link / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效电脑上课链接数',
            '数值': valid_pc_link,
            '占比': f'{pc_link_rate:.1f}%',
            '说明': '包含有效电脑上课链接的记录数'
        })

    if '秒题技巧链接' in test_df.columns:
        valid_tips_link = len(test_df[test_df['秒题技巧链接'] != ''])
        tips_link_rate = (valid_tips_link / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效秒题技巧链接数',
            '数值': valid_tips_link,
            '占比': f'{tips_link_rate:.1f}%',
            '说明': '包含有效秒题技巧链接的记录数'
        })

    if '报课链接' in test_df.columns:
        valid_enroll_link = len(test_df[test_df['报课链接'] != ''])
        enroll_link_rate = (valid_enroll_link / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效报课链接数',
            '数值': valid_enroll_link,
            '占比': f'{enroll_link_rate:.1f}%',
            '说明': '包含有效报课链接的记录数'
        })

    if '试学链接' in test_df.columns:
        valid_trial_link = len(test_df[test_df['试学链接'] != ''])
        trial_link_rate = (valid_trial_link / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有效试学链接数',
            '数值': valid_trial_link,
            '占比': f'{trial_link_rate:.1f}%',
            '说明': '包含有效试学链接的记录数'
        })

    # CSM消息相关统计
    # 注：移除了AI生成消息、发送消息内容和消息一致性相关统计，因为客户消息没有这些字段

    # 最后销售消息统计
    if '最后销售消息' in test_df.columns:
        # 统计有最后销售消息的记录
        has_last_sales_msg = test_df['最后销售消息'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        valid_last_sales_count = has_last_sales_msg.sum()
        last_sales_rate = (valid_last_sales_count / total_records) * 100 if total_records > 0 else 0

        quality_stats.append({
            '质量指标': '有最后销售消息数',
            '数值': valid_last_sales_count,
            '占比': f'{last_sales_rate:.1f}%',
            '说明': '包含最后连续销售消息的记录数'
        })

        # 统计最后销售消息的平均数量
        if valid_last_sales_count > 0:
            avg_msg_count = test_df[has_last_sales_msg]['最后销售消息'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
            quality_stats.append({
                '质量指标': '平均销售消息数',
                '数值': f'{avg_msg_count:.1f}',
                '占比': '-',
                '说明': '每条记录平均包含的连续销售消息数量'
            })

    # 预期销售回复统计
    if '预期销售回复' in test_df.columns:
        valid_expected_reply = len(test_df[test_df['预期销售回复'] != ''])
        expected_reply_rate = (valid_expected_reply / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有预期销售回复数',
            '数值': valid_expected_reply,
            '占比': f'{expected_reply_rate:.1f}%',
            '说明': '包含预期销售回复的记录数（用于AI训练参考）'
        })

    if '备选销售回复' in test_df.columns:
        has_alternative_reply = test_df['备选销售回复'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        valid_alternative_count = has_alternative_reply.sum()
        alternative_rate = (valid_alternative_count / total_records) * 100 if total_records > 0 else 0
        quality_stats.append({
            '质量指标': '有备选销售回复数',
            '数值': valid_alternative_count,
            '占比': f'{alternative_rate:.1f}%',
            '说明': '包含备选销售回复的记录数（多种回复选择）'
        })

        # 统计备选回复的平均数量
        if valid_alternative_count > 0:
            avg_alternative_count = test_df[has_alternative_reply]['备选销售回复'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
            quality_stats.append({
                '质量指标': '平均备选回复数',
                '数值': f'{avg_alternative_count:.1f}',
                '占比': '-',
                '说明': '每条记录平均包含的备选销售回复数量'
            })

    return pd.DataFrame(quality_stats)

def save_test_dataset_results(test_df, sales_summary, cycle_label_stats):
    """保存测试集结果，格式类似processed_all_data.xlsx"""

    output_file = '按销售ID分组的测试集和周期标签分析_with_thought_unit_final2.xlsx'

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 1. 标准格式测试集（包含用户信息库、销售信息库、CSM字段、最后销售消息和预期回复）
        standard_columns = ['销售ID', '客户ID', '发送方', '历史对话', '最新客户消息', 'rag', 'thought_unit', 'thought_unit_source', '强遵循标签', 'FAQ判断', '知识问答判断', '销售一级节点', '销售二级节点', 'reference_script', '周期标签', '是否是问句', '客户消息时间', '发送时间', 'rag来源', '批次ID', '问卷内容', '问卷标题', '问卷ID', '标签', '用户昵称', '试卷名称', '试卷ID', '分数', '天', '正确率', '周期批次ID', '直播课时间', '激活直播课链接', '学习档案链接', '手机平板上课链接', '电脑上课链接', '秒题技巧链接', '报课链接', '试学链接', '最后销售消息']

        # 如果有CSM回复相关列，也包含进来
        if 'CSM_AI生成消息' in test_df.columns:
            standard_columns.extend(['CSM_AI生成消息', 'CSM_发送消息内容', 'CSM_AI与审核结果是否一致'])

        # 如果有预期回复列，也包含进来
        if '预期销售回复' in test_df.columns:
            standard_columns.extend(['预期销售回复', '备选销售回复'])

        standard_test_df = test_df[standard_columns].copy()
        standard_test_df.to_excel(writer, sheet_name='测试集_标准格式', index=False)

        # 2. 完整测试集（包含额外分析字段）
        test_df.to_excel(writer, sheet_name='测试集_完整信息', index=False)

        # 3. 销售汇总信息
        sales_summary.to_excel(writer, sheet_name='销售汇总', index=False)

        # 4. 周期标签分析
        cycle_analysis_df, sales_cycle_analysis_df = analyze_cycle_labels(cycle_label_stats, test_df)
        cycle_analysis_df.to_excel(writer, sheet_name='周期标签总体分布', index=False)
        sales_cycle_analysis_df.to_excel(writer, sheet_name='各销售周期标签分布', index=False)

        # 5. 问句分析
        question_analysis = []
        for sales_id in test_df['销售ID'].unique():
            sales_data = test_df[test_df['销售ID'] == sales_id]
            sales_name = sales_data['销售名称'].iloc[0]
            sales_questions = len(sales_data[sales_data['是否是问句'] == '是'])
            question_rate = (sales_questions / len(sales_data)) * 100 if len(sales_data) > 0 else 0

            question_analysis.append({
                '销售ID': sales_id,
                '销售名称': sales_name,
                '总客户数': len(sales_data),
                '问句数量': sales_questions,
                '问句率': f"{question_rate:.1f}%"
            })

        question_analysis_df = pd.DataFrame(question_analysis)
        question_analysis_df.to_excel(writer, sheet_name='问句分析', index=False)

        # 6. Rag处理流程分析表
        rag_process_flow = create_rag_process_flow_table(test_df)
        rag_process_flow.to_excel(writer, sheet_name='Rag处理流程分析', index=False)

        # 7. 数据质量统计表
        data_quality_stats = create_data_quality_stats_table(test_df)
        data_quality_stats.to_excel(writer, sheet_name='数据质量统计', index=False)

        # 8. 为每个销售创建详细视图
        unique_sales = test_df['销售ID'].unique()
        for sales_id in unique_sales[:8]:  # 所有8个销售
            sales_data = test_df[test_df['销售ID'] == sales_id]
            sales_name = sales_data['销售名称'].iloc[0]

            # 创建销售详细视图
            sales_detail = sales_data[['销售内客户顺序', '客户ID', '最新客户消息', '是否是问句', '周期标签', '原始记录数']].copy()
            sales_detail = sales_detail.sort_values('销售内客户顺序')

            sheet_name = f"{sales_name[:10]}_客户详情"  # 使用销售名称
            sales_detail.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n?? 测试集和周期标签分析结果已保存到: {output_file}")
    print("?? 包含以下工作表:")
    if '预期销售回复' in test_df.columns:
        print("  - 测试集_标准格式: 覆盖用户信息库（批次ID、标签、用户昵称、问卷信息等）、销售信息库链接、最后销售消息以及预期/备选销售回复字段")
    else:
        print("  - 测试集_标准格式: 覆盖用户信息库（批次ID、标签、用户昵称、问卷信息等）、销售信息库链接和最后销售消息等核心字段")
    print("  - 测试集_完整信息: 包含额外分析字段的完整数据")
    print("  - 销售汇总: 每个销售的客户统计")
    print("  - 周期标签总体分布: 所有周期标签的分布统计")
    print("  - 各销售周期标签分布: 每个销售的周期标签详细分布")
    print("  - 问句分析: 各销售的问句统计")
    print("  - Rag处理流程分析: 数据处理步骤和关键逻辑说明（含未发送处理）")
    print("  - 数据质量统计: 各项数据质量指标、rag来源统计和预期回复统计")
    print("  - xxx_客户详情: 每个销售的客户详细信息（8个工作表）")

    if '预期销售回复' in test_df.columns:
        valid_expected_reply = len(test_df[test_df['预期销售回复'] != ''])
        expected_reply_rate = (valid_expected_reply / len(test_df)) * 100 if len(test_df) > 0 else 0
        print(f"\n?? 预期回复统计:")
        print(f"  - 有预期销售回复的记录: {valid_expected_reply} 条 ({expected_reply_rate:.1f}%)")
        if '备选销售回复' in test_df.columns:
            has_alternative_reply = test_df['备选销售回复'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
            valid_alternative_count = has_alternative_reply.sum()
            alternative_rate = (valid_alternative_count / len(test_df)) * 100 if len(test_df) > 0 else 0
            print(f"  - 有备选销售回复的记录: {valid_alternative_count} 条 ({alternative_rate:.1f}%)")
        print(f"  ?? 这些预期回复可以作为sales-agent测试的参考答案和人工标注的基础")


def add_csm_replies_to_customer_messages(df):
    """
    将发送方为CSM的回复信息添加到对应的客户消息中

    匹配逻辑：
    1. 以客户消息时间为条件
    2. 如果发送方为CSM的客户消息时间和发送方为客户的客户消息时间相同
    3. 则将CSM的AI生成消息、发送消息内容、AI与审核结果是否一致添加到客户记录中
    """
    print("=== 添加CSM回复信息到客户消息 ===")

    # 创建数据副本
    df_copy = df.copy()

    # 新增列，存储CSM回复信息
    df_copy['CSM_AI生成消息'] = ''
    df_copy['CSM_发送消息内容'] = ''
    df_copy['CSM_AI与审核结果是否一致'] = ''

    # 分别提取客户和CSM记录
    customer_records = df_copy[df_copy['发送方'] == '客户'].copy()
    csm_records = df_copy[df_copy['发送方'] == 'CSM'].copy()

    print(f"客户记录数: {len(customer_records)}")
    print(f"CSM记录数: {len(csm_records)}")

    matched_count = 0

    # 遍历客户记录，寻找对应的CSM回复
    for customer_idx in customer_records.index:
        customer_time = customer_records.loc[customer_idx, '客户消息时间']
        customer_id = customer_records.loc[customer_idx, '客户ID']

        if pd.isna(customer_time) or customer_time == '':
            continue

        # 查找相同客户ID和相同时间的CSM记录
        matching_csm = csm_records[
            (csm_records['客户消息时间'] == customer_time) &
            (csm_records['客户ID'] == customer_id)
        ]

        if not matching_csm.empty:
            # 如果有多个匹配的CSM记录，取第一个
            csm_record = matching_csm.iloc[0]

            # 将CSM信息添加到客户记录
            df_copy.loc[customer_idx, 'CSM_AI生成消息'] = str(csm_record.get('AI生成消息', '')) if pd.notna(csm_record.get('AI生成消息', '')) else ''
            df_copy.loc[customer_idx, 'CSM_发送消息内容'] = str(csm_record.get('发送消息内容', '')) if pd.notna(csm_record.get('发送消息内容', '')) else ''
            df_copy.loc[customer_idx, 'CSM_AI与审核结果是否一致'] = str(csm_record.get('AI与审核结果是否一致', '')) if pd.notna(csm_record.get('AI与审核结果是否一致', '')) else ''

            matched_count += 1
            print(f"匹配成功: 客户ID {customer_id} 记录 {customer_idx} 匹配到CSM回复，时间: {customer_time}")

    print(f"总计匹配成功: {matched_count} 条客户消息获得了CSM回复信息")

    return df_copy

def propagate_thought_unit_to_customers(df):
    """
    将发送方为CSM或未发送的thought_unit传播给同一客户ID下的客户记录

    简化逻辑（内容匹配）：
    1. 按客户ID分组，保持原始顺序
    2. 每个客户消息匹配相同内容的CSM/未发送记录
    3. 如果有多个匹配，选择最近的
    4. 确保一对一映射，避免重复分配
    """
    print("\n=== 开始thought_unit传播处理（内容匹配） ===")

    # 添加thought_unit_source列
    if 'thought_unit_source' not in df.columns:
        df['thought_unit_source'] = ''

    # 按客户ID分组处理
    unique_customers = df['客户ID'].unique()
    thought_unit_propagated = 0

    for customer_id in unique_customers:
        customer_data = df[df['客户ID'] == customer_id].copy()

        # 简单按索引排序，保持原始顺序
        customer_data = customer_data.reset_index(drop=False)

        print(f"\n  处理客户 {customer_id}")

        # 收集所有客户记录（无thought_unit）
        customer_records = []
        for idx, row in customer_data.iterrows():
            if (row['发送方'] == '客户' and
                (pd.isna(row['thought_unit']) or row['thought_unit'] == '')):
                customer_records.append({
                    'idx': idx,
                    'df_index': row['index'],
                    'message': row.get('客户消息', ''),
                    'position': idx
                })

        # 收集所有CSM/未发送记录（有thought_unit）
        response_records = []
        for idx, row in customer_data.iterrows():
            if (row['发送方'] in ['CSM', '未发送'] and
                pd.notna(row['thought_unit']) and str(row['thought_unit']) != 'nan'):
                response_records.append({
                    'idx': idx,
                    'sender': row['发送方'],
                    'thought_unit': row['thought_unit'],
                    'message': row.get('客户消息', ''),
                    'position': idx,
                    'used': False
                })

        print(f"    找到 {len(customer_records)} 个客户记录, {len(response_records)} 个响应记录")

        # 精确内容匹配
        for customer_record in customer_records:
            customer_msg = customer_record['message']
            best_match = None
            min_distance = float('inf')

            # 查找完全相同消息的CSM/未发送记录
            for response_record in response_records:
                if (response_record['message'] == customer_msg and
                    not response_record['used']):

                    # 计算距离
                    distance = abs(response_record['position'] - customer_record['position'])

                    if distance < min_distance:
                        min_distance = distance
                        best_match = response_record

            # 如果找到匹配，分配thought_unit
            if best_match:
                df.at[customer_record['df_index'], 'thought_unit'] = best_match['thought_unit']
                df.at[customer_record['df_index'], 'thought_unit_source'] = best_match['sender']
                best_match['used'] = True
                thought_unit_propagated += 1
                print(f"    精确匹配: \"{customer_msg}\" → {best_match['sender']}")
            else:
                print(f"    未匹配: \"{customer_msg}\" → 无对应记录")

    print(f"=== thought_unit传播完成，共传播 {thought_unit_propagated} 条记录 ===")
    return df

def extract_thought_unit_fields(df):
    """
    从thought_unit中提取各种字段信息

    提取字段：
    1. 强遵循标签: endpoint_step == '强遵循' 时为True
    2. FAQ判断: rag_fast_chat_result非空时为True
    3. 知识问答判断: knowledge_scenario_result为True时为True
    4. 销售阶段: 提取node_1st和node_2nd
    5. reference_script: 提取reference_script字段内容
    """
    print("\n=== 开始提取thought_unit字段信息 ===")

    # 添加新列
    new_columns = {
        '强遵循标签': '',
        'FAQ判断': '',
        '知识问答判断': '',
        '销售一级节点': '',
        '销售二级节点': '',
        'reference_script': ''
    }

    for col in new_columns:
        if col not in df.columns:
            df[col] = ''

    extracted_count = 0

    for idx, row in df.iterrows():
        thought_unit = row.get('thought_unit', '')

        if pd.notna(thought_unit) and str(thought_unit) != 'nan' and thought_unit != '':
            try:
                tu_obj = json.loads(str(thought_unit))

                # 1. 强遵循标签判断
                endpoint_step = tu_obj.get('endpoint_step', '')
                df.at[idx, '强遵循标签'] = 'True' if endpoint_step == '强遵循' else 'False'

                # 2. FAQ判断
                rag_fast_result = tu_obj.get('rag_fast_chat_result', '')
                is_faq = False
                if rag_fast_result:
                    # 检查是否为非空（不是空字符串、空列表等）
                    if isinstance(rag_fast_result, str):
                        # 去除空白字符并检查是否为空列表字符串
                        clean_result = rag_fast_result.strip()
                        is_faq = clean_result and clean_result != '[]' and clean_result != ''
                    elif isinstance(rag_fast_result, list):
                        is_faq = len(rag_fast_result) > 0
                    else:
                        is_faq = bool(rag_fast_result)

                df.at[idx, 'FAQ判断'] = 'True' if is_faq else 'False'

                # 3. 知识问答判断
                knowledge_result = tu_obj.get('knowledge_scenario_result', False)
                df.at[idx, '知识问答判断'] = 'True' if knowledge_result is True else 'False'

                # 4. 销售阶段识别
                rag_history_body = tu_obj.get('rag_history_chat_request_body', {})
                if rag_history_body and isinstance(rag_history_body, dict):
                    node_1st = rag_history_body.get('node_1st', '')
                    node_2nd = rag_history_body.get('node_2nd', '')
                    df.at[idx, '销售一级节点'] = node_1st if node_1st else ''
                    df.at[idx, '销售二级节点'] = node_2nd if node_2nd else ''
                else:
                    df.at[idx, '销售一级节点'] = ''
                    df.at[idx, '销售二级节点'] = ''

                # 5. reference_script提取
                reference_script = tu_obj.get('reference_script', '')
                # 处理reference_script可能是列表或字符串的情况
                if isinstance(reference_script, list):
                    df.at[idx, 'reference_script'] = str(reference_script) if reference_script else ''
                else:
                    df.at[idx, 'reference_script'] = str(reference_script) if reference_script else ''

                extracted_count += 1

            except json.JSONDecodeError as e:
                print(f"  第{idx+1}行JSON解析失败: {e}")
                # 设置默认值
                df.at[idx, '强遵循标签'] = 'False'
                df.at[idx, 'FAQ判断'] = 'False'
                df.at[idx, '知识问答判断'] = 'False'
                df.at[idx, '销售一级节点'] = ''
                df.at[idx, '销售二级节点'] = ''
                df.at[idx, 'reference_script'] = ''
            except Exception as e:
                print(f"  第{idx+1}行处理错误: {e}")
                # 设置默认值
                df.at[idx, '强遵循标签'] = 'False'
                df.at[idx, 'FAQ判断'] = 'False'
                df.at[idx, '知识问答判断'] = 'False'
                df.at[idx, '销售一级节点'] = ''
                df.at[idx, '销售二级节点'] = ''
                df.at[idx, 'reference_script'] = ''
        else:
            # 没有thought_unit的记录，设置默认值
            df.at[idx, '强遵循标签'] = 'False'
            df.at[idx, 'FAQ判断'] = 'False'
            df.at[idx, '知识问答判断'] = 'False'
            df.at[idx, '销售一级节点'] = ''
            df.at[idx, '销售二级节点'] = ''
            df.at[idx, 'reference_script'] = ''

    print(f"=== thought_unit字段提取完成，共处理 {extracted_count} 条记录 ===")

    # 统计提取结果
    print("\n提取结果统计:")
    for col in ['强遵循标签', 'FAQ判断', '知识问答判断']:
        true_count = (df[col] == 'True').sum()
        false_count = (df[col] == 'False').sum()
        print(f"  {col}: True={true_count}条, False={false_count}条")

    # 销售节点统计
    node_1st_count = (df['销售一级节点'] != '').sum()
    node_2nd_count = (df['销售二级节点'] != '').sum()
    print(f"  销售一级节点: {node_1st_count}条有值")
    print(f"  销售二级节点: {node_2nd_count}条有值")

    # reference_script统计
    reference_script_count = (df['reference_script'] != '').sum()
    print(f"  reference_script: {reference_script_count}条有值")

    return df

def main():
    # file_path = 'chengla_all_2025-09-14_to_2025-09-18.xlsx'
    file_path = '/Users/luojiatai/Documents/trae1/文件合并/chengla_all_merged.xlsx'
    # file_path = '测试跑1.xlsx'
    # file_path = '回流case_知识中含公告.xlsx'



    try:
        # 读取Excel文件
        df = read_and_process_excel(file_path)

        # 添加CSM回复信息到客户消息
        df = add_csm_replies_to_customer_messages(df)

        # 传播thought_unit给客户记录
        df = propagate_thought_unit_to_customers(df)

        # 提取thought_unit中的字段信息
        df = extract_thought_unit_fields(df)

        # 按销售ID分组处理数据，生成测试集
        test_df, sales_summary, cycle_label_stats = process_data_by_sales_id(df)

        # 显示销售汇总信息
        print("\n=== 销售汇总信息 ===")
        print(sales_summary)

        # 保存测试集和分析结果
        save_test_dataset_results(test_df, sales_summary, cycle_label_stats)

        # 显示一些统计信息
        if not test_df.empty:
            print(f"\n=== 整体统计 ===")
            print(f"总销售数: {test_df['销售ID'].nunique()}")
            print(f"总客户数: {test_df['客户ID'].nunique()}")
            print(f"测试集记录数: {len(test_df)}")
            print(f"有最新客户消息的记录: {len(test_df[test_df['最新客户消息'] != ''])}")
            print(f"问句数量: {len(test_df[test_df['是否是问句'] == '是'])}")
            print(f"有周期标签的记录: {len(test_df[test_df['周期标签'] != ''])}")

            # 显示每个销售的客户分布
            print(f"\n=== 各销售客户数分布 ===")
            sales_customer_count = test_df.groupby(['销售ID', '销售名称'])['客户ID'].nunique().reset_index()
            sales_customer_count.columns = ['销售ID', '销售名称', '客户数']
            sales_customer_count = sales_customer_count.sort_values('客户数', ascending=False)
            for _, row in sales_customer_count.iterrows():
                print(f"  {row['销售名称']}: {row['客户数']}个客户")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
