#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
挖需回流BadCase处理脚本
功能：
1. 处理"挖需回流badcase.xlsx"文件
2. 清理历史对话中的销售信息（将[销售XXX]格式改为[销售]）
3. 截取历史对话到最后一条客户消息
4. 提取RAG内容中的用户信息库和销售信息库数据
5. 生成类似"按销售ID分组的测试集和周期标签分析.xlsx"的格式
"""

import pandas as pd
import numpy as np
import json
import ast
import re
from collections import defaultdict
import sys
import os

class WaxuBadcaseProcessor:
    def __init__(self):
        """初始化挖需回流BadCase处理器"""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.df = None
        self.processed_data = []

    def read_and_process_excel(self, file_path):
        """读取并处理Excel文件"""
        print("正在读取挖需回流badcase.xlsx文件...")

        try:
            self.df = pd.read_excel(file_path)
            print(f"数据总行数: {len(self.df)}")
            print(f"数据列名: {list(self.df.columns)}")
            print("\n前3行数据:")
            print(self.df.head(3))
            return True
        except Exception as e:
            print(f"读取文件失败: {e}")
            return False

    def clean_sales_message_format(self, history_text):
        """
        清理历史对话中的销售消息格式
        将[销售橙啦公考助教乔乔老师~][时间]格式改为[销售][时间]
        将[CSM][时间]格式也改为[销售][时间]
        """
        if pd.isna(history_text) or history_text == '':
            return ''

        history_str = str(history_text)

        # 使用正则表达式匹配[销售XXX][时间]格式，替换为[销售][时间]
        # 匹配模式：[销售...任意内容...][时间戳]
        pattern1 = r'\[销售[^\]]*\](\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])'
        cleaned_history = re.sub(pattern1, r'[销售]\1', history_str)

        # 使用正则表达式匹配[CSM][时间]格式，替换为[销售][时间]
        # 匹配模式：[CSM][时间戳]
        pattern2 = r'\[CSM\](\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])'
        cleaned_history = re.sub(pattern2, r'[销售]\1', cleaned_history)

        return cleaned_history

    def extract_last_customer_message_history(self, history_text):
        """
        提取历史对话到最后一条客户消息为止
        删除最后一条客户消息之后的所有销售消息
        """
        if pd.isna(history_text) or history_text == '':
            return '', ''

        history_str = str(history_text)

        # 先清理销售消息格式
        cleaned_history = self.clean_sales_message_format(history_str)

        # 分割对话消息
        messages = []

        # 使用更加稳健的正则表达式匹配消息块，允许引用消息中出现带[]的内容
        pattern = r'(\[(?:客户|销售)\]\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]:[\s\S]*?)(?=\n\[(?:客户|销售)\]\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]:|\Z)'
        matches = re.findall(pattern, cleaned_history)

        if not matches:
            # 如果正则匹配失败，尝试简单的行分割
            lines = cleaned_history.split('\n')
            current_message = ""
            for line in lines:
                if re.match(r'\[(?:客户|销售)\]\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]:', line):
                    if current_message:
                        messages.append(current_message.strip())
                    current_message = line
                else:
                    if current_message:
                        current_message += "\n" + line
            if current_message:
                messages.append(current_message.strip())
        else:
            messages = matches
        latest_customer_content = ''

        if not messages:
            return cleaned_history, latest_customer_content

        # 找到最后一条客户消息的位置
        last_customer_index = -1
        for i, message in enumerate(messages):
            if message.startswith('[客户]'):
                last_customer_index = i

        if last_customer_index == -1:
            # 如果没有找到客户消息，返回原始内容
            return cleaned_history, latest_customer_content

        # 截取到最后一条客户消息为止
        truncated_messages = messages[:last_customer_index + 1]
        latest_message_raw = messages[last_customer_index]
        latest_customer_content = re.sub(
            r'^\[客户\]\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]:\s*', '', latest_message_raw
        ).strip()

        return '\n'.join(truncated_messages), latest_customer_content

    def extract_rag_content(self, rag_text):
        """
        提取RAG内容中的用户信息库、问卷、试卷以及销售信息库数据
        """
        result = {
            # 用户信息库字段
            '批次ID': '',
            '标签': '',
            '用户昵称': '',
            '问卷内容': '',
            '问卷标题': '',
            '问卷ID': '',
            '试卷名称': '',
            '试卷ID': '',
            '分数': '',
            '天': '',
            '正确率': '',
            '周期批次ID': '',
            '周期标签': '',
            # 销售信息库字段
            '直播课时间': '',
            '激活直播课链接': '',
            '学习档案链接': '',
            '手机平板上课链接': '',
            '电脑上课链接': '',
            '秒题技巧链接': '',
            '报课链接': '',
            '试学链接': '',
            # RAG来源
            'rag来源': 'CSM'
        }

        if pd.isna(rag_text) or rag_text == '':
            return result

        def normalize_content(content):
            """将content统一转换为列表形式，方便遍历"""
            if isinstance(content, list):
                return content
            if isinstance(content, dict):
                return [content]
            if isinstance(content, str):
                text = content.strip()
                if not text:
                    return []
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return [parsed]
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, dict):
                        return [parsed]
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    return []
            return []

        def update_sales_info(sales_data, info_dict):
            """更新销售信息字段"""
            fields = [
                '直播课时间',
                '激活直播课链接',
                '学习档案链接',
                '手机平板上课链接',
                '电脑上课链接',
                '秒题技巧链接',
                '报课链接',
                '试学链接'
            ]
            for field in fields:
                value = sales_data.get(field, '')
                if value:
                    info_dict[field] = value
            return info_dict

        rag_data = {}
        try:
            if isinstance(rag_text, dict):
                rag_data = rag_text
            elif isinstance(rag_text, str):
                text = rag_text.strip()
                if text:
                    try:
                        rag_data = json.loads(text)
                    except Exception:
                        try:
                            rag_data = ast.literal_eval(text)
                        except Exception:
                            rag_data = {}
        except Exception:
            rag_data = {}

        if not isinstance(rag_data, dict):
            return result

        recall_items = []
        if 'data' in rag_data and isinstance(rag_data['data'], dict):
            recall_items = rag_data['data'].get('recall', [])
        elif 'recall' in rag_data:
            recall_items = rag_data.get('recall', [])

        if not isinstance(recall_items, list):
            recall_items = []

        for item in recall_items:
            if not isinstance(item, dict):
                continue

            db_name = item.get('db_name', '')
            content = item.get('content', [])
            source = item.get('source')
            if source:
                result['rag来源'] = source

            content_list = normalize_content(content)

            if db_name == '用户信息库':
                for user_data in content_list:
                    if not isinstance(user_data, dict):
                        continue
                    batch_id = user_data.get('批次ID', '')
                    cycle_batch_id = user_data.get('周期批次ID', '')
                    cycle_label = user_data.get('周期标签', '')

                    if batch_id and not result['批次ID']:
                        result['批次ID'] = batch_id
                    if cycle_batch_id:
                        result['周期批次ID'] = cycle_batch_id
                    elif batch_id and not result['周期批次ID']:
                        result['周期批次ID'] = batch_id
                    if cycle_label:
                        result['周期标签'] = cycle_label
                    tag_value = user_data.get('标签')
                    if isinstance(tag_value, str) and tag_value and not result['标签']:
                        result['标签'] = tag_value
                    nickname = user_data.get('用户昵称')
                    if isinstance(nickname, str) and nickname and not result['用户昵称']:
                        result['用户昵称'] = nickname

            elif db_name == '销售信息库':
                for sales_data in content_list:
                    if isinstance(sales_data, dict):
                        update_sales_info(sales_data, result)

            elif db_name == '调查问卷库':
                for questionnaire_data in content_list:
                    if not isinstance(questionnaire_data, dict):
                        continue
                    content_value = questionnaire_data.get('问卷内容')
                    if content_value and not result['问卷内容']:
                        if isinstance(content_value, (dict, list)):
                            result['问卷内容'] = json.dumps(content_value, ensure_ascii=False)
                        else:
                            result['问卷内容'] = str(content_value)
                    title = questionnaire_data.get('问卷标题', '')
                    if title and not result['问卷标题']:
                        result['问卷标题'] = title
                    questionnaire_id = questionnaire_data.get('问卷ID', '')
                    if questionnaire_id and not result['问卷ID']:
                        result['问卷ID'] = questionnaire_id

            elif db_name == '试卷库':
                for exam_data in content_list:
                    if not isinstance(exam_data, dict):
                        continue
                    if exam_data.get('试卷名称') and not result['试卷名称']:
                        result['试卷名称'] = exam_data.get('试卷名称', '')
                    if exam_data.get('试卷ID') and not result['试卷ID']:
                        result['试卷ID'] = exam_data.get('试卷ID', '')
                    if exam_data.get('分数') and not result['分数']:
                        result['分数'] = str(exam_data.get('分数', ''))
                    if exam_data.get('天') and not result['天']:
                        result['天'] = str(exam_data.get('天', ''))
                    if exam_data.get('正确率') and not result['正确率']:
                        result['正确率'] = str(exam_data.get('正确率', ''))

        return result

    def is_question_keyword_based(self, message):
        """基于更精准的规则判断消息是否为问句"""
        if pd.isna(message) or message == '':
            return '否'

        message_str = str(message).strip()

        if not message_str:
            return '否'

        # 1. 直接包含问号
        if any(ch in message_str for ch in ('?', '？')):
            return '是'

        # 标准化尾部标点
        stripped = re.sub(r'[。．\.…!！；;,\s]+$', '', message_str)

        # 2. 常见问句结尾粒度（要求出现在末尾以减少误报）
        ending_patterns = [
            r'(吗|嘛|么|呢|吧)$',
            r'(对吗|对吧|好不好|要不要|行不行|是不是|可不可以|能不能|行吗|行么|行嘛|好吗|好么|好嘛)$'
        ]
        for pattern in ending_patterns:
            if re.search(pattern, stripped):
                return '是'

        # 3.a 以能力/许可类词汇开头的疑问
        if stripped.startswith(('能否', '是否', '可否', '能不能')):
            return '是'

        # 3.b 以疑问代词开头的结构
        interrogative_starts = ('什么', '怎么', '怎样', '为何', '为什么', '哪个', '哪种', '哪些',
                                '哪里', '哪儿', '哪家', '哪位', '谁', '何时', '几时', '多少', '多久',
                                '多长', '多远', '多大', '多高', '多重', '几岁', '几天', '几月', '几号')
        for start in interrogative_starts:
            if stripped.startswith(start):
                return '是'

        # 3. 组合问句结构
        combo_patterns = [
            r'(?:可不可以|能不能|要不要|好不好|行不行)',
            r'(?:能否|是否|可否)[^。！？]*?(?:吗|呢|吧|\?|？)'
        ]
        for pattern in combo_patterns:
            if re.search(pattern, message_str):
                return '是'

        # 4. 含“请问”且伴随能力/许可类动词的句式
        if '请问' in message_str and re.search(r'(?:可以|能|能否|是否|可否)', message_str):
            return '是'

        return '否'

    def extract_thought_unit_info(self, thought_unit_text):
        """提取thought_unit中的周期标签信息"""
        if pd.isna(thought_unit_text) or thought_unit_text == '':
            return {'周期标签': ''}

        try:
            if isinstance(thought_unit_text, str):
                thought_data = json.loads(thought_unit_text)
            else:
                thought_data = thought_unit_text

            # 从rag_chat_request_body中提取周期标签
            if 'rag_chat_request_body' in thought_data:
                rag_request = thought_data['rag_chat_request_body']
                if 'contexts' in rag_request:
                    # 这里可以进一步处理contexts来提取周期信息
                    pass

            # 从rag_chat_result中提取用户信息
            if 'rag_chat_result' in thought_data:
                rag_result = thought_data['rag_chat_result']
                if isinstance(rag_result, str):
                    try:
                        rag_result_data = json.loads(rag_result)
                        for item in rag_result_data:
                            if item.get('db_name') == '用户信息库':
                                content = item.get('content', [])
                                if content and len(content) > 0:
                                    user_data = content[0]
                                    cycle_label = user_data.get('周期标签', '')
                                    if cycle_label:
                                        return {'周期标签': cycle_label}
                    except:
                        pass

        except Exception as e:
            print(f"解析thought_unit数据时出错: {str(e)}")

        return {'周期标签': ''}

    def process_data_by_sales_id(self):
        """按销售ID分组处理数据，生成测试集格式"""
        print("\n=== 开始处理挖需回流badcase数据 ===")

        if self.df is None or self.df.empty:
            print("❌ 没有可用的数据")
            return False

        # 获取所有唯一的销售ID
        unique_sales_ids = self.df['销售ID'].unique()
        print(f"共发现 {len(unique_sales_ids)} 个销售ID")

        all_processed_data = []
        sales_summary = []

        for sales_id in unique_sales_ids:
            print(f"\n处理销售ID: {sales_id[:30]}...")

            # 筛选当前销售的数据
            sales_data = self.df[self.df['销售ID'] == sales_id].copy()
            print(f"  原始记录数: {len(sales_data)}")

            # 获取销售名称
            sales_name = sales_data['销售名称'].iloc[0] if not sales_data.empty and '销售名称' in sales_data.columns else "未知"

            # 按客户ID分组处理
            customers_in_sales = sales_data['客户ID'].unique()
            print(f"  客户数量: {len(customers_in_sales)}")

            processed_count = 0

            for customer_id in customers_in_sales:
                # 获取该客户的所有记录
                customer_data = sales_data[sales_data['客户ID'] == customer_id].copy()

                # 为每条记录创建一个测试用例
                for _, record in customer_data.iterrows():
                    try:
                        # 处理历史对话并提取最新客户消息
                        cleaned_history, latest_customer_msg = self.extract_last_customer_message_history(record['历史对话'])
                        original_customer_msg = str(record.get('客户消息', '') or '').strip()
                        latest_customer_msg = latest_customer_msg if latest_customer_msg else original_customer_msg

                        # 提取RAG内容
                        rag_data = self.extract_rag_content(record['rag'])

                        # 提取thought_unit信息
                        thought_info = self.extract_thought_unit_info(record['thought_unit'])

                        # 判断是否是问句
                        is_question = self.is_question_keyword_based(latest_customer_msg)

                        # 周期标签优先使用RAG结果，其次thought_unit
                        cycle_label = rag_data.get('周期标签') or thought_info.get('周期标签', '')

                        # 构建处理后的记录
                        processed_record = {
                            '销售ID': record['销售ID'],
                            '客户ID': record['客户ID'],
                            '发送方': record.get('发送方', ''),
                            '历史对话': cleaned_history,
                            '原始客户消息': original_customer_msg,
                            '最新客户消息': latest_customer_msg,
                            'rag': record.get('rag', ''),
                            'thought_unit': record.get('thought_unit', ''),
                            '周期标签': cycle_label,
                            '是否是问句': is_question,
                            '客户消息时间': record.get('客户消息时间', ''),
                            '发送时间': record.get('发送时间', ''),
                            'rag来源': rag_data['rag来源'],
                            # 原表中的重要字段
                            '当前销售阶段': record.get('当前销售阶段', ''),
                            '当前销售动作': record.get('当前销售动作', ''),
                            '回复策略': record.get('回复策略', ''),
                            'AI生成消息': record.get('AI生成消息', ''),
                            '发送消息内容': record.get('发送消息内容', ''),
                            '质检结果': record.get('质检结果', ''),
                            '质检原因': record.get('质检原因', ''),
                            # 用户信息库字段
                            '批次ID': rag_data['批次ID'],
                            '问卷内容': rag_data['问卷内容'],
                            '问卷标题': rag_data['问卷标题'],
                            '问卷ID': rag_data['问卷ID'],
                            '试卷名称': rag_data['试卷名称'],
                            '试卷ID': rag_data['试卷ID'],
                            '分数': rag_data['分数'],
                            '天': rag_data['天'],
                            '正确率': rag_data['正确率'],
                            '周期批次ID': rag_data['周期批次ID'],
                            '标签': rag_data['标签'],
                            '用户昵称': rag_data['用户昵称'],
                            # 销售信息库字段
                            '直播课时间': rag_data['直播课时间'],
                            '激活直播课链接': rag_data['激活直播课链接'],
                            '学习档案链接': rag_data['学习档案链接'],
                            '手机平板上课链接': rag_data['手机平板上课链接'],
                            '电脑上课链接': rag_data['电脑上课链接'],
                            '秒题技巧链接': rag_data['秒题技巧链接'],
                            '报课链接': rag_data['报课链接'],
                            '试学链接': rag_data['试学链接'],
                            # 预期回复字段（保留为空，后续可以填充）
                            '最后销售消息': [],
                            '预期销售回复': '',
                            '备选销售回复': []
                        }

                        all_processed_data.append(processed_record)
                        processed_count += 1

                    except Exception as e:
                        print(f"  ❌ 处理记录失败: {e}")
                        continue

            # 统计信息
            sales_summary.append({
                '销售ID': sales_id,
                '销售名称': sales_name,
                '客户数量': len(customers_in_sales),
                '记录数': len(sales_data),
                '处理成功数': processed_count
            })

            print(f"  处理完成: {processed_count} 条记录")

        self.processed_data = all_processed_data

        print(f"\n=== 处理完成 ===")
        print(f"总处理记录数: {len(all_processed_data)}")

        return True

    def save_results(self, output_file="1104挖需修改数据-yyds.xlsx"):
        """保存处理结果"""
        if not self.processed_data:
            print("❌ 没有处理后的数据可保存")
            return False

        try:
            # 转换为DataFrame
            test_df = pd.DataFrame(self.processed_data)

            # 提取thought_unit字段信息
            test_df = extract_thought_unit_fields(test_df)
            self.processed_data = test_df.to_dict('records')

            # 标准格式列（包含原表重要字段）
            standard_columns = [
                '销售ID', '客户ID', '发送方', '历史对话', '原始客户消息', '最新客户消息', 'rag', 'thought_unit',
                '强遵循标签', 'FAQ判断', '知识问答判断', '销售一级节点', '销售二级节点', 'reference_script',
                '周期标签', '是否是问句', '客户消息时间', '发送时间', 'rag来源',
                # 原表中的重要字段
                '当前销售阶段', '当前销售动作', '回复策略', 'AI生成消息', '发送消息内容', '质检结果', '质检原因',
                # RAG提取的字段
                '批次ID', '问卷内容', '问卷标题', '问卷ID', '试卷名称', '试卷ID', '分数', '天', '正确率', '周期批次ID',
                '标签', '用户昵称',
                '直播课时间', '激活直播课链接', '学习档案链接', '手机平板上课链接', '电脑上课链接',
                '秒题技巧链接', '报课链接', '试学链接', '最后销售消息', '预期销售回复', '备选销售回复'
            ]

            # 确保所有列都存在
            for col in standard_columns:
                if col not in test_df.columns:
                    test_df[col] = ''

            standard_test_df = test_df[standard_columns].copy()

            # 保存到Excel文件
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 标准格式测试集
                standard_test_df.to_excel(writer, sheet_name='测试集_标准格式', index=False)

                # 完整信息测试集
                test_df.to_excel(writer, sheet_name='测试集_完整信息', index=False)

            print(f"✅ 处理结果已保存到: {output_file}")
            print(f"包含工作表:")
            print(f"  - 测试集_标准格式: {len(standard_test_df)} 条记录")
            print(f"  - 测试集_完整信息: {len(test_df)} 条记录")

            return True

        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
            return False

def extract_thought_unit_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    从thought_unit字段中提取强遵循、FAQ判断、知识问答判断、销售节点和reference_script信息
    """
    target_columns = {
        '强遵循标签': 'False',
        'FAQ判断': 'False',
        '知识问答判断': 'False',
        '销售一级节点': '',
        '销售二级节点': '',
        'reference_script': ''
    }

    for col, default in target_columns.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    if 'thought_unit' not in df.columns:
        df['thought_unit'] = ''
        return df

    for idx, row in df.iterrows():
        thought_unit = row.get('thought_unit', '')

        # 默认值
        df.at[idx, '强遵循标签'] = 'False'
        df.at[idx, 'FAQ判断'] = 'False'
        df.at[idx, '知识问答判断'] = 'False'
        df.at[idx, '销售一级节点'] = ''
        df.at[idx, '销售二级节点'] = ''
        df.at[idx, 'reference_script'] = ''

        if pd.isna(thought_unit) or str(thought_unit).strip() == '':
            continue

        try:
            tu_obj = json.loads(str(thought_unit))
        except json.JSONDecodeError:
            continue
        except Exception:
            continue

        # 1. 强遵循标签
        endpoint_step = tu_obj.get('endpoint_step', '')
        if endpoint_step == '强遵循':
            df.at[idx, '强遵循标签'] = 'True'

        # 2. FAQ判断
        rag_fast_result = tu_obj.get('rag_fast_chat_result', '')
        is_faq = False
        if isinstance(rag_fast_result, str):
            clean_result = rag_fast_result.strip()
            is_faq = bool(clean_result and clean_result != '[]')
        elif isinstance(rag_fast_result, list):
            is_faq = len(rag_fast_result) > 0
        elif rag_fast_result:
            is_faq = True
        df.at[idx, 'FAQ判断'] = 'True' if is_faq else 'False'

        # 3. 知识问答判断
        knowledge_result = tu_obj.get('knowledge_scenario_result', False)
        df.at[idx, '知识问答判断'] = 'True' if knowledge_result is True else 'False'

        # 4. 销售节点
        rag_history_body = tu_obj.get('rag_history_chat_request_body', {})
        if isinstance(rag_history_body, dict):
            node_1st = rag_history_body.get('node_1st', '')
            node_2nd = rag_history_body.get('node_2nd', '')
            if node_1st:
                df.at[idx, '销售一级节点'] = node_1st
            if node_2nd:
                df.at[idx, '销售二级节点'] = node_2nd

        # 5. reference_script
        reference_script = tu_obj.get('reference_script', '')
        if isinstance(reference_script, list):
            df.at[idx, 'reference_script'] = json.dumps(reference_script, ensure_ascii=False)
        elif reference_script:
            df.at[idx, 'reference_script'] = str(reference_script)

    return df


def main():
    """主函数"""
    print("🤖 挖需回流BadCase处理脚本")
    print("作者: Claude Code")
    print("版本: v1.0")
    print()

    # 创建处理器实例
    processor = WaxuBadcaseProcessor()

    # 读取并处理Excel文件
    input_file = "1104挖需修改数据2.xlsx"
    if not processor.read_and_process_excel(input_file):
        print("💥 读取文件失败")
        return 1

    # 处理数据
    if not processor.process_data_by_sales_id():
        print("💥 数据处理失败")
        return 1

    # 保存结果
    if not processor.save_results():
        print("💥 保存结果失败")
        return 1

    print("\n🎉 挖需回流BadCase处理完成！")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
