"""
真实销售场景测试用例筛选器
基于销售ID分组和周期标签，还原真实线上测试场景的完整覆盖
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
import re
from datetime import datetime

# 注意：所有数据提取函数已移除，因为我们直接使用pre_process_improved.py生成的现成列

class RealisticSalesTestSelector:
    def __init__(self):
        """初始化真实场景测试筛选器"""
        self.df = None
        self.test_cases = []

        # 真实测试场景维度
        self.scenario_weights = {
            'cycle_coverage': 0.4,        # 周期标签覆盖（最重要）
            'sales_id_coverage': 0.25,    # 销售ID覆盖
            'message_pattern': 0.2,       # 消息模式多样性
            'question_coverage': 0.1,     # 问句覆盖
            'rag_source_coverage': 0.05   # RAG来源覆盖
        }

    def load_data(self, file_path):
        """加载真实测试数据"""
        print("正在加载真实测试数据...")
        # 从包含最后销售消息的标准格式工作表加载数据
        self.df = pd.read_excel(file_path, sheet_name='测试集_标准格式')

        print(f"原始数据量: {len(self.df)}")

        # 过滤空消息和无效消息
        self.df = self._filter_valid_messages(self.df)

        print(f"过滤后数据量: {len(self.df)}")
        print(f"销售ID数量: {self.df['销售ID'].nunique()}")
        print(f"客户ID数量: {self.df['客户ID'].nunique()}")

        # 数据预处理
        self._preprocess_data()

    def _filter_valid_messages(self, df):
        """过滤有效的客户消息"""
        print("正在过滤无效消息...")

        # 过滤空消息
        df = df[~(df['最新客户消息'].isna() |
                 (df['最新客户消息'].astype(str).str.strip() == '') |
                 (df['最新客户消息'].astype(str) == 'nan'))]

        # 统计极短消息
        short_messages = df[df['最新客户消息'].astype(str).str.len() <= 1]
        if len(short_messages) > 0:
            print(f"发现 {len(short_messages)} 条极短消息（≤1字符），已保留")

        # 过滤一些明显无意义的消息
        df = self._filter_meaningless_messages(df)

        # 添加时间过滤：仅对周期标签为{'行课期': 1}且发送时间超过17:20的记录进行过滤
        df = self._filter_by_time_for_specific_cycle(df)

        original_count = len(pd.read_excel('按销售ID分组的测试集和周期标签分析_with_thought_unit_final.xlsx'))
        print(f"过滤掉 {original_count - len(df)} 条无效消息")
        return df

    def _filter_meaningless_messages(self, df):
        """过滤明显无意义的消息"""
        original_count = len(df)

        # 过滤设备相关消息（手机、平板、电脑等上课设备问题）
        device_keywords = ['手机', '平板', '电脑', 'ipad', 'pc', '设备', '怎么上课', '在哪上课', '如何上课']
        device_pattern = '|'.join(device_keywords)
        device_messages = df['最新客户消息'].astype(str).str.contains(device_pattern, case=False, na=False)

        # 过滤图片相关消息（减少图片类消息）
        image_messages = df['最新客户消息'].astype(str).str.contains('<image|图片|截图', case=False, na=False)

        # 过滤特定的系统消息
        specific_message = "我已经添加了你，现在我们可以开始聊天了。"
        system_messages = df['最新客户消息'].astype(str).str.strip() == specific_message

        # 保留一些设备消息作为参考，但大幅减少
        device_sample = df[device_messages].sample(n=min(10, len(df[device_messages])), random_state=42) if len(df[device_messages]) > 0 else df[device_messages]
        image_sample = df[image_messages].sample(n=min(5, len(df[image_messages])), random_state=42) if len(df[image_messages]) > 0 else df[image_messages]

        # 过滤掉设备消息、图片消息和特定系统消息
        df_filtered = df[~(device_messages | image_messages | system_messages)]

        # 添加少量设备和图片样本回来（但不包含系统消息）
        df_final = pd.concat([df_filtered, device_sample, image_sample]).drop_duplicates()

        # 统计各类过滤的数量
        system_count = sum(system_messages)
        device_count = sum(device_messages) - len(device_sample)  # 减去保留的样本
        image_count = sum(image_messages) - len(image_sample)     # 减去保留的样本

        filtered_count = original_count - len(df_final)
        if filtered_count > 0:
            print(f"过滤掉 {filtered_count} 条消息：系统消息({system_count}条) + 设备消息({device_count}条) + 图片消息({image_count}条)（保留少量设备/图片样本）")

        return df_final

    def _filter_by_time_for_specific_cycle(self, df):
        """时间过滤：仅对周期标签为{'行课期': 1}且发送时间超过17:20的记录进行过滤"""
        if '发送时间' not in df.columns or '周期标签' not in df.columns:
            print("⚠️ 缺少发送时间或周期标签列，跳过时间过滤")
            return df

        original_count = len(df)

        def should_filter_by_time(row):
            """判断是否应该根据时间过滤该记录"""
            try:
                # 检查周期标签是否为{'行课期': 1}
                cycle_label = row.get('周期标签', '')
                if pd.isna(cycle_label) or cycle_label == '':
                    return False

                # 解析周期标签
                try:
                    if isinstance(cycle_label, str):
                        cycle_dict = eval(cycle_label)
                    else:
                        cycle_dict = cycle_label

                    # 只对{'行课期': 1}进行时间过滤
                    if not (isinstance(cycle_dict, dict) and
                           cycle_dict.get('行课期') == 1 and
                           len(cycle_dict) == 1):
                        return False  # 不是{'行课期': 1}，不过滤

                except:
                    return False  # 解析失败，不过滤

                # 检查发送时间是否超过17:20
                send_time = row.get('发送时间', '')
                if pd.isna(send_time) or send_time == '':
                    return False  # 空时间，不过滤

                try:
                    # 处理各种时间格式
                    if isinstance(send_time, str):
                        # 提取时间部分，如 "2025-09-18 18:51:53" -> "18:51:53"
                        if ' ' in send_time:
                            time_part = send_time.split(' ')[1]
                        else:
                            time_part = send_time

                        # 提取小时和分钟
                        time_components = time_part.split(':')
                        if len(time_components) >= 2:
                            hour = int(time_components[0])
                            minute = int(time_components[1])

                            # 检查是否超过17:20
                            if hour > 17 or (hour == 17 and minute > 20):
                                return True  # 超过17:20，需要过滤
                            return False

                    elif hasattr(send_time, 'hour'):  # pandas Timestamp 或 datetime 对象
                        hour = send_time.hour
                        minute = send_time.minute
                        if hour > 17 or (hour == 17 and minute > 20):
                            return True  # 超过17:20，需要过滤
                        return False

                    return False  # 其他格式不过滤

                except (ValueError, AttributeError, IndexError):
                    return False  # 解析失败不过滤

            except Exception:
                return False  # 任何异常都不过滤

        # 应用时间过滤：保留不需要过滤的记录
        df_filtered = df[~df.apply(should_filter_by_time, axis=1)].copy()
        filtered_count = original_count - len(df_filtered)

        if filtered_count > 0:
            print(f"时间过滤: 对行课期1记录应用17:20后过滤，移除 {filtered_count} 条记录")
        else:
            print("时间过滤: 无需过滤的记录（非行课期1或时间在17:20前）")

        return df_filtered

    def _filter_media_only_sales_messages(self, df):
        """过滤最后销售消息只包含媒体文件的记录"""
        def is_media_only_sales_messages(last_sales_messages):
            """判断销售消息列表是否只包含媒体文件"""
            if pd.isna(last_sales_messages) or last_sales_messages == [] or last_sales_messages == '[]':
                return False

            try:
                # 如果是字符串，尝试解析为列表
                if isinstance(last_sales_messages, str):
                    if last_sales_messages.strip() == '[]':
                        return False
                    messages_list = eval(last_sales_messages)
                else:
                    messages_list = last_sales_messages

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

        # 应用过滤条件
        media_only_mask = df['最后销售消息'].apply(is_media_only_sales_messages)

        # 显示一些样本用于调试
        media_only_samples = df[media_only_mask]['最后销售消息'].head(3)
        if len(media_only_samples) > 0:
            print("被过滤的媒体文件样本:")
            for i, sample in enumerate(media_only_samples):
                print(f"  样本{i+1}: {sample}")

        # 返回过滤后的数据（保留非纯媒体的记录）
        return df[~media_only_mask]

    def _preprocess_data(self):
        """数据预处理"""
        print("正在预处理数据...")

        # 解析周期标签
        self.df['周期标签-解析'] = self.df['周期标签'].apply(self._parse_cycle_label)

        # 消息长度
        self.df['消息长度'] = self.df['最新客户消息'].astype(str).str.len()

        # 消息类型分类
        self.df['消息类型'] = self.df.apply(self._classify_message_pattern, axis=1)

        # 销售阶段（从thought_unit中提取）
        self.df['销售阶段'] = self.df.apply(self._extract_sales_stage, axis=1)

        # 历史对话复杂度
        self.df['对话复杂度'] = self.df['历史对话'].astype(str).str.len()
        self.df['对话轮次'] = self.df['历史对话'].astype(str).str.count('\[客户\]')

        # 检查是否有最后销售消息列
        if '最后销售消息' not in self.df.columns:
            print("警告：数据中没有找到'最后销售消息'列，请确保使用的是process_excel_by_sales.py生成的完整数据")
        else:
            # 过滤最后销售消息只包含媒体文件的记录
            original_count = len(self.df)
            self.df = self._filter_media_only_sales_messages(self.df)
            filtered_count = original_count - len(self.df)
            if filtered_count > 0:
                print(f"过滤掉 {filtered_count} 条最后销售消息只包含媒体文件的记录")

    def _parse_cycle_label(self, cycle_str):
        """解析周期标签"""
        try:
            if pd.isna(cycle_str):
                return {'未知': 1}
            cycle_dict = eval(cycle_str) if isinstance(cycle_str, str) else cycle_str
            return cycle_dict if isinstance(cycle_dict, dict) else {'未知': 1}
        except:
            return {'未知': 1}

    def _classify_message_pattern(self, row):
        """消息模式分类（基于真实场景）"""
        content = str(row['最新客户消息']).lower()

        # 基于真实销售场景的分类
        if any(word in content for word in ['好的', '好', '可以', '是的', 'ok', '收到']):
            return '确认回复'
        elif any(word in content for word in ['不', '没', '不要', '不需要', '不行']):
            return '拒绝回复'
        elif row['是否是问句'] == '是':
            return '疑问询问'
        elif any(word in content for word in ['谢谢', '感谢', '辛苦']):
            return '感谢表达'
        elif len(content) > 50:
            return '详细描述'
        elif any(word in content for word in ['了解', '明白', '清楚', '知道']):
            return '理解确认'
        elif content in ['我已经添加了你，现在我们可以开始聊天了。']:
            return '系统消息'
        # 移除设备相关分类，这些消息在前面已经被过滤
        else:
            return '其他回复'

    def _extract_sales_stage(self, row):
        """提取销售阶段信息"""
        try:
            thought_unit = row.get('thought_unit', '')
            if pd.isna(thought_unit) or thought_unit == '':
                return '未知阶段'

            if isinstance(thought_unit, str):
                # 尝试解析JSON
                try:
                    thought_data = json.loads(thought_unit)
                    if 'sales_stage' in thought_data:
                        stage_info = thought_data['sales_stage']
                        if isinstance(stage_info, dict):
                            return stage_info.get('当前销售阶段', '未知阶段')
                except:
                    pass

                # 从文本中提取阶段信息
                if '挖需' in thought_unit:
                    return '挖需阶段'
                elif '转化' in thought_unit:
                    return '转化阶段'
                elif '服务' in thought_unit:
                    return '服务阶段'
                elif '跟进' in thought_unit:
                    return '跟进阶段'

            return '未知阶段'
        except:
            return '未知阶段'

    def analyze_real_scenarios(self):
        """分析真实场景分布"""
        print("\n=== 真实场景分布分析 ===")

        # 1. 详细周期标签分布分析
        print("1. 详细周期标签分布:")
        cycle_detailed_analysis = self._analyze_detailed_cycle_labels()

        for cycle_detail, info in cycle_detailed_analysis.items():
            print(f"  {cycle_detail}: {info['count']}条 ({info['percentage']:.1f}%)")

        # 2. 销售ID场景分布
        print("\n2. 销售ID分布:")
        sales_dist = self.df['销售ID'].value_counts().head(10)
        for sales_id, count in sales_dist.items():
            print(f"  {sales_id[:20]}...: {count}条")

        # 3. 消息模式分布
        print("\n3. 消息模式分布:")
        pattern_dist = self.df['消息类型'].value_counts()
        for pattern, count in pattern_dist.items():
            print(f"  {pattern}: {count}条")

        # 4. 问句比例分析
        print("\n4. 问句分析:")
        question_dist = self.df['是否是问句'].value_counts()
        print(f"  问句: {question_dist.get('是', 0)}条")
        print(f"  非问句: {question_dist.get('否', 0)}条")

        # 5. RAG来源分布
        print("\n5. RAG来源分布:")
        rag_dist = self.df['rag来源'].value_counts()
        for source, count in rag_dist.items():
            print(f"  {source}: {count}条")

        return {
            'cycle_detailed_analysis': cycle_detailed_analysis,
            'sales_dist': sales_dist,
            'pattern_dist': pattern_dist,
            'question_dist': question_dist,
            'rag_dist': rag_dist
        }

    def _analyze_detailed_cycle_labels(self):
        """详细分析周期标签，包含具体天数信息"""
        detailed_analysis = {}
        total_count = len(self.df)

        for cycle_label in self.df['周期标签-解析']:
            # 生成详细的周期描述
            cycle_description = self._generate_cycle_description(cycle_label)

            if cycle_description not in detailed_analysis:
                detailed_analysis[cycle_description] = {'count': 0, 'percentage': 0}

            detailed_analysis[cycle_description]['count'] += 1

        # 计算百分比
        for cycle_desc in detailed_analysis:
            count = detailed_analysis[cycle_desc]['count']
            detailed_analysis[cycle_desc]['percentage'] = (count / total_count) * 100

        # 按数量排序
        sorted_analysis = dict(sorted(detailed_analysis.items(),
                                    key=lambda x: x[1]['count'], reverse=True))

        return sorted_analysis

    def _generate_cycle_description(self, cycle_label):
        """生成详细的周期描述"""
        if not cycle_label or not isinstance(cycle_label, dict):
            return "未知周期"

        descriptions = []

        # 处理行课期
        if '行课期' in cycle_label:
            day = cycle_label['行课期']
            if day == 0:
                descriptions.append("行课期第0天")
            else:
                descriptions.append(f"行课期第{day}天")

        # 处理服务期
        if '服务期' in cycle_label:
            day = cycle_label['服务期']
            if day == 0:
                descriptions.append("服务期第0天")
            else:
                descriptions.append(f"服务期第{day}天")

        # 处理其他周期类型
        for cycle_type, day in cycle_label.items():
            if cycle_type not in ['行课期', '服务期']:
                if day == 0:
                    descriptions.append(f"{cycle_type}第0天")
                else:
                    descriptions.append(f"{cycle_type}第{day}天")

        # 如果没有识别到任何周期，返回原始标签
        if not descriptions:
            return str(cycle_label)

        # 合并多个周期描述
        return " + ".join(descriptions)

    def select_realistic_test_cases(self, target_total=1000, use_proportional=True):
        """
        选择测试用例 - 简化版本
        只考虑周期占比，大于100条随机选100条，小于100条按比例，小于10条全选
        """
        return self._select_by_simple_cycle_proportion()

    def _select_by_simple_cycle_proportion(self):
        """
        两阶段筛选算法：
        总目标：300条
        第1阶段：每个周期标签保底5条，不足5条全选
        第2阶段：剩余名额按占比分配
        """
        TARGET_TOTAL = 300
        GUARANTEED_MIN = 5

        print(f"\n=== 开始两阶段筛选 ===")
        print(f"目标总数: {TARGET_TOTAL}条")
        print(f"每个周期标签保底: {GUARANTEED_MIN}条")

        # 1. 统计详细周期标签分布
        detailed_cycles = self._analyze_detailed_cycle_labels()
        total_original = len(self.df)

        print(f"原始数据总量: {total_original}")

        # 第1阶段：计算保底分配
        print(f"\n=== 第1阶段：保底分配 ===")
        phase1_allocation = {}
        total_guaranteed = 0

        for cycle_desc, info in detailed_cycles.items():
            original_count = info['count']
            proportion = original_count / total_original

            if original_count < GUARANTEED_MIN:
                # 不足保底数量，全选
                guaranteed = original_count
                strategy = f"全选({original_count}条)"
            else:
                # 达到保底数量
                guaranteed = GUARANTEED_MIN
                strategy = f"保底({GUARANTEED_MIN}条)"

            phase1_allocation[cycle_desc] = {
                'original_count': original_count,
                'proportion': proportion,
                'guaranteed': guaranteed,
                'strategy': strategy
            }
            total_guaranteed += guaranteed

            print(f"{cycle_desc:35} | 原始:{original_count:4}条 | 保底:{guaranteed:3}条 | {strategy}")

        print(f"第1阶段总保底: {total_guaranteed} 条")

        # 第2阶段：剩余名额按占比分配
        remaining_quota = TARGET_TOTAL - total_guaranteed
        print(f"\n=== 第2阶段：剩余{remaining_quota}条按占比分配 ===")

        final_allocation = {}
        total_final = 0

        # 只考虑有剩余分配潜力的周期标签（原始数量 > 保底数量）
        eligible_cycles = {}
        eligible_total_original = 0

        for cycle_desc, phase1_info in phase1_allocation.items():
            if phase1_info['original_count'] > phase1_info['guaranteed']:
                eligible_cycles[cycle_desc] = phase1_info
                eligible_total_original += phase1_info['original_count']

        for cycle_desc, phase1_info in phase1_allocation.items():
            guaranteed = phase1_info['guaranteed']
            original_count = phase1_info['original_count']

            if cycle_desc in eligible_cycles and eligible_total_original > 0:
                # 按有效占比分配剩余名额
                eligible_proportion = original_count / eligible_total_original
                additional = int(remaining_quota * eligible_proportion)
                final_count = guaranteed + additional

                # 确保不超过原始数量
                final_count = min(final_count, original_count)
                actual_additional = final_count - guaranteed
                strategy = f"保底{guaranteed}+按比例{actual_additional}={final_count}"
            else:
                # 没有剩余分配潜力，只有保底数量
                final_count = guaranteed
                strategy = f"仅保底{guaranteed}"

            final_allocation[cycle_desc] = {
                'original_count': original_count,
                'proportion': phase1_info['proportion'],
                'guaranteed': guaranteed,
                'final_count': final_count,
                'strategy': strategy
            }
            total_final += final_count

            print(f"{cycle_desc:35} | 原始:{original_count:4}条 | 最终:{final_count:3}条 | {strategy}")

        print(f"最终计划总数: {total_final} 条")

        # 3. 执行筛选
        print(f"\n=== 执行筛选 ===")
        selected_cases = []
        self.selection_stats = {}

        for cycle_desc, plan in final_allocation.items():
            if plan['final_count'] == 0:
                continue

            # 获取该周期标签的所有数据
            cycle_data = self._get_data_by_cycle_description(cycle_desc)

            if len(cycle_data) == 0:
                continue

            # 执行选择
            target_count = plan['final_count']
            if len(cycle_data) <= target_count:
                # 数据量不够，全选
                selected = cycle_data
            else:
                # 随机采样
                import numpy as np
                np.random.seed(42)  # 固定种子确保可重现
                selected_indices = np.random.choice(cycle_data.index, size=target_count, replace=False)
                selected = cycle_data.loc[selected_indices]

            # 转换为测试用例
            for _, row in selected.iterrows():
                case = self._create_realistic_test_case(row, row.name, f"两阶段筛选:{cycle_desc}")
                selected_cases.append(case)

            self.selection_stats[cycle_desc] = {
                'original_count': plan['original_count'],
                'proportion': plan['proportion'],
                'guaranteed': plan['guaranteed'],
                'final_count': plan['final_count'],
                'actual': len(selected),
                'strategy': plan['strategy']
            }

            print(f"{cycle_desc:35} | 计划:{target_count:3}条 | 实选:{len(selected):3}条")

        # 4. 轻量去重（只去除完全相同的记录，保留更多样本）
        selected_cases = self._light_deduplication(selected_cases)

        print(f"\n=== 两阶段筛选结果 ===")
        print(f"最终选择: {len(selected_cases)} 条测试用例")

        return selected_cases

    def _get_data_by_cycle_description(self, cycle_desc):
        """根据周期描述获取对应的数据"""
        matched_indices = []

        for idx, row in self.df.iterrows():
            cycle_parsed = row['周期标签-解析']
            generated_desc = self._generate_cycle_description(cycle_parsed)
            if generated_desc == cycle_desc:
                matched_indices.append(idx)

        return self.df.loc[matched_indices]

    def _quality_selection_within_cycle(self, cycle_data, target_count):
        """在周期标签内进行质量筛选"""
        if len(cycle_data) <= target_count:
            return cycle_data

        # 多维度质量评分筛选
        cycle_data_copy = cycle_data.copy()

        # 1. 优先选择问句
        questions = cycle_data_copy[cycle_data_copy['是否是问句'] == '是']
        non_questions = cycle_data_copy[cycle_data_copy['是否是问句'] != '是']

        selected = pd.DataFrame()
        remaining_count = target_count

        # 2. 优先分配给问句（最多占30%）
        question_quota = min(len(questions), max(1, int(target_count * 0.3)))
        if question_quota > 0 and len(questions) > 0:
            question_selected = self._diverse_sample_by_multiple_dims(questions, question_quota)
            selected = pd.concat([selected, question_selected], ignore_index=True)
            remaining_count -= len(question_selected)

        # 3. 从非问句中选择剩余数量
        if remaining_count > 0 and len(non_questions) > 0:
            non_question_selected = self._diverse_sample_by_multiple_dims(non_questions, remaining_count)
            selected = pd.concat([selected, non_question_selected], ignore_index=True)

        return selected[:target_count]


    def _light_deduplication(self, cases):
        """轻量去重处理，只去除完全相同的记录"""
        # 构建完全相同记录的标识符（使用字符串拼接避免类型问题）
        seen_records = set()
        deduplicated = []

        for case in cases:
            # 使用字符串拼接作为唯一标识，减少过度去重
            context = str(case.get('最终传参上下文', ''))
            record_key = f"{case['最新客户消息'].strip()}|{context[:100]}|{case.get('销售ID', '')}|{case.get('客户ID', '')}"

            if record_key not in seen_records:
                seen_records.add(record_key)
                deduplicated.append(case)

        removed_count = len(cases) - len(deduplicated)
        print(f"轻量去重: 移除 {removed_count} 条完全重复记录，保留 {len(deduplicated)} 条")
        return deduplicated

    def _final_deduplication(self, cases):
        """最终去重处理"""
        seen_messages = set()
        deduplicated = []

        # 按优先级排序，保留高优先级的重复消息
        cases_sorted = sorted(cases, key=lambda x: x['test_priority'], reverse=True)

        for case in cases_sorted:
            message = case['最新客户消息'].strip()
            if message not in seen_messages:
                seen_messages.add(message)
                deduplicated.append(case)

        print(f"去重前: {len(cases)} 条，去重后: {len(deduplicated)} 条")
        return deduplicated

    def _select_by_cycle_coverage(self, count, used_indices):
        """按周期标签覆盖选择"""
        cases = []

        # 统计不同周期类型
        cycle_types = {}
        for idx, row in self.df.iterrows():
            if idx in used_indices:
                continue
            cycle_label = row['周期标签-解析']
            for cycle_type in cycle_label.keys():
                if cycle_type not in cycle_types:
                    cycle_types[cycle_type] = []
                cycle_types[cycle_type].append(idx)

        # 为每种周期类型分配样本
        per_cycle = max(1, count // len(cycle_types))

        for cycle_type, indices in cycle_types.items():
            available_indices = [idx for idx in indices if idx not in used_indices]
            selected_count = min(per_cycle, len(available_indices))

            if selected_count > 0:
                # 在每个周期类型内部进行多样化采样
                selected_indices = self._diverse_sampling(available_indices, selected_count)

                for idx in selected_indices:
                    row = self.df.loc[idx]
                    cases.append(self._create_realistic_test_case(row, idx, f'周期:{cycle_type}'))
                    used_indices.add(idx)

        return cases[:count]

    def _select_by_sales_id_coverage(self, count, used_indices):
        """按销售ID覆盖选择"""
        cases = []

        # 获取所有销售ID
        sales_ids = self.df['销售ID'].unique()
        per_sales = max(1, count // len(sales_ids))

        for sales_id in sales_ids:
            sales_data = self.df[
                (self.df['销售ID'] == sales_id) &
                (~self.df.index.isin(used_indices))
            ]

            if len(sales_data) > 0:
                selected_count = min(per_sales, len(sales_data))
                # 多样化采样：不同消息类型和周期
                selected = self._diverse_sample_by_multiple_dims(sales_data, selected_count)

                for _, row in selected.iterrows():
                    cases.append(self._create_realistic_test_case(row, row.name, f'销售ID:{sales_id[:10]}...'))
                    used_indices.add(row.name)

        return cases[:count]

    def _select_by_message_patterns(self, count, used_indices):
        """按消息模式选择"""
        cases = []

        pattern_types = self.df['消息类型'].unique()
        per_pattern = max(1, count // len(pattern_types))

        for pattern in pattern_types:
            pattern_data = self.df[
                (self.df['消息类型'] == pattern) &
                (~self.df.index.isin(used_indices))
            ]

            if len(pattern_data) > 0:
                selected_count = min(per_pattern, len(pattern_data))
                selected = self._diverse_sample_by_cycle(pattern_data, selected_count)

                for _, row in selected.iterrows():
                    cases.append(self._create_realistic_test_case(row, row.name, f'模式:{pattern}'))
                    used_indices.add(row.name)

        return cases[:count]

    def _select_question_scenarios(self, count, used_indices):
        """选择问句场景"""
        cases = []

        # 所有问句
        questions = self.df[
            (self.df['是否是问句'] == '是') &
            (~self.df.index.isin(used_indices))
        ]

        if len(questions) > 0:
            selected = self._diverse_sample_by_cycle(questions, min(count, len(questions)))

            for _, row in selected.iterrows():
                cases.append(self._create_realistic_test_case(row, row.name, '问句场景'))
                used_indices.add(row.name)

        return cases

    def _select_diverse_additional(self, count, used_indices):
        """多样化补充选择"""
        cases = []

        # 剩余可用数据
        available = self.df[~self.df.index.isin(used_indices)]

        if len(available) > 0:
            # 综合多维度多样化采样
            selected = self._comprehensive_diverse_sampling(available, min(count, len(available)))

            for _, row in selected.iterrows():
                cases.append(self._create_realistic_test_case(row, row.name, '多样化补充'))
                used_indices.add(row.name)

        return cases

    def _diverse_sampling(self, indices, count):
        """多样化采样"""
        if len(indices) <= count:
            return indices

        # 随机采样确保多样性
        np.random.seed(42)  # 固定种子确保可重现
        return np.random.choice(indices, count, replace=False).tolist()

    def _diverse_sample_by_multiple_dims(self, data, count):
        """基于多维度的多样化采样（去重）"""
        if len(data) <= count:
            return data

        # 先去重相同的客户消息
        data_dedup = self._deduplicate_messages(data)

        # 尝试在消息类型和周期上分层采样
        try:
            groups = data_dedup.groupby(['消息类型'])
            samples = []
            per_group = max(1, count // len(groups))

            for name, group in groups:
                sample_size = min(per_group, len(group))
                if sample_size > 0:
                    # 在每个组内进一步多样化
                    sample = self._smart_sample_within_group(group, sample_size)
                    samples.append(sample)

            result = pd.concat(samples) if samples else data_dedup.sample(min(count, len(data_dedup)), random_state=42)
            return result.head(count)
        except:
            return data_dedup.sample(min(count, len(data_dedup)), random_state=42)

    def _deduplicate_messages(self, data):
        """去重相同的客户消息"""
        # 按客户消息内容去重，保留第一个
        return data.drop_duplicates(subset=['最新客户消息'], keep='first')

    def _smart_sample_within_group(self, group, sample_size):
        """组内智能采样"""
        if len(group) <= sample_size:
            return group

        # 优先选择不同销售ID和不同周期的样本
        try:
            # 尝试分层采样
            stratified = []
            sales_ids = group['销售ID'].unique()
            per_sales = max(1, sample_size // len(sales_ids))

            for sales_id in sales_ids:
                sales_group = group[group['销售ID'] == sales_id]
                if len(sales_group) > 0:
                    sample_count = min(per_sales, len(sales_group))
                    sample = sales_group.sample(sample_count, random_state=42)
                    stratified.append(sample)

            if stratified:
                result = pd.concat(stratified)
                return result.head(sample_size)
            else:
                return group.sample(sample_size, random_state=42)
        except:
            return group.sample(sample_size, random_state=42)

    def _diverse_sample_by_cycle(self, data, count):
        """基于周期的多样化采样"""
        if len(data) <= count:
            return data

        try:
            # 基于周期标签分层
            cycle_groups = defaultdict(list)
            for idx, row in data.iterrows():
                cycle_key = str(row['周期标签-解析'])
                cycle_groups[cycle_key].append(idx)

            samples = []
            per_group = max(1, count // len(cycle_groups))

            for cycle_key, indices in cycle_groups.items():
                sample_size = min(per_group, len(indices))
                if sample_size > 0:
                    selected_indices = np.random.choice(indices, sample_size, replace=False)
                    samples.extend(selected_indices)

            return data.loc[samples[:count]]
        except:
            return data.sample(count, random_state=42)

    def _comprehensive_diverse_sampling(self, data, count):
        """综合多维度多样化采样"""
        if len(data) <= count:
            return data

        try:
            # 多维度分层：消息类型 + 销售ID + 周期标签
            stratified_samples = []

            # 按消息类型分组
            message_groups = data.groupby('消息类型')
            samples_per_type = max(1, count // len(message_groups))

            for msg_type, msg_group in message_groups:
                if len(msg_group) > 0:
                    # 在消息类型内按销售ID再分层
                    sales_groups = msg_group.groupby('销售ID')
                    samples_per_sales = max(1, samples_per_type // len(sales_groups))

                    for sales_id, sales_group in sales_groups:
                        sample_size = min(samples_per_sales, len(sales_group))
                        if sample_size > 0:
                            sample = sales_group.sample(sample_size, random_state=42)
                            stratified_samples.append(sample)

            if stratified_samples:
                result = pd.concat(stratified_samples)
                return result.head(count)
            else:
                return data.sample(count, random_state=42)

        except:
            return data.sample(count, random_state=42)

    def _create_realistic_test_case(self, row, index, selection_reason):
        """创建真实场景测试用例"""

        # 直接从数据中获取用户信息库和销售信息库字段（已在process_excel_by_sales.py中提取）
        rag_content = row.get('rag', '')

        # 用户信息库字段
        batch_id = row.get('批次ID', '')
        questionnaire_content = row.get('问卷内容', '')
        questionnaire_title = row.get('问卷标题', '')
        questionnaire_id = row.get('问卷ID', '')
        exam_paper_name = row.get('试卷名称', '')
        exam_paper_id = row.get('试卷ID', '')
        exam_score = row.get('分数', '')
        exam_days = row.get('天', '')
        exam_accuracy = row.get('正确率', '')

        # 销售信息库字段
        live_time = row.get('直播课时间', '')
        activate_link = row.get('激活直播课链接', '')
        study_file_link = row.get('学习档案链接', '')
        mobile_link = row.get('手机平板上课链接', '')
        pc_link = row.get('电脑上课链接', '')
        quick_tips_link = row.get('秒题技巧链接', '')
        enroll_link = row.get('报课链接', '')
        trial_link = row.get('试学链接', '')

        # 直接从数据中获取最后销售消息（已在process_excel_by_sales.py中生成）
        last_sales_messages = row.get('最后销售消息', [])

        # 预期销售回复字段（已在process_excel_by_sales.py中生成）
        expected_sales_reply = row.get('预期销售回复', '')
        alternative_sales_replies = row.get('备选销售回复', [])

        return {
            '销售ID': row['销售ID'],
            '客户ID': row['客户ID'],
            '最终传参上下文': row['历史对话'],
            '最新客户消息': row['最新客户消息'],
            '周期标签-解析': row['周期标签-解析'],
            '周期标签': row['周期标签'],
            '是否问答': row['是否是问句'],
            '发送时间': row.get('发送时间', ''),
            # 新增字段：thought_unit提取的信息
            '原始thought_unit': row.get('thought_unit', ''),
            '强遵循标签': row.get('强遵循标签', 'False'),
            'FAQ判断': row.get('FAQ判断', 'False'),
            '知识问答判断': row.get('知识问答判断', 'False'),
            '销售一级节点': row.get('销售一级节点', ''),
            '销售二级节点': row.get('销售二级节点', ''),
            'reference_script': row.get('reference_script', ''),
            '标签': row.get('标签', ''),
            '用户昵称': row.get('用户昵称', ''),
            # 用户信息库字段
            '批次ID': batch_id,
            '问卷内容': questionnaire_content,
            '问卷标题': questionnaire_title,
            '问卷ID': questionnaire_id,
            '试卷名称': exam_paper_name,
            '试卷ID': exam_paper_id,
            '分数': exam_score,
            '天': exam_days,
            '正确率': exam_accuracy,
            # 销售信息库字段
            '直播课时间': live_time,
            '激活直播课链接': activate_link,
            '学习档案链接': study_file_link,
            '手机平板上课链接': mobile_link,
            '电脑上课链接': pc_link,
            '秒题技巧链接': quick_tips_link,
            '报课链接': enroll_link,
            '试学链接': trial_link,
            # 最后销售消息字段
            '最后销售消息': last_sales_messages,
            # 预期回复字段
            '预期销售回复': expected_sales_reply,
            '备选销售回复': alternative_sales_replies,
            # CSM回复信息字段
            'CSM_AI生成消息': row.get('CSM_AI生成消息', ''),
            'CSM_发送消息内容': row.get('CSM_发送消息内容', ''),
            'CSM_AI与审核结果是否一致': row.get('CSM_AI与审核结果是否一致', ''),
            # 内部使用字段（用于分析但不在最终输出中显示）
            'cycle_parsed': row['周期标签-解析'],
            'test_priority': self._calculate_test_priority(row),
            'scenario_category': self._categorize_scenario(row),
            'selection_reason': selection_reason
        }

    def _predict_expected_response(self, row):
        """预测期望的回复类型"""
        message = str(row['最新客户消息']).lower()

        if row['是否是问句'] == '是':
            return '信息回答'
        elif any(word in message for word in ['好的', '可以', '是的']):
            return '后续引导'
        elif any(word in message for word in ['不', '没', '不要']):
            return '异议处理'
        elif any(word in message for word in ['谢谢', '感谢']):
            return '礼貌回应'
        else:
            return '常规跟进'

    def _calculate_test_priority(self, row):
        """计算测试优先级"""
        priority = 0

        # 问句优先级更高
        if row['是否是问句'] == '是':
            priority += 3

        # 服务期/行课期优先级
        cycle_label = row['周期标签-解析']
        if '行课期' in cycle_label:
            priority += 2
        if '服务期' in cycle_label:
            priority += 1

        # 对话复杂度
        if row['对话轮次'] > 5:
            priority += 1

        # 消息长度
        if row['消息长度'] > 50:
            priority += 1

        return min(priority, 5)  # 最高5级

    def _categorize_scenario(self, row):
        """场景分类"""
        cycle_label = row['周期标签-解析']
        message_type = row['消息类型']

        if '行课期' in cycle_label:
            return f'行课期-{message_type}'
        elif '服务期' in cycle_label:
            return f'服务期-{message_type}'
        else:
            return f'未知期-{message_type}'

    def export_realistic_test_cases(self, test_cases, output_file='chengla_test_cases-3.xlsx'):
        """导出真实场景测试用例"""
        print(f"\n正在导出测试用例到 {output_file}...")

        # 转换为DataFrame
        df_test = pd.DataFrame(test_cases)

        # 创建Excel文件
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 主测试用例表 - 只保留指定的列
            display_columns = [
                '销售ID', '客户ID', '最终传参上下文', '最新客户消息', '周期标签', '周期标签-解析', '是否问答',
                '发送时间', '原始thought_unit', '强遵循标签', 'FAQ判断', '知识问答判断', '销售一级节点', '销售二级节点', 'reference_script', '标签', '用户昵称',
                '批次ID', '问卷内容', '问卷标题', '问卷ID', '试卷名称', '试卷ID', '分数', '天', '正确率',
                '直播课时间', '激活直播课链接', '学习档案链接', '手机平板上课链接', '电脑上课链接',
                '秒题技巧链接', '报课链接', '试学链接', '最后销售消息', '预期销售回复', '备选销售回复',
                'CSM_AI生成消息', 'CSM_发送消息内容', 'CSM_AI与审核结果是否一致'
            ]
            df_display = df_test[display_columns]
            df_display.to_excel(writer, sheet_name='测试用例', index=False)

            # 周期标签选择策略分析 - 简化版本
            cycle_selection_analysis = self._generate_cycle_selection_analysis(test_cases)
            pd.DataFrame(cycle_selection_analysis).to_excel(writer, sheet_name='周期标签选择策略', index=False)

            # 原表数据 - 包含所有重要字段
            original_data_columns = [
                '销售ID', '客户ID', '批次ID', '最终传参上下文', '历史对话', '最新客户消息',
                '周期标签', '周期标签-解析', '是否问答', '是否是问句',
                'thought_unit', '强遵循标签', 'FAQ判断', '知识问答判断', '销售一级节点', '销售二级节点', 'reference_script', '标签', '用户昵称',
                '问卷内容', '问卷标题', '问卷ID', '试卷名称', '试卷ID', '分数', '天', '正确率',
                '直播课时间', '激活直播课链接', '学习档案链接', '手机平板上课链接', '电脑上课链接',
                '秒题技巧链接', '报课链接', '试学链接', '最后销售消息', '预期销售回复', '备选销售回复',
                'CSM_AI生成消息', 'CSM_发送消息内容', 'CSM_AI与审核结果是否一致'
            ]

            # 从原始数据中选择存在的列
            available_columns = [col for col in original_data_columns if col in self.df.columns]
            df_original = self.df[available_columns]
            df_original.to_excel(writer, sheet_name='原表', index=False)

        print(f"✅ 真实场景测试用例已导出: {len(test_cases)} 条")
        print(f"✅ 原表数据已导出: {len(self.df)} 条")
        return output_file

    def _generate_scenario_statistics(self, test_cases):
        """生成场景统计"""
        stats = []

        # 按详细周期分类统计
        detailed_cycle_counts = Counter()
        for case in test_cases:
            cycle_parsed = case['cycle_parsed']
            cycle_description = self._generate_cycle_description(cycle_parsed)
            detailed_cycle_counts[cycle_description] += 1

        for cycle_desc, count in detailed_cycle_counts.items():
            stats.append({
                '场景类型': f'详细周期-{cycle_desc}',
                '测试用例数': count,
                '占比': f"{count/len(test_cases)*100:.1f}%"
            })

        # 按场景分类统计
        scenario_counts = Counter([case['scenario_category'] for case in test_cases])
        for scenario, count in scenario_counts.items():
            stats.append({
                '场景类型': scenario,
                '测试用例数': count,
                '占比': f"{count/len(test_cases)*100:.1f}%"
            })

        # 按选择原因统计
        reason_counts = Counter([case['selection_reason'] for case in test_cases])
        for reason, count in reason_counts.items():
            stats.append({
                '场景类型': f'选择原因-{reason}',
                '测试用例数': count,
                '占比': f"{count/len(test_cases)*100:.1f}%"
            })

        return stats

    def _generate_priority_analysis(self, test_cases):
        """生成优先级分析"""
        priority_stats = []

        priority_counts = Counter([case['test_priority'] for case in test_cases])
        for priority, count in priority_counts.items():
            priority_stats.append({
                '优先级': f'P{priority}',
                '测试用例数': count,
                '占比': f"{count/len(test_cases)*100:.1f}%",
                '建议': self._get_priority_suggestion(priority)
            })

        return priority_stats

    def _get_priority_suggestion(self, priority):
        """获取优先级建议"""
        suggestions = {
            5: '最高优先级，必须测试',
            4: '高优先级，重点测试',
            3: '中等优先级，常规测试',
            2: '低优先级，选择性测试',
            1: '最低优先级，补充测试',
            0: '基础测试'
        }
        return suggestions.get(priority, '常规测试')

    def _generate_coverage_analysis(self, test_cases):
        """生成覆盖率分析"""
        coverage = {}

        # 详细周期标签覆盖（包含具体天数）
        detailed_cycle_coverage = []
        detailed_cycle_counts = Counter()

        for case in test_cases:
            cycle_parsed = case['cycle_parsed']
            cycle_description = self._generate_cycle_description(cycle_parsed)
            detailed_cycle_counts[cycle_description] += 1

        # 获取原始数据的详细周期分布
        original_detailed_cycles = self._analyze_detailed_cycle_labels()

        for cycle_desc, count in detailed_cycle_counts.items():
            original_count = original_detailed_cycles.get(cycle_desc, {}).get('count', 0)
            coverage_rate = (count / original_count * 100) if original_count > 0 else 0

            detailed_cycle_coverage.append({
                '详细周期类型': cycle_desc,
                '测试用例数': count,
                '原始数据量': original_count,
                '覆盖率': f"{coverage_rate:.1f}%"
            })

        # 按测试用例数量排序
        detailed_cycle_coverage.sort(key=lambda x: x['测试用例数'], reverse=True)
        coverage['详细周期覆盖'] = detailed_cycle_coverage

        # 传统周期标签覆盖（兼容性）
        cycle_coverage = []
        cycle_counts = Counter()
        for case in test_cases:
            cycle_parsed = case['cycle_parsed']
            for cycle_type in cycle_parsed.keys():
                cycle_counts[cycle_type] += 1

        for cycle_type, count in cycle_counts.items():
            cycle_coverage.append({
                '周期类型': cycle_type,
                '覆盖用例数': count,
                '原始数据量': self._get_original_cycle_count(cycle_type),
                '覆盖率': f"{count/self._get_original_cycle_count(cycle_type)*100:.1f}%"
            })

        coverage['基础周期覆盖'] = cycle_coverage

        # 销售ID覆盖
        sales_coverage = []
        sales_counts = Counter([case['销售ID'] for case in test_cases])
        original_sales_dist = self.df['销售ID'].value_counts()

        for sales_id, count in sales_counts.items():
            original_count = original_sales_dist.get(sales_id, 0)
            sales_coverage.append({
                '销售ID': sales_id,
                '覆盖用例数': count,
                '原始数据量': original_count,
                '覆盖率': f"{count/original_count*100:.1f}%" if original_count > 0 else "0%"
            })

        coverage['销售ID覆盖'] = sales_coverage

        # 用户信息库字段覆盖率统计
        user_info_coverage = []
        total_cases = len(test_cases)

        # 批次ID覆盖率
        batch_id_count = len([case for case in test_cases if case.get('批次ID', '') != ''])
        user_info_coverage.append({
            '字段名称': '批次ID',
            '有效用例数': batch_id_count,
            '总用例数': total_cases,
            '覆盖率': f"{batch_id_count/total_cases*100:.1f}%"
        })

        # 问卷内容覆盖率
        questionnaire_count = len([case for case in test_cases if case.get('问卷内容', '') != ''])
        user_info_coverage.append({
            '字段名称': '问卷内容',
            '有效用例数': questionnaire_count,
            '总用例数': total_cases,
            '覆盖率': f"{questionnaire_count/total_cases*100:.1f}%"
        })

        # 问卷标题覆盖率
        questionnaire_title_count = len([case for case in test_cases if case.get('问卷标题', '') != ''])
        user_info_coverage.append({
            '字段名称': '问卷标题',
            '有效用例数': questionnaire_title_count,
            '总用例数': total_cases,
            '覆盖率': f"{questionnaire_title_count/total_cases*100:.1f}%"
        })

        # 问卷ID覆盖率
        questionnaire_id_count = len([case for case in test_cases if case.get('问卷ID', '') != ''])
        user_info_coverage.append({
            '字段名称': '问卷ID',
            '有效用例数': questionnaire_id_count,
            '总用例数': total_cases,
            '覆盖率': f"{questionnaire_id_count/total_cases*100:.1f}%"
        })

        # 试卷信息覆盖率
        exam_paper_count = len([case for case in test_cases if case.get('试卷名称', '') != ''])
        user_info_coverage.append({
            '字段名称': '试卷信息',
            '有效用例数': exam_paper_count,
            '总用例数': total_cases,
            '覆盖率': f"{exam_paper_count/total_cases*100:.1f}%"
        })


        coverage['用户信息库覆盖'] = user_info_coverage

        # 销售信息库字段覆盖率统计
        sales_info_coverage = []

        # 直播课时间覆盖率
        live_time_count = len([case for case in test_cases if case.get('直播课时间', '') != ''])
        sales_info_coverage.append({
            '字段名称': '直播课时间',
            '有效用例数': live_time_count,
            '总用例数': total_cases,
            '覆盖率': f"{live_time_count/total_cases*100:.1f}%"
        })

        # 激活直播课链接覆盖率
        activate_link_count = len([case for case in test_cases if case.get('激活直播课链接', '') != ''])
        sales_info_coverage.append({
            '字段名称': '激活直播课链接',
            '有效用例数': activate_link_count,
            '总用例数': total_cases,
            '覆盖率': f"{activate_link_count/total_cases*100:.1f}%"
        })

        # 学习档案链接覆盖率
        study_file_link_count = len([case for case in test_cases if case.get('学习档案链接', '') != ''])
        sales_info_coverage.append({
            '字段名称': '学习档案链接',
            '有效用例数': study_file_link_count,
            '总用例数': total_cases,
            '覆盖率': f"{study_file_link_count/total_cases*100:.1f}%"
        })

        # 手机平板上课链接覆盖率
        mobile_link_count = len([case for case in test_cases if case.get('手机平板上课链接', '') != ''])
        sales_info_coverage.append({
            '字段名称': '手机平板上课链接',
            '有效用例数': mobile_link_count,
            '总用例数': total_cases,
            '覆盖率': f"{mobile_link_count/total_cases*100:.1f}%"
        })

        # 电脑上课链接覆盖率
        pc_link_count = len([case for case in test_cases if case.get('电脑上课链接', '') != ''])
        sales_info_coverage.append({
            '字段名称': '电脑上课链接',
            '有效用例数': pc_link_count,
            '总用例数': total_cases,
            '覆盖率': f"{pc_link_count/total_cases*100:.1f}%"
        })

        # 秒题技巧链接覆盖率
        quick_tips_link_count = len([case for case in test_cases if case.get('秒题技巧链接', '') != ''])
        sales_info_coverage.append({
            '字段名称': '秒题技巧链接',
            '有效用例数': quick_tips_link_count,
            '总用例数': total_cases,
            '覆盖率': f"{quick_tips_link_count/total_cases*100:.1f}%"
        })

        # 报课链接覆盖率
        enroll_link_count = len([case for case in test_cases if case.get('报课链接', '') != ''])
        sales_info_coverage.append({
            '字段名称': '报课链接',
            '有效用例数': enroll_link_count,
            '总用例数': total_cases,
            '覆盖率': f"{enroll_link_count/total_cases*100:.1f}%"
        })

        # 试学链接覆盖率
        trial_link_count = len([case for case in test_cases if case.get('试学链接', '') != ''])
        sales_info_coverage.append({
            '字段名称': '试学链接',
            '有效用例数': trial_link_count,
            '总用例数': total_cases,
            '覆盖率': f"{trial_link_count/total_cases*100:.1f}%"
        })

        coverage['销售信息库覆盖'] = sales_info_coverage

        # 最后销售消息覆盖率统计
        last_sales_coverage = []

        # 最后销售消息覆盖率
        last_sales_count = len([case for case in test_cases if case.get('最后销售消息', []) != []])
        last_sales_coverage.append({
            '字段名称': '最后销售消息',
            '有效用例数': last_sales_count,
            '总用例数': total_cases,
            '覆盖率': f"{last_sales_count/total_cases*100:.1f}%"
        })

        # 平均销售消息数量
        total_messages = sum(len(case.get('最后销售消息', [])) for case in test_cases)
        avg_messages = total_messages / total_cases if total_cases > 0 else 0
        last_sales_coverage.append({
            '字段名称': '平均销售消息数',
            '有效用例数': f"{avg_messages:.1f}",
            '总用例数': total_cases,
            '覆盖率': '-'
        })

        coverage['最后销售消息覆盖'] = last_sales_coverage

        # CSM回复信息覆盖率统计
        csm_info_coverage = []

        # CSM AI生成消息覆盖率
        csm_ai_count = len([case for case in test_cases if case.get('CSM_AI生成消息', '') != ''])
        csm_info_coverage.append({
            '字段名称': 'CSM_AI生成消息',
            '有效用例数': csm_ai_count,
            '总用例数': total_cases,
            '覆盖率': f"{csm_ai_count/total_cases*100:.1f}%"
        })

        # CSM发送消息内容覆盖率
        csm_content_count = len([case for case in test_cases if case.get('CSM_发送消息内容', '') != ''])
        csm_info_coverage.append({
            '字段名称': 'CSM_发送消息内容',
            '有效用例数': csm_content_count,
            '总用例数': total_cases,
            '覆盖率': f"{csm_content_count/total_cases*100:.1f}%"
        })

        # CSM AI与审核结果是否一致覆盖率
        csm_consistency_count = len([case for case in test_cases if case.get('CSM_AI与审核结果是否一致', '') != ''])
        csm_info_coverage.append({
            '字段名称': 'CSM_AI与审核结果是否一致',
            '有效用例数': csm_consistency_count,
            '总用例数': total_cases,
            '覆盖率': f"{csm_consistency_count/total_cases*100:.1f}%"
        })

        # 有任何CSM回复信息的用例统计
        any_csm_count = len([case for case in test_cases if any([
            case.get('CSM_AI生成消息', '') != '',
            case.get('CSM_发送消息内容', '') != '',
            case.get('CSM_AI与审核结果是否一致', '') != ''
        ])])
        csm_info_coverage.append({
            '字段名称': '任意CSM回复信息',
            '有效用例数': any_csm_count,
            '总用例数': total_cases,
            '覆盖率': f"{any_csm_count/total_cases*100:.1f}%"
        })

        coverage['CSM回复信息覆盖'] = csm_info_coverage

        return coverage

    def _generate_cycle_selection_analysis(self, test_cases):
        """生成两阶段筛选的周期标签选择策略分析"""
        analysis_results = []

        # 使用存储的选择统计信息
        if hasattr(self, 'selection_stats'):
            for cycle_desc, stats in self.selection_stats.items():
                analysis_results.append({
                    '周期标签': cycle_desc,
                    '原表数量': stats['original_count'],
                    '原表占比': f"{stats['proportion']*100:.1f}%",
                    '保底数量': stats.get('guaranteed', 0),
                    '最终分配': stats['final_count'],
                    '实际选择': stats['actual']
                })

        # 按原表数量排序
        analysis_results.sort(key=lambda x: x['原表数量'], reverse=True)

        return analysis_results

    def _evaluate_selection_effect(self, actual_count, recommended_count, original_count):
        """评估选择效果"""
        if original_count <= 10:
            if actual_count == original_count:
                return "✓ 完美（全部选择）"
            elif actual_count > original_count * 0.8:
                return "✓ 良好（接近全选）"
            else:
                return "⚠ 不足（应全选）"
        else:
            deviation = abs(actual_count - recommended_count) / recommended_count if recommended_count > 0 else 1
            if deviation <= 0.1:
                return "✓ 完美（偏差≤10%）"
            elif deviation <= 0.2:
                return "✓ 良好（偏差≤20%）"
            elif deviation <= 0.3:
                return "△ 一般（偏差≤30%）"
            else:
                return "✗ 偏差较大（>30%）"

    def _get_original_cycle_count(self, cycle_type):
        """获取原始数据中某周期类型的数量"""
        count = 0
        for cycle_label in self.df['周期标签-解析']:
            if cycle_type in cycle_label:
                count += 1
        return max(count, 1)

    def generate_realistic_test_report(self, test_cases):
        """生成真实场景测试报告"""
        report = f"""
# 真实销售场景测试用例选择报告

## 数据概览
- 原始数据总量: {len(self.df):,} 条
- 筛选测试用例: {len(test_cases)} 条
- 覆盖率: {len(test_cases)/len(self.df)*100:.2f}%
- 销售ID覆盖: {len(set([case['销售ID'] for case in test_cases]))} 个

## 选择策略

本次测试用例选择完全基于真实线上销售场景，确保测试的实用性和有效性。

### 核心维度

#### 1. 周期标签覆盖 (40%)
- **行课期场景**: 学员正在上课期间的对话
- **服务期场景**: 课程服务期间的跟进对话
- **混合周期**: 跨越多个周期的复杂场景

#### 2. 销售ID覆盖 (25%)
- 确保不同销售人员的对话风格都有覆盖
- 验证系统对不同销售场景的适应性

#### 3. 消息模式覆盖 (20%)
- **确认回复**: "好的"、"可以"、"收到"等
- **拒绝回复**: "不要"、"不需要"等
- **疑问询问**: 客户主动提问场景
- **详细描述**: 客户详细说明情况

#### 4. 问句场景覆盖 (10%)
- 专门覆盖客户提问场景
- 测试系统的问答处理能力

### 真实场景特色

#### 1. 基于真实对话流程
- 保留完整的历史对话上下文
- 包含真实的RAG检索数据
- 保持原有的销售思维链路

#### 2. 周期感知测试
- 不同学习周期的客户行为模式
- 周期转换时的关键对话节点

#### 3. 销售个性化
- 不同销售ID的话术风格
- 个性化的客户关系维护

## 测试场景分类

### 高优先级场景 (P4-P5)
1. **行课期问句**: 学员上课期间的疑问
2. **服务期异议**: 服务期间的客户异议
3. **复杂对话**: 多轮对话的深度场景

### 中等优先级场景 (P2-P3)
1. **标准确认**: 常见的确认回复
2. **设备问题**: 技术相关咨询
3. **感谢表达**: 客户感谢场景

### 基础场景 (P0-P1)
1. **系统消息**: 自动添加好友等
2. **简单回复**: 单字符回复
3. **其他场景**: 未分类的场景

## 测试建议

### 1. 分阶段测试
- **第一阶段**: 高优先级场景 (P4-P5)
- **第二阶段**: 中等优先级场景 (P2-P3)
- **第三阶段**: 基础场景 (P0-P1)

### 2. 关键验证点
- **上下文理解**: 基于历史对话的准确回复
- **周期适应**: 不同学习周期的差异化服务
- **个性化程度**: 保持销售个人风格

### 3. 成功标准
- **回复准确性**: 根据上下文给出合适回复
- **周期敏感性**: 识别并适应不同周期需求
- **一致性**: 与原销售风格保持一致

## 预期收益

通过这套真实场景测试体系，可以:

1. **验证真实效果**: 基于真实对话数据的测试更有说服力
2. **发现实际问题**: 识别系统在真实场景中的不足
3. **优化用户体验**: 提升客户对话的满意度
4. **提高转化效果**: 保持甚至提升销售转化率

这套测试用例完全还原了真实的线上销售场景，确保AI系统能够在实际应用中发挥最佳效果。
        """

        return report

def main():
    """主函数"""
    selector = RealisticSalesTestSelector()

    # 加载真实数据
    selector.load_data('按销售ID分组的测试集和周期标签分析_with_thought_unit_final1.xlsx')

    # 分析真实场景
    scenarios = selector.analyze_real_scenarios()

    # 选择真实测试用例 (使用简化筛选逻辑)
    test_cases = selector.select_realistic_test_cases()

    # 导出测试用例
    output_file = selector.export_realistic_test_cases(test_cases)

    # 生成报告
    report = selector.generate_realistic_test_report(test_cases)

    # 保存报告
    with open('realistic_test_case_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n=== 真实场景测试用例筛选完成 ===")
    print(f"输出文件: {output_file}")
    print(f"测试报告: realistic_test_case_report.md")
    print(f"测试用例数量: {len(test_cases)}")

    # 显示周期标签选择策略分析摘要
    print("\n=== 周期标签选择策略分析摘要 ===")
    cycle_selection_analysis = selector._generate_cycle_selection_analysis(test_cases)

    # 显示前10个主要周期的选择情况
    print("主要周期标签选择情况:")
    for i, analysis in enumerate(cycle_selection_analysis[:10]):
        print(f"  {analysis['周期标签']:35} | 原表:{analysis['原表数量']:4}条 | 占比:{analysis['原表占比']:6} | 保底:{analysis['保底数量']:2}条 | 最终:{analysis['最终分配']:2}条 | 实选:{analysis['实际选择']:2}条")

    # 显示详细周期分布
    print("\n=== 测试用例详细周期分布 ===")
    detailed_cycle_dist = Counter()
    for case in test_cases:
        cycle_parsed = case['cycle_parsed']
        cycle_description = selector._generate_cycle_description(cycle_parsed)
        detailed_cycle_dist[cycle_description] += 1

    for cycle_desc, count in detailed_cycle_dist.most_common(10):
        percentage = (count / len(test_cases)) * 100
        print(f"{cycle_desc}: {count} 条 ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
