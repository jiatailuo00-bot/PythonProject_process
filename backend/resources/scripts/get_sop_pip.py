import traceback
import jieba
import pandas as pd
import os
import json
import re

from typing import Dict
from tqdm import tqdm


def preprocess_sentence(sentence):
    """文本预处理：全角转半角、小写转换、清理特殊字符和空格"""
    if not isinstance(sentence, str) or sentence.strip() == "":
        return ""

    sentence = sentence.lower()  # 小写转换
    sentence = re.sub(r"\s+", " ", sentence).strip() # 移除多余空格
    return sentence

def calculate_sentence_similarity(sentence1, sentence2):
    """计算两句话的余弦相似度（支持中文，仅返回相似度值0~1）"""
    # 1. 预处理句子
    processed1 = preprocess_sentence(sentence1)
    processed2 = preprocess_sentence(sentence2)



    # 2. 处理空句子（任一为空则相似度0）
    if not processed1 or not processed2:
        return 0.0

    # 3. 中文分词（关键修复：用jieba拆分中文，替代原有的split()）
    words1 = list(jieba.cut(processed1))
    words2 = list(jieba.cut(processed2))



    # 4. 构建词汇表（两句话的所有唯一词）
    vocab = list(set(words1 + words2))

    # 5. 统计词频（将词转换为向量）
    def count_words(words, vocab):
        return [words.count(word) for word in vocab]
    vec1 = count_words(words1, vocab)
    vec2 = count_words(words2, vocab)



    # 6. 计算余弦相似度（纯Python实现）
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    norm1 = (sum(v ** 2 for v in vec1)) ** 0.5
    norm2 = (sum(v ** 2 for v in vec2)) ** 0.5



    # 避免除以0（防止两句话都无有效词）
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # 保留4位小数返回相似度
    similarity = round(dot_product / (norm1 * norm2), 4)
    return similarity


def extract_dialogs(file_path, start_id, end_id):
    """
    从Excel文件中提取指定ID范围内的对话内容并格式化

    参数:
        file_path (str): Excel文件路径
        start_id (int): 起始ID
        end_id (int): 结束ID

    返回:
        str: 格式化后的对话字符串
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 检查必要的列是否存在
        required_columns = ['ID', '发送消息角色', '处理后对话内容', '对话时间']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Excel文件中缺少必要的列: {col}")

        # 筛选指定ID范围内的数据
        mask = (df['ID'] >= start_id) & (df['ID'] <= end_id)
        filtered_df = df[mask]
        # 按ID排序
        filtered_df = filtered_df.sort_values('ID')
        # 如果没有找到数据
        if filtered_df.empty:
            return f"在ID范围 [{start_id}, {end_id}] 内没有找到任何数据"

        # 拼接对话内容
        dialog_str = ""
        for _, row in filtered_df.iterrows():
            line = f"[{row['发送消息角色']}][{row['对话时间']}]：{row['处理后对话内容']}\n"
            dialog_str += line
        return dialog_str.strip()  # 去除最后一个换行符

    except FileNotFoundError:
        return f"错误：文件 '{file_path}' 不存在"
    except Exception as e:
        return f"处理过程中发生错误：{str(e)}"


def find_stop_row_with_single_customer_reply(file_path, start_row, end_row):
    """
    从Excel文件的起止行之间，找到第一句客户的对话行，并返回该行号和对话内容
    若未找到客户对话，返回(-1, "")

    参数：
        file_path: Excel文件路径
        start_row: 开始遍历的行号（从0开始，与pandasDataFrame行号一致）
        end_row: 结束遍历的行号（不包含end_row，即遍历范围为[start_row, end_row)）

    返回：
        tuple: (找到的客户对话行号, 客户对话内容)；未找到时返回(-1, "")
    """
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return -1, ""
    try:
        df = pd.read_excel(file_path, usecols=['发送消息角色', '处理后对话内容', '对话时间'])
        current_row = start_row
        client_messages = []
        while current_row < end_row:
            role = str(df.iloc[current_row]['发送消息角色']).strip().lower()
            msg_time = str(df.iloc[current_row]['对话时间']).strip()
            # 情况1：如果是销售，继续遍历
            if role == '销售':
                client_messages = []
                current_row += 1
            # 情况2：如果是客户，继续遍历直到遇到销售
            elif role == '客户':
                client_msg = str(f"[{role}][{msg_time}]：{df.iloc[current_row]['处理后对话内容']}").strip()
                current_row += 1
                client_messages.append(client_msg)
                return current_row, '\n'.join(client_messages) if client_messages else ""
            # 情况3：遇到其他角色，视为无效数据
            else:
                print(f"警告：第{current_row}行发现未知角色 '{role}'，停止遍历")
                return -1, ""
        # 如果遍历到文件末尾都没有符合条件的行
        return -1, '\n'.join(client_messages) if client_messages else ""
    except KeyError:
        print("错误：Excel文件中缺少'发送消息角色'或'处理后对话内容'列")
        return -1, ""
    except Exception as e:
        print(traceback.format_exc())
        print(f"处理文件时发生错误：{str(e)}")
        return -1, ""


def find_stop_row_with_mult_customer_reply(file_path, start_row, end_row):
    """
    从Excel文件中找到第一个销售和客户之间的对话行，并返回该行号和对话内容
    """
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return -1, ""
    try:
        df = pd.read_excel(file_path, usecols=['发送消息角色', '处理后对话内容', '对话时间'])
        current_row = start_row
        client_messages = []
        while current_row < end_row:
            role = str(df.iloc[current_row]['发送消息角色']).strip().lower()
            msg_time = str(df.iloc[current_row]['对话时间']).strip()
            # 情况1：如果是销售，继续遍历
            if role == '销售':
                client_messages = []
                current_row += 1
            # 情况2：如果是客户，继续遍历直到遇到销售
            elif role == '客户':
                client_msg = str(f"[{role}][{msg_time}]：{df.iloc[current_row]['处理后对话内容']}").strip()
                client_messages.append(client_msg)
                current_row += 1
                while current_row < end_row:
                    next_role = str(df.iloc[current_row]['发送消息角色']).strip().lower()
                    if next_role == '销售':
                        return current_row, '\n'.join(client_messages) if client_messages else ""
                    elif next_role == '客户':
                        next_msg = str(f"[{role}][{msg_time}]：{df.iloc[current_row]['处理后对话内容']}").strip()
                        client_messages.append(next_msg)
                        current_row += 1
                    else:
                        print(f"警告：第{current_row}行发现未知角色 '{next_role}'，停止遍历")
                        return -1, '\n'.join(client_messages) if client_messages else ""
                return -1, '\n'.join(client_messages) if client_messages else ""
            # 情况3：遇到其他角色，视为无效数据
            else:
                print(f"警告：第{current_row}行发现未知角色 '{role}'，停止遍历")
                return -1, ""
        # 如果遍历到文件末尾都没有符合条件的行
        return -1, '\n'.join(client_messages) if client_messages else ""
    except KeyError:
        print("错误：Excel文件中缺少'发送消息角色'或'处理后对话内容'列")
        return -1, ""
    except Exception as e:
        print(traceback.format_exc())
        print(f"处理文件时发生错误：{str(e)}")
        return -1, ""


def list_of_dicts_to_xlsx(data_list, output_file):
    """
    将列表[{}, {}, ...]`结构的数据转换为xlsx文件

    参数:
        data_list: 包含字典的列表，每个字典代表一行数据，键为列名
        output_file: 输出的xlsx文件路径
    """
    # 检查输入数据是否为空
    if not data_list:
        print("错误：输入数据为空列表")
        return False

    # 检查数据是否为正确的结构
    if not all(isinstance(item, dict) for item in data_list):
        print("错误：输入列表中包含非字典元素")
        return False

    try:
        # 创建DataFrame
        df = pd.DataFrame(data_list)

        # 获取输出目录并确保目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 保存为xlsx文件，不保留索引
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"成功将数据保存到 {output_file}")
        print(f"包含 {len(df)} 行数据，{len(df.columns)} 列数据")
        return True

    except Exception as e:
        print(f"转换失败：{str(e)}")
        return False


def context_to_testcase(id_list, first_node_list, second_node_list, excel_path, mode):
    """
    根据提供的ID列表和Excel文件路径，处理销售对话数据并生成案例列表

    该函数主要功能是：
    1. 读取Excel文件中的对话数据
    2. 根据ID列表将对话内容分割成多个案例
    3. 为每个案例关联对应的一级和二级SOP节点
    4. 提取每个案例的上下文对话内容

    参数:
    id_list (list): 包含对话分割点ID的列表，这些ID作为每个案例的起始点
    first_node_list (list): 与id_list对应的一级SOP节点列表
    second_node_list (list): 与id_list对应的二级SOP节点列表
    excel_path (str): Excel文件的路径

    返回:
    list: 包含处理后案例字典的列表，每个字典包含一个完整案例的相关信息
    """
    try:
        df = pd.read_excel(excel_path)
        # 检查必要列是否存在
        required_cols = ['ID', '发送消息角色', '处理后对话内容', '对话时间']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"错误：Excel文件缺少必要列：{', '.join(missing_cols)}")
            return
        # 读取Excel文件，获取"发送消息角色"列
        df = pd.read_excel(excel_path, usecols=['ID'])
        total_rows = len(df)
        id_list.append(total_rows - 1)

        # 将ID列转换为字符串（避免数字ID匹配问题）
        df['ID'] = df['ID'].astype(str)
    except Exception:
        return

    # 一级循环：遍历数组1中的每个ID作为起点
    case_list = []
    for i, start_id in enumerate(id_list):
        # 最后一个sop节点以后的对话先不关注，直接退出
        if i == len(id_list) - 1:
            continue

        start_index = start_id
        end_index = id_list[i + 1]
        case_cut_times = 1
        for j in range(start_index, end_index):
            if j < start_index:
                continue
            # 获取分割点和分割点开始的最新客户消息
            if mode == "single":
                stop_line, customer_reply = find_stop_row_with_single_customer_reply(excel_path, start_index, id_list[i + 1])
            elif mode == "mult":
                stop_line, customer_reply = find_stop_row_with_mult_customer_reply(excel_path, start_index, id_list[i + 1])

            if stop_line != -1:
                start_index = stop_line
                first_sop_node = first_node_list[i]
                second_sop_node = second_node_list[i]
                # 获取从分割点的下一行开始的所有下文
                subsequent = extract_dialogs(excel_path, start_index + 1, total_rows)
                case_dict = concat_dialogues_by_id(
                    excel_path,
                    start_index,
                    first_sop_node,
                    second_sop_node,
                    customer_reply,
                    subsequent,
                    case_cut_times
                )
                case_cut_times += 1
                case_list.append(case_dict)
            else:
                continue
    return case_list


def concat_dialogues_by_id(file_path, target_id, first_sop, second_sop, customer_reply, subsequent, case_cut_times):
    """
    读取xlsx文件中ID从1到指定ID的所有行，按格式拼接成字符串并保存到字典中
    参数:
        file_path: xlsx文件路径
        target_id: 目标ID编号（整数）
        first_sop: SOP一级节点
        second_sop: SOP二级节点
        customer_reply: 客户回复
        subsequent: 下文
        case_cut_times: 案例距离SOP节点切分的次数，例如：SOP下第一句客户的对话，那么切分次数为1，SOP下第二句客户的对话，切分次数为2
    返回:
        包含对话信息的字典，如果出错则返回空字典
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return {}

    # 检查文件格式
    if not file_path.endswith('.xlsx'):
        print(f"错误：{file_path} 不是xlsx格式文件")
        return {}

    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 检查必要的列是否存在
        required_columns = ['ID', '发送消息角色', '处理后对话内容', '对话时间']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"错误：文件缺少必要列：{', '.join(missing_columns)}")
            return {}

        # 确保ID列是整数类型（处理可能的浮点型ID）
        df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
        if df['ID'].isna().any():
            print("警告：ID列中存在非数值类型，将被忽略")

        # 筛选ID从1到target_id的行
        mask = (df['ID'] >= 1) & (df['ID'] <= target_id)
        filtered_df = df[mask].sort_values(by='ID')  # 按ID排序

        if filtered_df.empty:
            print(f"警告：没有找到ID在1到{target_id}范围内的数据")
            return {}

        # 不带图片标签的上文
        result_lines_without_img_tag = []
        # 携带图片标签的上文
        result_lines_with_img_tag = []
        for _, row in filtered_df.iterrows():
            # 处理第一条规则：无图片标志行
            line_without_img_tag = f"[{row['发送消息角色']}][{row['对话时间']}]: {row['处理后对话内容']}"
            result_lines_without_img_tag.append(line_without_img_tag)
            # 处理第二条规则：带图片标志行
            if row['消息类型'] == "image":
                line_with_img_tag = f"[{row['发送消息角色']}][{row['对话时间']}]: {row['文件地址']}"
            else:
                line_with_img_tag = f"[{row['发送消息角色']}][{row['对话时间']}]: {row['处理后对话内容']}"
            result_lines_with_img_tag.append(line_with_img_tag)

        # 拼接后的对话字符串
        dialogue_str_without_img_tag = '\n'.join(result_lines_without_img_tag)
        dialogue_str_with_img_tag = '\n'.join(result_lines_with_img_tag)

        # 获取文件名（去掉路径和后缀）
        file_name = os.path.basename(file_path)
        # file_id代表文件名字，文件名含义如下：{客户ID}-{新客户语料}-{客户对话14句}-{销售对话27句}-{销售ID}-{租户ID}
        # 96d0fc6e671dd4121743031e8b2fcf82-新-C14-S27-wmsVYfBwAAhS7OwE1OPIq2LuLBUbex6A-fivedoctors
        file_id = os.path.splitext(file_name)[0]
        # 销售ID
        sales_id = file_id.split("-")[4]
        # 客户ID
        customer_id = file_id.split("-")[0]

        # 创建结果字典
        result_dict = {
            "id": file_id,
            "销售ID": sales_id,
            "客户ID": customer_id,
            "SOP一级节点": first_sop,
            "SOP二级节点": second_sop,
            "上文-不带img路径": dialogue_str_without_img_tag,
            "上文-带img路径": dialogue_str_with_img_tag,
            "最新客户消息": customer_reply,
            "切分次数": case_cut_times,
            "下文": subsequent
        }
        return result_dict

    except Exception as e:
        print(traceback.format_exc())
        print(f"处理文件时发生错误：{str(e)}")
        return {}


def read_json_file(file_path: str) -> Dict:
    """读取JSON文件并返回数据字典"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式")
        return {}
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return {}


def read_xlsx_files(path, sop_logic_tree_path, mode, use_similar_match, similarity=0.95):
    """
    遍历指定路径下的所有xlsx文件，读取并处理指定列的值，仅显示一个总体进度条
    参数:
        path: 要遍历的文件夹路径
    """
    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"错误：路径 {path} 不存在")
        return []

    # 获取所有xlsx文件列表
    xlsx_files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
    total_files = len(xlsx_files)

    if total_files == 0:
        print("未找到任何XLSX文件")
        return []

    print(f"发现 {total_files} 个XLSX文件，开始处理...\n")
    all_xlsx_file_case = []

    # 读取JSON数据（只读取一次）
    json_data = read_json_file(sop_logic_tree_path)
    if not json_data:
        print("无法读取JSON文件，终止处理")
        return []

    first_level_nodes = list(json_data.keys())

    # 创建一个总体进度条
    with tqdm(total=total_files, desc="处理进度", unit="文件") as pbar:
        for filename in xlsx_files:
            # 更新进度条描述，显示当前处理的文件名
            pbar.set_postfix(file=os.path.splitext(filename)[0][:150] + "...")  # 显示文件名前20个字符
            file_path = os.path.join(path, filename)
            try:
                # 读取Excel文件
                df = pd.read_excel(file_path)
                # 检查所需列是否存在
                required_columns = ["处理后对话内容", "ID", "对话时间"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"\n警告：文件 {filename} 缺少列 {', '.join(missing_columns)}，跳过此文件")
                    pbar.update(1)
                    continue

                xlsx_first_sop_list = []
                xlsx_second_sop_list = []
                xlsx_id_list = []

                # 遍历文件中的所有行
                for _, row in df.iterrows():
                    sales_context = row['处理后对话内容']
                    # 如果是客户的对话，则跳过
                    if row["发送消息角色"] == "客户":
                        continue
                    row_id = row['ID']
                    # 检查是否存在对应的sop话术
                    found = False
                    for first_node in first_level_nodes:
                        second_level_nodes = json_data.get(first_node, {})
                        for second_node, content in second_level_nodes.items():
                            similar_list = content.get("相似话术", [])
                            if use_similar_match:
                                for similar_sentences in similar_list:
                                    if calculate_sentence_similarity(sales_context, similar_sentences) >= similarity:
                                        xlsx_first_sop_list.append(first_node)
                                        xlsx_second_sop_list.append(second_node)
                                        xlsx_id_list.append(row_id)
                                        found = True
                                        # 如果当前的销售对话已经命中了某个SOP节点，则无需继续遍历剩余的节点，开始下一句对话的SOP节点匹配
                                        break
                            else:
                                if sales_context in similar_list:
                                    xlsx_first_sop_list.append(first_node)
                                    xlsx_second_sop_list.append(second_node)
                                    xlsx_id_list.append(row_id)
                                    found = True
                                    # 如果当前的销售对话已经命中了某个SOP节点，则无需继续遍历剩余的节点，开始下一句对话的SOP节点匹配
                                    break
                        # 如果当前的销售对话已经命中了某个SOP节点，则无需继续遍历剩余的节点，开始下一句对话的SOP节点匹配
                        if found:
                            break
                # 根据每个SOP的ID（对应在聊天语料中的位置）、SOP一级节点、SOP二级节点等信息，对案例进行切分
                single_xlsx_case_list = context_to_testcase(
                    xlsx_id_list,
                    xlsx_first_sop_list,
                    xlsx_second_sop_list,
                    file_path,
                    mode
                )
                all_xlsx_file_case.extend(single_xlsx_case_list)
            except Exception as e:
                print(f"\n处理文件 {filename} 时出错: {str(e)}")
            finally:
                # 更新进度条
                pbar.update(1)
    print(f"\n所有文件处理完成，共找到 {len(all_xlsx_file_case)} 个匹配项")
    return all_xlsx_file_case


def parse_conversation_history(history_str):
    """
    解析历史对话字符串，提取对话列表
    支持两种格式：
    1. JSON字符串: [{'role': '', 'content': '', 'time': ''}, ...]
    2. 多行字符串: 角色: 消息内容 或 [角色][时间]: 消息内容

    参数:
        history_str: 历史对话字符串

    返回:
        list: 对话列表，每个元素包含 {'role': '', 'content': '', 'time': ''}
    """
    if not history_str or pd.isna(history_str):
        return []

    conversations = []
    try:
        # 尝试解析为JSON格式
        json_data = json.loads(history_str)
        if isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, dict) and all(k in item for k in ['role', 'content']):
                    role = item['role'].strip()
                    content = item['content'].strip()
                    time = item.get('time', '').strip()
                    if role in ['销售', '客户']:
                        conversations.append({'role': role, 'content': content, 'time': time})
            return conversations
    except json.JSONDecodeError:
        # 如果不是JSON格式，则继续按行解析
        pass

    # 按行分割对话
    lines = str(history_str).strip().split('\n')

    current_conversation = None  # 当前正在构建的对话

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 尝试解析带时间戳的格式：[角色][时间]: 消息内容
        timestamp_pattern = r'\[(.+?)\]\[(.+?)\]:\s*(.+)'
        match = re.match(timestamp_pattern, line)

        if match:
            # 保存之前的对话（如果有）
            if current_conversation:
                conversations.append(current_conversation)

            role = match.group(1).strip()
            time = match.group(2).strip()
            content = match.group(3).strip()

            # 只保留销售和客户的对话
            if role in ['销售', '客户']:
                current_conversation = {
                    'role': role,
                    'content': content,
                    'time': time
                }
            else:
                current_conversation = None
        else:
            # 尝试解析简单格式：角色: 消息内容
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    # 保存之前的对话（如果有）
                    if current_conversation:
                        conversations.append(current_conversation)

                    role = parts[0].strip()
                    content = parts[1].strip()

                    # 只保留销售和客户的对话
                    if role in ['销售', '客户']:
                        current_conversation = {
                            'role': role,
                            'content': content,
                            'time': ''  # 简单格式中没有时间信息
                        }
                    else:
                        current_conversation = None
            else:
                # 这是一个续行，属于当前对话的一部分
                if current_conversation:
                    current_conversation['content'] += '\n' + line

    # 保存最后一个对话（如果有）
    if current_conversation:
        conversations.append(current_conversation)

    return conversations


def find_nearest_sales_sop_match(conversations, sop_data, similarity_threshold=0.95):
    """
    从后往前遍历对话，找到离客户最近的匹配SOP节点的销售消息

    参数:
        conversations: 对话列表（已按时间顺序排列）
        sop_data: SOP逻辑树数据
        similarity_threshold: 相似度阈值

    返回:
        dict: 匹配结果，包含SOP节点信息，如果没找到则返回None
    """
    # 从后往前遍历对话
    for conv in reversed(conversations):
        # 只处理销售消息
        if conv['role'] == '销售':
            sales_message = conv['content']

            # 寻找匹配的SOP节点
            match_result = find_best_sop_match(sales_message, sop_data, similarity_threshold)
            if match_result:
                # 找到匹配就立即返回
                match_result['sales_message'] = sales_message
                match_result['sales_time'] = conv['time']
                return match_result

    # 没找到任何匹配
    return None


def analyze_conversation_sop_labels(sales_corpus_xlsx, sop_logic_tree_path, similarity_threshold=0.95, batch_size=100):
    """
    分析销售对话汇总文件，为每条记录打上SOP节点标签

    新的处理逻辑：
    1. 读取销售对话汇总文件（包含历史对话字段）
    2. 对于每行数据，解析历史对话内容
    3. 从后往前遍历历史对话，找销售消息
    4. 对每个销售消息检查是否匹配SOP节点
    5. 一旦找到匹配就停止，返回该SOP节点
    6. 如果遍历完都没找到匹配，则为空
    7. 每处理完batch_size行就保存到临时Excel文件

    参数:
        sales_corpus_xlsx: 销售对话汇总Excel文件路径
        sop_logic_tree_path: SOP逻辑树JSON文件路径
        similarity_threshold: 相似度阈值
        batch_size: 批次大小，每处理多少行保存一次

    返回:
        包含SOP标签的记录列表
    """
    try:
        # 读取销售对话汇总文件
        df = pd.read_excel(sales_corpus_xlsx, engine='openpyxl')
        print(f"成功读取销售对话汇总文件，共 {len(df)} 条记录")

        # 检查必要的列
        if '最终传参上下文' not in df.columns:
            print("错误：缺少必要的对话列 '最终传参上下文'")
            return []

        print("使用对话列：最终传参上下文")

        # 读取SOP逻辑树
        sop_data = read_json_file(sop_logic_tree_path)
        if not sop_data:
            print("无法读取SOP逻辑树文件")
            return []

        print(f"成功读取SOP逻辑树，共 {len(sop_data)} 个一级节点")

        # 为每条记录分析并打标签
        labeled_records = []
        batch_count = 0

        # 生成临时文件路径
        temp_dir = os.path.dirname(sales_corpus_xlsx)
        temp_file_base = os.path.splitext(os.path.basename(sales_corpus_xlsx))[0]
        temp_file_path = os.path.join(temp_dir, f"{temp_file_base}_temp_progress.xlsx")

        print(f"每处理 {batch_size} 行将保存进度到：{temp_file_path}")

        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="分析对话记录")):
            # 解析历史对话
            history_str = str(row.get('最终传参上下文', '')).strip()
            if not history_str or history_str == 'nan':
                continue

            conversations = parse_conversation_history(history_str)
            if not conversations:
                continue

            # 检查最后一条消息是否为客户消息
            if conversations and conversations[-1]['role'] != '客户':
                print(f"警告：第{row.name}行的最后一条消息不是客户消息，跳过")
                continue

            # 从后往前找最近的匹配SOP节点的销售消息
            sop_match = find_nearest_sales_sop_match(conversations, sop_data, similarity_threshold)

            # 获取最后一条客户消息
            last_customer_msg = conversations[-1]['content'] if conversations else ''

            # 创建带标签的记录，先添加SOP分析结果
            labeled_record = {
                "最后客户消息": last_customer_msg,
                "最近销售消息": sop_match.get('sales_message', '') if sop_match else '',
                "销售消息时间": sop_match.get('sales_time', '') if sop_match else '',
                "SOP一级节点": sop_match.get('first_node', '') if sop_match else '',
                "SOP二级节点": sop_match.get('second_node', '') if sop_match else '',
                "SOP三级节点": sop_match.get('third_node', '') if sop_match else '',
                "SOP四级节点": sop_match.get('fourth_node', '') if sop_match else '',
                "SOP五级节点": sop_match.get('fifth_node', '') if sop_match else '',
                "SOP六级节点": sop_match.get('sixth_node', '') if sop_match else '',
                "SOP七级节点": sop_match.get('seventh_node', '') if sop_match else '',
                "SOP八级节点": sop_match.get('eighth_node', '') if sop_match else '',
                "SOP九级节点": sop_match.get('ninth_node', '') if sop_match else '',
                "匹配相似度": sop_match.get('similarity', 0.0) if sop_match else 0.0,
                "匹配的参考话术": sop_match.get('matched_reference', '') if sop_match else '',
                "匹配类型": sop_match.get('match_type', '') if sop_match else ''
            }

            # 添加各层级的匹配信息（动态支持2-9级）
            level_names = ['', '一', '二', '三', '四', '五', '六', '七', '八', '九']

            if sop_match and 'all_level_matches' in sop_match:
                all_level_matches = sop_match['all_level_matches']

                # 动态处理各级节点匹配信息
                for level in range(2, 10):  # 支持二级到九级
                    level_name = level_names[level] if level < len(level_names) else f"第{level}"

                    if level in all_level_matches:
                        level_match = all_level_matches[level]
                        labeled_record[f"{level_name}级节点匹配话术"] = level_match.get('matched_reference', '')
                        labeled_record[f"{level_name}级节点相似度"] = level_match.get('similarity', 0.0)
                        labeled_record[f"{level_name}级节点匹配类型"] = level_match.get('match_type', '')
                    else:
                        labeled_record[f"{level_name}级节点匹配话术"] = ''
                        labeled_record[f"{level_name}级节点相似度"] = 0.0
                        labeled_record[f"{level_name}级节点匹配类型"] = ''
            else:
                # 没有匹配或者匹配结果不包含各层级信息
                for level in range(2, 10):  # 支持二级到九级
                    level_name = level_names[level] if level < len(level_names) else f"第{level}"
                    labeled_record[f"{level_name}级节点匹配话术"] = ''
                    labeled_record[f"{level_name}级节点相似度"] = 0.0
                    labeled_record[f"{level_name}级节点匹配类型"] = ''

            # 保留chengla-0925测试集.xlsx中的所有原始列
            for col in df.columns:
                if col not in labeled_record:
                    labeled_record[col] = row.get(col, '')

            labeled_records.append(labeled_record)

            # 每处理完batch_size行保存一次
            if (idx + 1) % batch_size == 0:
                batch_count += 1
                try:
                    temp_df = pd.DataFrame(labeled_records)
                    temp_df.to_excel(temp_file_path, index=False, engine='openpyxl')
                    print(f"\n已处理 {idx + 1} 行，保存进度到临时文件（批次 {batch_count}）")
                except Exception as save_error:
                    print(f"\n警告：保存临时文件失败 - {str(save_error)}")

        # 处理完成后保存最终结果
        if labeled_records:
            try:
                final_df = pd.DataFrame(labeled_records)
                final_df.to_excel(temp_file_path, index=False, engine='openpyxl')
                print(f"\n最终保存 {len(labeled_records)} 条记录到临时文件")
            except Exception as save_error:
                print(f"\n警告：最终保存临时文件失败 - {str(save_error)}")

        # 清理：如果处理成功，删除临时文件
        try:
            if os.path.exists(temp_file_path):
                print(f"处理完成，临时文件保存在：{temp_file_path}")
                print("如需清理临时文件，请手动删除")
        except Exception:
            pass

        print(f"完成处理，共生成 {len(labeled_records)} 条带标签的记录")
        return labeled_records

    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")
        print(traceback.format_exc())
        return []


def find_all_level_sop_matches(sales_message, sop_data, similarity_threshold):
    """
    为销售消息找到各层级的最佳匹配SOP节点

    新策略：
    1. 为每个层级（一级、二级、三级、四级）分别找到最佳匹配
    2. 返回所有层级的匹配结果，每个层级包含匹配的话术
    3. 支持完整的层级匹配输出

    参数:
        sales_message: 销售消息内容
        sop_data: SOP逻辑树数据
        similarity_threshold: 相似度阈值

    返回:
        dict: 包含各层级匹配信息的字典
    """
    all_matches = []  # 收集所有匹配项

    # Helper to recursively traverse the SOP tree
    def _traverse_sop_tree(current_node, current_path, depth):
        if not isinstance(current_node, dict):
            return

        # Check for '参考话术' or '相似话术' at the current node
        scripts_to_check = []
        if "参考话术" in current_node:
            scripts_to_check.append(("参考话术", current_node["参考话术"]))
        if "相似话术" in current_node:
            scripts_to_check.append(("相似话术", current_node["相似话术"]))

        for script_type, scripts in scripts_to_check:
            for script in scripts:
                if script and script.strip():
                    similarity = calculate_sentence_similarity(sales_message, script)
                    if similarity >= similarity_threshold:
                        match_info = {
                            'similarity': similarity,
                            'matched_reference': script,
                            'match_type': script_type,
                            'depth': depth,
                            'path': current_path
                        }
                        all_matches.append(match_info)
        
        # Recursively traverse child nodes
        for key, value in current_node.items():
            if isinstance(value, dict) and key not in ["参考话术", "相似话术", "下一步动作", "预期话术"]:
                _traverse_sop_tree(value, current_path + [key], depth + 1)

    _traverse_sop_tree(sop_data, [], 0) # Start traversal from the root

    if not all_matches:
        return None

    # Find the best match (highest similarity, then deepest level)
    all_matches.sort(key=lambda x: (-x['similarity'], -x['depth'], 0 if x['match_type'] == '参考话术' else 1))
    best_overall_match = all_matches[0]

    result = {
        'similarity': best_overall_match['similarity'],
        'matched_reference': best_overall_match['matched_reference'],
        'match_type': best_overall_match['match_type'],
        'all_level_matches': {}
    }
    
    best_path = best_overall_match['path']
    node_keys = ['first_node', 'second_node', 'third_node', 'fourth_node',
                 'fifth_node', 'sixth_node', 'seventh_node', 'eighth_node', 'ninth_node']

    for i, node_name in enumerate(best_path):
        result[node_keys[i]] = node_name

    # Find best match for each level on the best path
    for i in range(len(best_path)):
        level = i + 1
        sub_path = best_path[:level]
        
        best_match_for_level = None
        for match in all_matches:
            if match['path'] == sub_path:
                if best_match_for_level is None or match['similarity'] > best_match_for_level['similarity']:
                    best_match_for_level = match
        
        if best_match_for_level:
            result['all_level_matches'][level] = best_match_for_level

    # 仅保留具备参考话术的层级
    pruned_level_matches = {
        level: match
        for level, match in result['all_level_matches'].items()
        if match.get('matched_reference')
    }

    if not pruned_level_matches:
        return None

    result['all_level_matches'] = pruned_level_matches
    result['matched_levels'] = sorted(pruned_level_matches.keys())
    result['best_path'] = best_path
    result['unmatched_levels'] = [
        idx + 1 for idx in range(len(best_path)) if (idx + 1) not in pruned_level_matches
    ]

    return result


def find_best_sop_match(sales_message, sop_data, similarity_threshold):
    """
    为销售消息找到最匹配的SOP节点 - 兼容原有接口

    参数:
        sales_message: 销售消息内容
        sop_data: SOP逻辑树数据
        similarity_threshold: 相似度阈值

    返回:
        包含最佳匹配信息的字典，如果没有匹配则返回None
    """
    # 调用新的全层级匹配函数
    return find_all_level_sop_matches(sales_message, sop_data, similarity_threshold)


def find_best_similarity_candidate(sales_message, sop_data):
    """
    查找与销售话术最相近的SOP节点（无阈值，仅用于诊断信息）
    """
    best_match = None

    def _traverse(current_node, current_path, depth):
        nonlocal best_match
        if not isinstance(current_node, dict):
            return

        scripts_to_check = []
        if "参考话术" in current_node:
            scripts_to_check.append(("参考话术", current_node["参考话术"]))
        if "相似话术" in current_node:
            scripts_to_check.append(("相似话术", current_node["相似话术"]))

        for script_type, scripts in scripts_to_check:
            for script in scripts:
                if script and script.strip():
                    similarity = calculate_sentence_similarity(sales_message, script)
                    if best_match is None or similarity > best_match['similarity']:
                        best_match = {
                            'matched_reference': script,
                            'match_type': script_type,
                            'similarity': similarity,
                            'depth': depth,
                            'path': current_path
                        }

        for key, value in current_node.items():
            if isinstance(value, dict) and key not in ["参考话术", "相似话术", "下一步动作", "预期话术"]:
                _traverse(value, current_path + [key], depth + 1)

    _traverse(sop_data, [], 0)
    return best_match


def func_main(**kwargs):
    config_data = kwargs["config_data"]

    # 获取销售对话汇总文件路径（get_sales_query.py的输出）
    # 对于SOP分析，使用corpus_dir下的测试集文件
    corpus_dir = config_data["corpus_dir"]
    # 获取最终案例的保存路径
    pipeline_case_path = config_data["pipeline_case_path"]
    # 逻辑树节点路径
    sop_logic_tree = config_data["sop_logic_tree"]

    # 获取配置参数
    functions = config_data["functions"]
    similarity = 0.90  # 降低默认阈值从0.95到0.90
    batch_size = 100  # 默认批次大小

    for func_config in functions:
        if func_config.get("name") == "get_sop_case.func_main":
            similarity = func_config.get("similarity", 0.90)  # 默认0.90
            batch_size = func_config.get("batch_size", 100)

    print(f"开始分析对话记录并打SOP标签...")
    print(f"输入文件：{corpus_dir}")
    print(f"SOP逻辑树：{sop_logic_tree}")
    print(f"相似度阈值：{similarity}")
    print(f"批次大小：{batch_size}")
    print(f"处理逻辑：从历史对话最后一条客户消息开始，往前找最近的匹配SOP节点的销售消息")

    # 分析对话并打标签
    labeled_records = analyze_conversation_sop_labels(
        corpus_dir,
        sop_logic_tree,
        similarity,
        batch_size
    )

    # 保存结果
    if labeled_records:
        success = list_of_dicts_to_xlsx(labeled_records, pipeline_case_path)
        if success:
            print(f"成功保存标签结果到：{pipeline_case_path}")

            # 统计结果
            sop_stats = {}
            for record in labeled_records:
                if record['SOP一级节点']:
                    # 动态构建SOP路径，支持任意级别
                    path_parts = [record['SOP一级节点']]

                    # 检查各级节点是否存在
                    node_keys = ['SOP二级节点', 'SOP三级节点', 'SOP四级节点', 'SOP五级节点',
                                'SOP六级节点', 'SOP七级节点', 'SOP八级节点', 'SOP九级节点']

                    for node_key in node_keys:
                        if record.get(node_key):
                            path_parts.append(record[node_key])
                        else:
                            break  # 遇到空节点就停止

                    key = ' -> '.join(path_parts)
                    sop_stats[key] = sop_stats.get(key, 0) + 1

            print(f"\nSOP节点匹配统计：")
            for sop_path, count in sorted(sop_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {sop_path}: {count}条")

            total_labeled = sum(1 for r in labeled_records if r['SOP一级节点'])
            print(f"\n总计：{len(labeled_records)}条记录，{total_labeled}条成功标记SOP节点 ({total_labeled/len(labeled_records)*100:.1f}%)")
        else:
            print("保存结果文件失败")
    else:
        print("没有生成任何标签记录")


# if __name__ == "__main__":
#     # 测试案例1：语义相似的句子
#     sent1 = "你好今天早上吃什么"
#     sent2 = "你今天早上吃什么"
#
#     # 使用TF-IDF方法计算
#     similar = calculate_sentence_similarity(sent1, sent2)
#     print(f"句子1：{sent1}")
#     print(f"句子2：{sent2}")
#     print(f"相似度：{similar}")
