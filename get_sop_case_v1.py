import traceback
import jieba
import pandas as pd
import os
import json
import re
import math

from collections import Counter
from typing import Dict, Tuple
from tqdm import tqdm


def preprocess_sentence(sentence):
    """文本预处理：全角转半角、小写转换、清理特殊字符和空格"""
    if not isinstance(sentence, str) or sentence.strip() == "":
        return ""

    # 全角转半角（解决数字/符号格式问题）
    def full_to_half(s):
        result = []
        for char in s:
            code = ord(char)
            if code == 12288:  # 全角空格转半角
                result.append(" ")
            elif 65281 <= code <= 65374:  # 全角字符转半角
                result.append(chr(code - 65248))
            else:
                result.append(char)
        return "".join(result)

    sentence = full_to_half(sentence)
    sentence = sentence.lower()  # 小写转换（英文生效，中文无影响）
    sentence = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", sentence)  # 保留中文/字母/数字
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence

_sentence_vector_cache: Dict[str, Tuple[Counter, float]] = {}

def get_sentence_vector(sentence: str) -> Tuple[Counter, float]:
    """返回句子的词频向量及其模长，带缓存以避免重复分词"""
    cached = _sentence_vector_cache.get(sentence)
    if cached is not None:
        return cached

    processed = preprocess_sentence(sentence)
    if not processed:
        vector = Counter(), 0.0
        _sentence_vector_cache[sentence] = vector
        return vector

    tokens = [token for token in jieba.cut(processed) if token]
    counter = Counter(tokens)
    norm = math.sqrt(sum(freq * freq for freq in counter.values()))
    vector = (counter, norm)
    _sentence_vector_cache[sentence] = vector
    return vector

def calculate_sentence_similarity(sentence1, sentence2):
    """计算两句话的余弦相似度（支持中文，仅返回相似度值0~1）"""
    vector1, norm1 = get_sentence_vector(sentence1)
    vector2, norm2 = get_sentence_vector(sentence2)

    if norm1 == 0 or norm2 == 0 or not vector1 or not vector2:
        return 0.0

    common_tokens = vector1.keys() & vector2.keys()
    dot_product = sum(vector1[token] * vector2[token] for token in common_tokens)

    return round(dot_product / (norm1 * norm2), 4) if dot_product else 0.0


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


def save_cases_in_batches(data_list, output_file, batch_size):
    """
    支持按照指定批次大小将案例列表拆分保存。
    batch_size<=0 或 None 时退化为单文件输出。
    """
    if batch_size is None or batch_size <= 0:
        list_of_dicts_to_xlsx(data_list, output_file)
        return

    if not data_list:
        print("没有可保存的案例数据，跳过批量写入")
        return

    base, ext = os.path.splitext(output_file)
    total = len(data_list)
    num_batches = (total + batch_size - 1) // batch_size
    padding = max(2, len(str(num_batches)))
    print(f"开始按批次写入案例：共 {total} 条，每批 {batch_size} 条，输出 {num_batches} 个文件")

    part_files = []
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        chunk = data_list[start:end]
        chunk_file = f"{base}_part{batch_idx + 1:0{padding}d}{ext}"
        print(f"写入第 {batch_idx + 1}/{num_batches} 批：{start}-{end - 1}")
        list_of_dicts_to_xlsx(chunk, chunk_file)
        part_files.append(chunk_file)

    print(f"批量写入完成，共生成 {len(part_files)} 个分段文件")


def context_to_testcase(id_list, first_node_list, second_node_list, third_node_list, excel_path, mode):
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
    third_node_list (list): 与id_list对应的三级SOP节点列表（当存在三级节点时填值，否则为空字符串）
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
                third_sop_node = third_node_list[i] if third_node_list else ""
                # 获取从分割点的下一行开始的所有下文
                subsequent = extract_dialogs(excel_path, start_index + 1, total_rows)
                case_dict = concat_dialogues_by_id(
                    excel_path,
                    start_index,
                    first_sop_node,
                    second_sop_node,
                    third_sop_node,
                    customer_reply,
                    subsequent,
                    case_cut_times
                )
                case_cut_times += 1
                case_list.append(case_dict)
            else:
                continue
    return case_list


def concat_dialogues_by_id(file_path, target_id, first_sop, second_sop, third_sop, customer_reply, subsequent, case_cut_times):
    """
    读取xlsx文件中ID从1到指定ID的所有行，按格式拼接成字符串并保存到字典中
    参数:
        file_path: xlsx文件路径
        target_id: 目标ID编号（整数）
        first_sop: SOP一级节点
        second_sop: SOP二级节点
        third_sop: SOP三级节点，可为空字符串
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
            "SOP三级节点": third_sop,
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

    third_level_entries = []
    second_level_entries = []

    for first_node, second_nodes_data in json_data.items():
        if not isinstance(second_nodes_data, dict):
            continue
        for second_node, content in second_nodes_data.items():
            if not isinstance(content, dict):
                continue
            second_similar = content.get("相似话术", []) or []
            if second_similar:
                for sentence in second_similar:
                    get_sentence_vector(sentence)
                second_level_entries.append({
                    "first": first_node,
                    "second": second_node,
                    "similar": second_similar,
                })
            for third_node, third_content in content.items():
                if third_node in {"参考话术", "相似话术"}:
                    continue
                if not isinstance(third_content, dict):
                    continue
                similar_list_lvl3 = third_content.get("相似话术", []) or []
                if not similar_list_lvl3:
                    continue
                for sentence in similar_list_lvl3:
                    get_sentence_vector(sentence)
                third_level_entries.append({
                    "first": first_node,
                    "second": second_node,
                    "third": third_node,
                    "similar": similar_list_lvl3,
                })

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
                xlsx_third_sop_list = []
                xlsx_id_list = []

                # 遍历文件中的所有行
                for index, row in df.iterrows():
                    sales_context = row['处理后对话内容']
                    # 如果是客户的对话，则跳过
                    if row["发送消息角色"] == "客户":
                        continue
                    row_id = row['ID']
                    # 检查是否存在对应的sop话术
                    found = False
                    get_sentence_vector(sales_context)  # 预先缓存销售话术的向量

                    # 优先匹配三级节点
                    for node in third_level_entries:
                        similar_list_lvl3 = node["similar"]
                        if not similar_list_lvl3:
                            continue
                        if use_similar_match:
                            for similar_sentence in similar_list_lvl3:
                                if calculate_sentence_similarity(sales_context, similar_sentence) >= similarity:
                                    xlsx_first_sop_list.append(node["first"])
                                    xlsx_second_sop_list.append(node["second"])
                                    xlsx_third_sop_list.append(node["third"])
                                    xlsx_id_list.append(row_id)
                                    found = True
                                    break
                        else:
                            if sales_context in similar_list_lvl3:
                                xlsx_first_sop_list.append(node["first"])
                                xlsx_second_sop_list.append(node["second"])
                                xlsx_third_sop_list.append(node["third"])
                                xlsx_id_list.append(row_id)
                                found = True
                        if found:
                            break

                    if found:
                        continue

                    # 二级节点匹配
                    for node in second_level_entries:
                        similar_list = node["similar"]
                        if not similar_list:
                            continue
                        if use_similar_match:
                            for similar_sentence in similar_list:
                                if calculate_sentence_similarity(sales_context, similar_sentence) >= similarity:
                                    xlsx_first_sop_list.append(node["first"])
                                    xlsx_second_sop_list.append(node["second"])
                                    xlsx_third_sop_list.append("")
                                    xlsx_id_list.append(row_id)
                                    found = True
                                    break
                        else:
                            if sales_context in similar_list:
                                xlsx_first_sop_list.append(node["first"])
                                xlsx_second_sop_list.append(node["second"])
                                xlsx_third_sop_list.append("")
                                xlsx_id_list.append(row_id)
                                found = True
                        if found:
                            break
                # 根据每个SOP的ID（对应在聊天语料中的位置）、SOP一级节点、SOP二级节点等信息，对案例进行切分
                single_xlsx_case_list = context_to_testcase(
                    xlsx_id_list,
                    xlsx_first_sop_list,
                    xlsx_second_sop_list,
                    xlsx_third_sop_list,
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


def func_main(**kwargs):
    config_data = kwargs["config_data"]
    # 获取所有语料的路径
    corpus_dir = config_data["corpus_dir"]
    # 获取最终案例的保存路径
    pipeline_case_path = config_data["pipeline_case_path"]
    # 逻辑树节点路径
    sop_logic_tree = config_data["sop_logic_tree"]

    functions = config_data["functions"]
    mode = ""
    # 设置默认不使用相似度匹配
    use_similar_match = False
    similarity = 0.95
    for func_config in functions:
        if func_config.get("name") == "get_sop_case.func_main":
            mode = func_config.get("mode")
            use_similar_match = func_config.get("use_similar_match")
            similarity = func_config.get("similarity")
            batch_size = func_config.get("batch_size", 100)
            break  # 找到配置即可退出循环
    else:
        batch_size = 0

    final_case_list = read_xlsx_files(corpus_dir, sop_logic_tree, mode, use_similar_match, similarity)
    save_cases_in_batches(final_case_list, pipeline_case_path, batch_size)
    if final_case_list:
        print("开始生成完整案例汇总文件")
        list_of_dicts_to_xlsx(final_case_list, pipeline_case_path)


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
