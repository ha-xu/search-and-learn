import argparse
import json
import sys
import os

def read_time_from_jsonl(file_path):
    """
    读取 .jsonl 文件，并打印每个样本中的 'time' 字段值。

    Args:
        file_path (str): .jsonl 文件的路径。
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到: {file_path}", file=sys.stderr)
        return

    print(f"--- 正在读取文件: {file_path} ---")

    try:
        # tokens_counts = []
        token_counts = []
        beam_counts_total = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                # 忽略空行或只包含空白字符的行
                if not line.strip():
                    continue
                try:
                    # 解析每一行为一个 JSON 对象
                    data = json.loads(line)

                    # # tokens_count_key = 'completion_tokens'
                    # llm_time_key = 'avg_llm_time_per_sample_batch'
                    # prm_time_key = 'avg_prm_time_per_sample_batch'
                    # total_time_key = 'avg_time_per_sample_batch'
                    # total_time_beam_search_key = 'total_time_beam_search'

                    token_count_key = 'completion_tokens'
                    beam_counts_total_key = 'beam_counts_total'
                    
                    token_count = data.get(token_count_key, None)
                    beam_counts_total = data.get(beam_counts_total_key, None)

                    if token_count is None:
                        print(f"警告: 样本 {line_number} 中缺少 '{token_count_key}' 字段。", file=sys.stderr)
                    if beam_counts_total is None:
                        print(f"警告: 样本 {line_number} 中缺少 '{beam_counts_total_key}' 字段。", file=sys.stderr)

                    if token_count is not None:
                        token_counts.append(token_count)
                    if beam_counts_total is not None:
                        beam_counts_total.append(beam_counts_total)
                    # # tokens_count = data.get(tokens_count_key, None)
                    # llm_time = data.get(llm_time_key, None)
                    # prm_time = data.get(prm_time_key, None)
                    # total_time = data.get(total_time_key, None)
                    # total_time_beam_search = data.get(total_time_beam_search_key, None)
                    
                    # # if tokens_count is not None:
                    # #     tokens_counts.append(sum(tokens_count))
                    
                    # if llm_time is not None:
                    #     llm_times.append(llm_time)
                    # if prm_time is not None:
                    #     prm_times.append(prm_time)
                    # if total_time is not None:
                    #     total_times.append(total_time)
                    # if total_time_beam_search is not None:
                    #     total_time_beam_searches.append(total_time_beam_search)
                except json.JSONDecodeError:
                    print(f"错误: 样本 {line_number} 不是有效的 JSON 格式，已跳过。", file=sys.stderr)
                except Exception as e:
                    print(f"处理样本 {line_number} 时发生未知错误: {e}", file=sys.stderr)
        # print(f"--- 读取完成，共处理 {len()} 个样本 ---")
        #计算token数平均值
        # if len(tokens_counts) > 0:
        #     avg_tokens = sum(tokens_counts) / len(tokens_counts)
        #     print(f"平均 Token 数: {avg_tokens:.2f}")
        # 计算并打印平均时间
        # if len(llm_times) > 0:
        #     avg_llm_time = sum(llm_times) / len(llm_times)
        #     print(f"平均 LLM 时间: {avg_llm_time:.6f} 秒")
        # if len(prm_times) > 0:
        #     avg_prm_time = sum(prm_times) / len(prm_times)
        #     print(f"平均 PRM 时间: {avg_prm_time:.6f} 秒")
        # if len(total_times) > 0:
        #     avg_total_time = sum(total_times) / len(total_times)
        #     print(f"平均 总时间: {avg_total_time:.6f} 秒")
        # if len(total_time_beam_searches) > 0:
        #     avg_total_time_beam_search = sum(total_time_beam_searches) / len(total_time_beam_searches)
        #     print(f"平均 Beam Search 总时间: {avg_total_time_beam_search:.6f} 秒")
        if len(token_counts) > 0:
            avg_token_count = sum(token_counts) / len(token_counts)
            print(f"平均 Token 数: {avg_token_count:.2f}")

        if len(beam_counts_total) > 0:
            avg_beam_counts_total = sum(beam_counts_total) / len(beam_counts_total)
            print(f"平均 Beam 数: {avg_beam_counts_total:.2f}")
    except IOError as e:
        print(f"错误: 无法打开或读取文件: {e}", file=sys.stderr)


def main():
    """
    设置命令行参数解析器并调用读取函数。
    """
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="从 .jsonl 文件中读取每个样本的 'time' 字段值。"
    )

    # 添加文件路径参数
    parser.add_argument(
        "jsonl_file",
        type=str,
        help="要读取的 .jsonl 文件路径。"
    )

    # 解析参数
    args = parser.parse_args()

    # 调用主读取函数
    read_time_from_jsonl(args.jsonl_file)


if __name__ == "__main__":
    main()