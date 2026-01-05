import json
import os


def convert_jsonl_to_qwen3_format(input_file: str, output_file: str, output_type: str = "jsonl"):
    """
    将原始JSONL数据转换为Qwen3微调格式
    :param input_file: 输入的原始JSONL文件路径
    :param output_file: 输出的Qwen3格式文件路径
    :param output_type: 输出格式，可选 "json"（JSON数组）或 "jsonl"（每行一个JSON）
    """
    # 校验输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件 {input_file} 不存在！")

    # 存储转换后的样本
    converted_samples = []

    # 读取并处理原始JSONL文件
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue

            try:
                # 解析单行JSON数据
                raw_data = json.loads(line)

                # 提取核心字段（处理字段缺失的情况）
                question = raw_data.get("question", "").strip()
                answer = raw_data.get("answer", "").strip()

                # 跳过无效样本（问题/回答为空）
                if not question or not answer:
                    print(f"第{line_num}行：问题或回答为空，跳过该样本")
                    continue

                # 构造Qwen3微调格式的样本
                qwen3_sample = {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                }
                converted_samples.append(qwen3_sample)

            except json.JSONDecodeError:
                print(f"第{line_num}行：JSON格式错误，跳过该样本")
            except Exception as e:
                print(f"第{line_num}行：处理失败 - {str(e)}，跳过该样本")

    # 保存转换后的结果
    with open(output_file, "w", encoding="utf-8") as f:
        if output_type == "json":
            # 保存为JSON数组（小样本量推荐）
            json.dump(converted_samples, f, ensure_ascii=False, indent=2)
        elif output_type == "jsonl":
            # 保存为JSONL（大样本量微调推荐，逐行写入）
            for sample in converted_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            raise ValueError("output_type 仅支持 'json' 或 'jsonl'！")

    print(f"转换完成！共处理 {len(converted_samples)} 个有效样本，结果已保存至 {output_file}")


# ==================== 示例调用 ====================
if __name__ == "__main__":
    # 替换为你的输入/输出文件路径
    INPUT_JSONL = "/Users/zhangjiamin/Downloads/format_data.jsonl"  # 原始JSONL数据文件
    OUTPUT_FILE = "qwen3_finetune.jsonl"  # 转换后的Qwen3微调数据文件

    # 执行转换（输出为JSONL格式，适合大样本微调）
    convert_jsonl_to_qwen3_format(
        input_file=INPUT_JSONL,
        output_file=OUTPUT_FILE,
        output_type="jsonl"
    )