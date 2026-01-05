import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer


def qwen3_inference(model_path, test_question, max_new_tokens=512):
    """
    Qwen3-0.6B最简推理函数（仅测试单个模型）
    :param model_path: 微调后模型路径
    :param test_question: 用户输入的问题（回车结束输入）
    :param max_new_tokens: 最大生成回答长度
    """
    # 加载模型和tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    # 推理模式
    FastLanguageModel.for_inference(model)
    # 流式输出（逐字打印回答）
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 构造Qwen3对话格式
    prompt = f"""<|im_start|>user
{test_question}<|im_end|>
<|im_start|>assistant
"""

    # 生成回答
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )


# ==================== 交互式单模型测试 ====================
if __name__ == "__main__":
    # 配置：仅需修改微调后模型的路径
    FINETUNED_MODEL_PATH = "qwen3-0.6b-finetuned"  # 替换为你的微调模型路径

    print("===== Qwen3-0.6B 微调模型推理测试 =====")
    print("提示：输入问题后按回车即可获取回答；输入'q'/'quit'/'退出'可退出程序\n")

    while True:
        # 接收用户输入（按回车结束输入）
        test_question = input("请输入测试问题：").strip()

        # 退出条件
        if test_question.lower() in ["q", "quit", "退出"]:
            print("\n程序已退出！")
            break

        # 空输入校验（仅按回车的情况）
        if not test_question:
            print("错误：问题不能为空，请重新输入！\n")
            continue

        # 触发推理（仅测试微调后的单个模型）
        print("\n===== 模型回答 =====")
        try:
            qwen3_inference(FINETUNED_MODEL_PATH, test_question)
        except Exception as e:
            print(f"推理出错：{str(e)}")
        # 分隔符，区分不同问题的回答
        print("\n" + "-" * 80 + "\n")