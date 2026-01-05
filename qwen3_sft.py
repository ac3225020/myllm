import os
import wandb
from datasets import load_from_disk
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer

# ====================== 1. 初始化W&B（可视化核心） ======================
# 先在终端执行：wandb login（输入你的W&B API Key，注册地址：https://wandb.ai/）
wandb.init(
    project="unsloth-qwen3-visual-finetune",  # 你的项目名
    name="qwen3-0.6b-finetune",                 # 本次微调任务名
    config={                                  # 要监控的参数（可自定义）
        "learning_rate": 2e-4,
        "batch_size": 4,
        "max_seq_length": 2048,
        "model_path": "Qwen3-0.6B",
    }
)

# ====================== 2. 加载模型（Unsloth常规操作） ======================
LOCAL_MODEL_PATH = "/root/autodl-tmp/Qwen3-0.6B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LOCAL_MODEL_PATH,
    local_files_only=True,
    torch_dtype="float16",
    load_in_4bit=True,  # 节省显存
)

# 应用Chat模板（以Qwen3为例）
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen",
    mapping={"role": "from", "content": "value", "user": "user", "assistant": "assistant"},
)

# ====================== 3. 准备微调数据 ======================
# 从本地加载标准Qwen3格式的数据集
DATASET_PATH = "/root/dataset/qwen3_finetune.jsonl"
train_dataset = load_from_disk(DATASET_PATH)

# 格式化数据函数
def format_data(examples):
    # 将数据转换为Qwen3对话格式
    texts = []
    for i in range(len(examples['conversations'])):
        # 将conversations转换为标准对话格式
        conversations = examples['conversations'][i]
        formatted_convs = []
        for conv in conversations:
            formatted_convs.append({
                "from": conv.get("from", conv.get("role", "user")),
                "value": conv.get("value", conv.get("content", ""))
            })
        # 应用chat模板
        text = tokenizer.apply_chat_template(formatted_convs, tokenize=False)
        texts.append(text)
    return {"text": texts}

# 对数据集进行格式化
train_dataset = train_dataset.map(
    format_data,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Formatting dataset"
)

# ====================== 4. 配置微调参数（加入W&B回调） ======================
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,  # 每1步记录一次日志（用于可视化）
    logging_dir="/root/sft/logs",  # 日志保存路径（供W&B读取）
    output_dir="/root/sft/unsloth-finetuned-model",
    report_to="wandb",  # 关键：将训练数据上报到W&B
    run_name="qwen3-0.6b-finetune",
)

# ====================== 5. 启动微调（自动可视化） ======================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
    max_seq_length=2048,
    dataset_text_field="text",
)

# 开始微调（W&B会实时监控）
trainer.train()

# 结束W&B监控
wandb.finish()