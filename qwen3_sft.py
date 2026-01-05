import os
import wandb
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer

# ====================== 1. 初始化W&B（保留所有配置） ======================
wandb.init(
    project="unsloth-qwen3-visual-finetune",
    name="qwen3-0.6b-finetune",
    config={
        "learning_rate": 1e-4,  # 修正：从2e-4改为1e-4（适配小模型）
        "batch_size": 4,
        "max_seq_length": 16384,  # 统一上下文长度
        "model_path": "Qwen3-0.6B",
    }
)

# ====================== 2. 加载模型（统一max_seq_length） ======================
LOCAL_MODEL_PATH = "/root/autodl-tmp/Qwen3-0.6B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LOCAL_MODEL_PATH,
    max_seq_length=16384,  # 核心修正：匹配Qwen3-0.6B官方最大上下文长度
    local_files_only=True,
    dtype=torch.float16,
    load_in_4bit=True,
)

# 应用Chat模板（保留）
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
    mapping={"role": "from", "content": "value", "user": "user", "assistant": "assistant"},
)

# ====================== 2.5 配置LoRA适配器（核心：增加防过拟合） ======================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.1,  # 核心修正：从0→0.1，随机丢弃参数防过拟合
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ====================== 3. 准备微调数据（逻辑无问题，保留） ======================
DATASET_PATH = "/root/dataset/qwen3_finetune.jsonl"
train_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_data(examples):
    texts = []
    for i in range(len(examples['messages'])):
        messages = examples['messages'][i]
        formatted_convs = []
        for msg in messages:
            formatted_convs.append({
                "from": msg.get("role", "user"),
                "value": msg.get("content", "")
            })
        text = tokenizer.apply_chat_template(formatted_convs, tokenize=False)
        texts.append(text)
    return {"text": texts}

train_dataset = train_dataset.map(
    format_data,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Formatting dataset"
)

# ====================== 4. 配置微调参数（核心：防过拟合+适配小模型） ======================
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    learning_rate=1e-4,  # 核心修正：从2e-4→1e-4（小模型降低学习率）
    num_train_epochs=1,  # 核心修正：从3→1（避免小模型过度训练）
    logging_steps=100,  # 修正：从1→100，减少冗余日志输出
    logging_dir="/root/sft/logs",
    output_dir="/root/sft/unsloth-finetuned-model",
    report_to="wandb",  # 保留W&B上报
    run_name="qwen3-0.6b-finetune",
    # 新增：防过拟合核心参数
    weight_decay=0.01,  # 权重衰减，抑制过拟合
    lr_scheduler_type="cosine",  # 余弦学习率衰减，平缓更新
    warmup_steps=1000,  # 学习率预热，避免前期更新过快
    # 补充：保证训练稳定性的基础配置
    fp16=True,  # 匹配模型float16精度
    seed=3407,  # 固定随机种子，结果可复现
    save_strategy="epoch",  # 按轮保存，避免频繁写盘
    gradient_checkpointing=True,  # 节省显存
)

# ====================== 5. 启动微调（统一max_seq_length） ======================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
    max_seq_length=16384,  # 核心修正：从20480→2048，匹配模型上限
    dataset_text_field="text",
)

# 开始微调
trainer.train()

# 保存最终模型（新增：避免训练后模型丢失）
trainer.save_model(os.path.join(training_args.output_dir, "final_model"))

# 结束W&B监控（保留）
wandb.finish()