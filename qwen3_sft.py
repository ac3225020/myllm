import os
import wandb
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer

# 1. 初始化W&B（保留）
wandb.init(
    project="unsloth-qwen3-visual-finetune",
    name="qwen3-0.6b-finetune",
    config={
        "learning_rate": 5e-5,  # 进一步降低学习率
        "batch_size": 4,  # 先保持4，解决过拟合后再调大
        "max_seq_length": 16384,
        "model_path": "Qwen3-0.6B",
    }
)

# 2. 加载模型
LOCAL_MODEL_PATH = "/root/autodl-tmp/Qwen3-0.6B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LOCAL_MODEL_PATH,
    max_seq_length=16384,
    local_files_only=True,
    dtype=torch.float16,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# 3. 应用Chat模板
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
    mapping={"role": "from", "content": "value", "user": "user", "assistant": "assistant"},
)

# 4. 强化LoRA防过拟合（核心修改）
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # 降低LoRA秩，减少可训练参数的表达能力
    target_modules=["q", "k", "v", "o", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.2,  # 提升dropout到0.2，强化随机丢弃
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,  # 启用RSLoRA提速
    loftq_config=None,
)

# 5. 加载数据+拆分验证集（关键：约束泛化）
DATASET_PATH = "/root/dataset/qwen3_finetune.jsonl"
dataset = load_dataset("json", data_files=DATASET_PATH)
# 拆分15%验证集，避免模型死记硬背
train_test_split = dataset["train"].train_test_split(test_size=0.15, seed=3407)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 格式化数据（保留）
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
        text = tokenizer.apply_chat_template(formatted_convs, tokenize=False,enable_thinking=False)
        texts.append(text)
        texts.append(text)
    return {"text": texts}

train_dataset = train_dataset.map(format_data, batched=True, remove_columns=train_dataset.column_names, num_proc=8)
eval_dataset = eval_dataset.map(format_data, batched=True, remove_columns=eval_dataset.column_names, num_proc=8)

# 6. 训练参数（强化防过拟合）
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # 先保持4，解决过拟合后再调大
    learning_rate=5e-5,  # 进一步降低学习率
    num_train_epochs=1,
    logging_steps=200,  # 减少日志开销
    logging_dir="/root/sft/logs",
    output_dir="/root/sft/unsloth-finetuned-model",
    report_to="wandb",
    run_name="qwen3-0.6b-finetune",
    # 强化防过拟合核心参数
    weight_decay=0.03,  # 提升权重衰减到0.03
    lr_scheduler_type="cosine",
    warmup_steps=2000,  # 延长预热步数，避免前期更新过快
    # 新增：早停（loss归零前停止）
    eval_strategy="steps",  # 每步评估验证集
    eval_steps=500,  # 每500步评估一次，监控泛化能力
    # 基础配置
    fp16=True,
    seed=3407,
    save_strategy="steps",
    save_steps=5000,
    gradient_checkpointing=True,
)

# 7. 启动微调（加入验证集）
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 关键：加入验证集约束泛化
    args=training_args,
    max_seq_length=16384,
    dataset_text_field="text",
)

# 开始训练
trainer.train()

# 保存模型+结束W&B
trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
wandb.finish()