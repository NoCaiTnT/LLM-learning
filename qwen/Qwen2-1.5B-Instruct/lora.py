# 模型下载环境
from modelscope import snapshot_download
# 模型加载、训练环境
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
# Lora微调
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
# 数据集
from datasets import Dataset
import pandas as pd
import torch

def train_with_lora(model_path, dataset_path):
    # #模型下载
    # model_dir = snapshot_download('qwen/Qwen2-1.5B-Instruct', local_dir='../../models/Qwen2-1.5B-Instruct')

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )

    # 开启梯度
    model.enable_input_require_grads()

    # 读取数据集
    # 将JSON文件转换为CSV文件
    df = pd.read_json(dataset_path)
    ds = Dataset.from_pandas(df)
    print(ds[:3])

    # 数据格式化
    def process_func(example):
        MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(f"<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # 映射数据集
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    print(tokenized_id)

    print(tokenizer.decode(tokenized_id[0]['input_ids']))
    print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"]))))

    # 配置loar参数
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,   # 训练模式
        r=8,                    # Lora 秩
        lora_alpha=32,          # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1        # Dropout 比例
    )

    model = get_peft_model(model, config)
    print(config)
    print(model.print_trainable_parameters())

    # 配置训练参数
    train_args = TrainingArguments(
        output_dir="../../lora_output/Qwen2-1.5B-Instruct-lora",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=32,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        train_dataset=tokenized_id
    )

    # 开始训练
    trainer.train()

# 加载lora, 与model结合
def apply_lora(model_name_or_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda:0", torch_dtype=torch.bfloat16,
                                                trust_remote_code=True)

    print(f"Loading the LoRA adapter from {lora_path}")
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    return model, base_tokenizer

if __name__ == '__main__':
    # 路径设置
    # model_path = '../../models/Qwen2-1.5B-Instruct'
    model_path = "../../lora_merge/Qwen2-1.5B-Instruct-merge"
    lora_path = "../../lora_output/Qwen2-1.5B-Instruct-lora/checkpoint-9"
    merge_path = "../../lora_merge/Qwen2-1.5B-Instruct-merge"
    dataset_path = "../../datasets/huanhuan.json"

    # 训练
    train_with_lora(model_path, dataset_path)

    # 加载合并lora权重
    model, tokenizer = apply_lora(model_path, lora_path)

    # 保存路径
    print(f"Saving the target model to {merge_path}")
    model.save_pretrained(merge_path)
    tokenizer.save_pretrained(merge_path)