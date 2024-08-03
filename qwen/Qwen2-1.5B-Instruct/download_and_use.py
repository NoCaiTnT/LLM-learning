import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

#模型下载
model_dir = snapshot_download('qwen/Qwen2-1.5B-Instruct', local_dir='../../models/Qwen2-1.5B-Instruct')

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto"
)

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 提示词
prompt = "给我介绍一下大语言模型"

# 构造消息文本
messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": prompt}
]

# 将消息分词, 并放入显卡
model_inputs = tokenizer.apply_chat_template(
    messages,
    tokenizer=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to(device)

# 生成文本
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

generated_ids = [
    output_ids[len(input_idx):] for input_idx, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码生成的文本
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)