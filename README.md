# LLM-learning

## 量化（llama.cpp）
### 下载文件到本地
```
git clone https://github.com/ggerganov/llama.cpp.git
```

### 编译
进入根目录下
```
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

### 创建量化环境
```
conda create -n llamacpp python=3.11
```

### 安装依赖包
进入根目录下
```
pip install -r requirements.txt
```

### 将模型转换为gguf格式
改为你模型的目录(safetensor所在的目录)
```
python convert_hf_to_gguf.py D:\PythonProjects\LLMLora\lora_merge\Qwen2-1.5B-Instruct-merge
```

### 量化
进入build\bin\Release下
- 第一个参数: 需要量化的gguf模型 路径
- 第二个参数: 量化后的gguf模型 路径
- 第三个参数: 量化的模式
```
llama-quantize D:\PythonProjects\LLMLora\lora_merge\Qwen2-1.5B-Instruct-merge\Qwen2-1.5B-Instruct-merge-F16.gguf D:\PythonProjects\LLMLora\lora_merge\Qwen2-1.5B-Instruct-merge\Qwen2-1.5B-Instruct-merge-Q8_0.gguf Q8_0
```

### 开启对话
```
llama-cli -m your_model.gguf -p "You are a helpful assistant" -cnv
```