from llama_cpp import Llama

# Load GGUF or quantized model
llm = Llama(model_path="path/to/llama-2-7b.gguf")

# Run inference
prompt = "Explain LoRA fine-tuning in simple terms."
output = llm(prompt, max_tokens=200)
print(output["choices"][0]["text"])
