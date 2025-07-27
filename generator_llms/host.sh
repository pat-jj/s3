export CUDA_VISIBLE_DEVICES=6,7

python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
    --port 8000 \
    --max-model-len 8192 \
    --tensor-parallel-size 2

    # --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
    # --model Qwen/Qwen2.5-7B-Instruct-GGUF\