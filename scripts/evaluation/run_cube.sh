python scripts/evaluation/context_cube.py \
  --context_file data/cube/output_sequences_s3_8_3_3_3948_cube_2wikimultihopqa/2wikimultihopqa_output_sequences.json \
  --dataset_file data/cube/2wikimultihopqa_with_index.json \
  --output_file data/cube/results_s3_cube_2wikimultihopqa_.json \
  --api_url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct

python scripts/evaluation/context_cube.py \
  --context_file data/cube/output_sequences_s3_8_3_3_3948_cube_hotpotqa/hotpotqa_output_sequences.json \
  --dataset_file data/cube/hotpotqa_with_index.json \
  --output_file data/cube/results_s3_cube_hotpotqa_.json \
  --api_url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct

python scripts/evaluation/context_cube.py \
  --context_file data/cube/output_sequences_r1_7b/2wikimultihopqa_output_sequences.json \
  --dataset_file data/cube/2wikimultihopqa_with_index.json \
  --output_file data/cube/results_r1_7b_cube_2wikimultihopqa_.json \
  --api_url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct

python scripts/evaluation/context_cube.py \
  --context_file data/cube/output_sequences_r1_7b/hotpotqa_output_sequences.json \
  --dataset_file data/cube/hotpotqa_with_index.json \
  --output_file data/cube/results_r1_7b_cube_hotpotqa_.json \
  --api_url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct