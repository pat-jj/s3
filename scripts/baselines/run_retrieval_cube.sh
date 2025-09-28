    # python scripts/baselines/e5_retrieval_cube.py \
    #     --input_parquet data/cube/test_e5_cube_2wikimultihopqa_pre.parquet \
    #     --output_dir data/cube/rag_e5_cube \
    #     --endpoint http://127.0.0.1:3000/retrieve \
    #     --data_sources 2wikimultihopqa

    python scripts/baselines/e5_retrieval_cube.py \
        --input_parquet data/cube/test_e5_cube_hotpotqa_pre.parquet \
        --output_dir data/cube/rag_e5_cube \
        --endpoint http://127.0.0.1:3000/retrieve \
        --data_sources hotpotqa


