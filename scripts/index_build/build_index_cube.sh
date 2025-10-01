# corpus_file=/shared/eng/pj20/cube_data/hotpotqa_corpus_with_index.json # json
# save_dir=/shared/eng/pj20/cube_data/hotpot_index
# retriever_name=e5 # this is for indexing naming
# retriever_model=intfloat/e5-base-v2

# # change faiss_type to HNSW32/64/128 for ANN indexing
# # change retriever_name to bm25 for BM25 indexing
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python s3/search/index_builder.py \
#     --retrieval_method $retriever_name \
#     --model_path $retriever_model \
#     --corpus_path $corpus_file \
#     --save_dir $save_dir \
#     --use_fp16 \
#     --max_length 256 \
#     --batch_size 512 \
#     --pooling_method mean \
#     --faiss_type Flat \
#     --save_embedding


corpus_file=/shared/eng/pj20/cube_data/musique_corpus_with_index.json # json
save_dir=/shared/eng/pj20/cube_data/musique_index
retriever_name=e5 # this is for indexing naming
retriever_model=intfloat/e5-base-v2

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python s3/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding


# corpus_file=/shared/eng/pj20/cube_data/hotpotqa_corpus.jsonl # jsonl
# save_dir=/shared/eng/pj20/cube_data/hotpot_index
# retriever_name=bm25 # this is for indexing naming
# retriever_model=bm25

# # change faiss_type to HNSW32/64/128 for ANN indexing
# # change retriever_name to bm25 for BM25 indexing
# CUDA_VISIBLE_DEVICES=0,1 python s3/search/index_builder.py \
#     --retrieval_method $retriever_name \
#     --model_path $retriever_model \
#     --corpus_path $corpus_file \
#     --save_dir $save_dir \
#     --use_fp16 \
#     --max_length 256 \
#     --batch_size 512 \
#     --pooling_method mean \
#     --faiss_type Flat \
#     --save_embedding