export CUDA_VISIBLE_DEVICES=0,1
file_path=/shared/eng/pj20/cube_data/2wqa_index
index_file=$file_path/e5_Flat.index
corpus_file=/shared/eng/pj20/cube_data/2wikimultihopqa_corpus_with_index.json
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python /home/pj20/server-04/search-c1/s3/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 12 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu \
                                            --port 3000


# export CUDA_VISIBLE_DEVICES=0,1
# file_path=/shared/eng/pj20/cube_data/hotpot_index
# index_file=$file_path/e5_Flat.index
# corpus_file=/shared/eng/pj20/cube_data/hotpotqa_corpus_with_index.json
# retriever_name=e5
# retriever_path=intfloat/e5-base-v2

# python /home/pj20/server-04/search-c1/s3/search/retrieval_server.py --index_path $index_file \
#                                             --corpus_path $corpus_file \
#                                             --topk 12 \
#                                             --retriever_name $retriever_name \
#                                             --retriever_model $retriever_path \
#                                             --faiss_gpu \
#                                             --port 3000
