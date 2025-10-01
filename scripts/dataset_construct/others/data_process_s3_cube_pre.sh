LOCAL_DIR=data/cube

pwd

## process multiple dataset search format test file
DATA=2wikimultihopqa
python scripts/data_process/test_s3_cube_pre.py --local_dir $LOCAL_DIR --data_sources $DATA --retriever e5

DATA=hotpotqa
python scripts/data_process/test_s3_cube_pre.py --local_dir $LOCAL_DIR --data_sources $DATA --retriever e5

DATA=lveval
python scripts/data_process/test_s3_cube_pre.py --local_dir $LOCAL_DIR --data_sources $DATA --retriever e5

DATA=musique
python scripts/data_process/test_s3_cube_pre.py --local_dir $LOCAL_DIR --data_sources $DATA --retriever e5