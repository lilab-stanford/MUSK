
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HOST_NUM=1
export INDEX=0
export CHIEF_IP=localhost
export HOST_GPU_NUM=8
export PORT_ID=1234

PROCESS_NUM=$((HOST_GPU_NUM * HOST_NUM))
echo ${PROCESS_NUM}

MODEL="pretrain_musk_large"
LOG_DIR=./results/${MODEL}/log
OUTPUT_DIR=./results/${MODEL}

accelerate launch --gpu_ids all --use_deepspeed --num_processes ${PROCESS_NUM} \
  --deepspeed_config_file zero2_config.json --mixed_precision bf16 \
  --num_machines ${HOST_NUM} --machine_rank ${INDEX} --main_process_ip ${CHIEF_IP} --main_process_port ${PORT_ID} \
  --deepspeed_multinode_launcher standard ../run_pretraining.py   \
  --output_dir ${OUTPUT_DIR} --log_dir ${LOG_DIR} --config "../configs/${MODEL}.yaml"

