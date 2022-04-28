#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6091
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1


########################## Evaluate Refcoco ##########################
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
selected_cols=0,4,2,3

data=../../dataset/refcoco_data/refcoco_val.tsv
path=../../checkpoints/refcoco_base_best.pt
result_path=../../results/refcoco
split='refcoco_val'
python3 ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

data=../../dataset/refcoco_data/refcoco_testA.tsv
path=../../checkpoints/refcoco_base_best.pt
result_path=../../results/refcoco
split='refcoco_testA'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

data=../../dataset/refcoco_data/refcoco_testB.tsv
path=../../checkpoints/refcoco_base_best.pt
result_path=../../results/refcoco
split='refcoco_testB'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
