#!/usr/bin/env bash
i=2
exper_name="debug${i}"
res=`screen -ls`
if [[ ${res} =~ "${exper_name}" ]]
then
    echo 'screen $exper_name already created.'
else
    screen -dmS $exper_name
    echo 'create screen $exper_name'
fi

screen -r $exper_name -X stuff "conda activate python36\n"

debug_data_dir=/home/yfliu/data/Github/gnn-dep-parsing/data
data_dir=/home/taoji/data/dataset/DEP/PTB
ckpt_dir=../ckpts/$exper_name
screen -r $exper_name -X stuff "CUDA_VISIBLE_DEVICES=5 python train.py \
--TRAIN $data_dir/train.conllu \
--DEV $data_dir/dev.conllu \
--TEST $data_dir/test.conllu \
--PRED_DEV $ckpt_dir/dev.pred \
--PRED_TEST $ckpt_dir/test.pred \
--LOG $ckpt_dir/$exper_name.log \
--MIN_COUNT 2 "

screen -r $exper_name -X stuff "--LAST $ckpt_dir/last.pt \
--BEST $ckpt_dir/best.pt \
--SEED 666 \
--N_EPOCH 200 \
--N_BATCH 100 \
--N_WORKER 2 \
--LR 0.002 \
--BETAS 0.9 0.9 \
--EPS 1e-8 \
--LR_DECAY 0.75 \
--LR_ANNEAL 8000 \
--CLIP 5.0 "

screen -r $exper_name -X stuff "--IS_FIX_GLOVE \
--D_TAG 50 \
--N_RNN_LAYER 3 \
--D_RNN_HID 400 \
--D_ARC 500 \
--D_REL 100 \
--EMB_DROP 0.33 \
--RNN_DROP 0.33 \
--MLP_DROP 0.33 \n"

echo 'cmd finish!'