#!/bin/bash

#PBS -P hn98
#PBS -q gpuvolta
#PBS -l walltime=20:00:00
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=48GB

module load python3/3.9.2

module list

cd /scratch/hn98/ap8021/DIEF

source ./myvenv/bin/activate
pip3 list

pwd
nvidia-smi
lscpu

cd /scratch/hn98/ap8021/DIEF/thuml_tslib_dief/

python3 ./run_DIEF.py \
    --is_training 1 \
    --model_id DIEF \
    --enc_in 4 \
    --c_out 240 \
    --model PatchTST \
    --e_layers 3 \
    --batch_size 2058 \
    --d_model 128 \
    --d_ff 256 \
    --top_k 3 \
    --des Exp \
    --learning_rate 0.01 \
    --pos_weight 30 \
    --patience 30 \
    --root_path '/scratch/hn98/ap8021/DIEF/dataset/' \
    --c_out 240 \
    --exp_folder '/scratch/hn98/ap8021/DIEF/e15_PatchTST/' \
    --seq_len 336 \
    --train_epochs 100 \
    --seed $s


