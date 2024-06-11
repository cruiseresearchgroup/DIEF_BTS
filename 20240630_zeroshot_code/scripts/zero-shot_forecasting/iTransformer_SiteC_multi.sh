#!/bin/bash

#PBS -q gpuvolta
#PBS -l walltime=47:59:00
#PBS -l ngpus=4
#PBS -l ncpus=48
#PBS -l mem=150GB
#PBS -l jobfs=200GB
#PBS -P po67
#PBS -l storage=scratch/po67+gdata/po67+gdata/hn98
#PBS -M dawn.lin@student.unsw.edu.au
#PBS -m b
#PBS -m e

source ~/.bashrc

condapbs_ex timesnet

cd /scratch/po67/dl8829/DIEF_forecasting
pwd
nvidia-smi
lscpu

conda list

export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=iTransformer

  python -u /scratch/po67/dl8829/DIEF_forecasting/run.py \
  --task_name zero_shot_forecast \
  --is_training 1 \
  --use_gpu True \
  --use_multi_gpu \
  --root_path /scratch/po67/dl8829/DIEF_forecasting/dataset/ \
  --data_path SiteC_July22toAug22.csv \
  --model_id SiteC_iTransformer \
  --model $model_name \
  --data SiteC \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --features S \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 512 \
  --d_model 512 \
  --des 'TrnOnSiteC' \
  --itr 1 \
  --learning_rate 0.01 \
  --train_epochs 20 \
  --loss 'MSE' 


python -u /scratch/po67/dl8829/DIEF_forecasting/run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --use_gpu True \
  --use_multi_gpu \
  --root_path /scratch/po67/dl8829/DIEF_forecasting/dataset/ \
  --data_path SiteA_July22toAug22.csv \
  --model_id SiteC_iTransformer \
  --model $model_name \
  --data SiteA \
  --checkpoint /scratch/po67/dl8829/DIEF_forecasting/checkpoints/iTransformer_TrnOnSiteC/checkpoint.pth \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --features S \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 512 \
  --d_model 512 \
  --des 'TstC2A' \
  --itr 1 \
  --learning_rate 0.01 \
  --train_epochs 20 \
  --loss 'MSE' 

python -u /scratch/po67/dl8829/DIEF_forecasting/run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --use_gpu True \
  --use_multi_gpu \
  --root_path /scratch/po67/dl8829/DIEF_forecasting/dataset/ \
  --data_path SiteB_July22toAug22.csv \
  --model_id SiteC_iTransformer \
  --model $model_name \
  --data SiteB \
  --checkpoint /scratch/po67/dl8829/DIEF_forecasting/checkpoints/iTransformer_TrnOnSiteC/checkpoint.pth \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --features S \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 512 \
  --d_model 512 \
  --des 'TstC2B' \
  --itr 1 \
  --learning_rate 0.01 \
  --train_epochs 20 \
  --loss 'MSE' 


python -u /scratch/po67/dl8829/DIEF_forecasting/run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --use_gpu True \
  --use_multi_gpu \
  --root_path /scratch/po67/dl8829/DIEF_forecasting/dataset/ \
  --data_path SiteC_July22toAug22.csv \
  --model_id SiteC_iTransformer \
  --model $model_name \
  --data SiteC \
  --checkpoint /scratch/po67/dl8829/DIEF_forecasting/checkpoints/iTransformer_TrnOnSiteC/checkpoint.pth \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --features S \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 512 \
  --d_model 512 \
  --des 'TstC2C' \
  --itr 1 \
  --learning_rate 0.01 \
  --train_epochs 20 \
  --loss 'MSE' 