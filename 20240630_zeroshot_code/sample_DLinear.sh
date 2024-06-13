#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=DLinear

python -u ./run.py \
  --task_name zero_shot_forecast \
  --is_training 1 \
  --use_gpu True \
  --use_multi_gpu \
  --root_path ./dataset/ \
  --data_path TODO: add SiteA data path here... \
  --model_id SiteA_DLinear \
  --model $model_name \
  --data SiteA \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 133 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 512 \
  --d_model 512 \
  --des 'TrnOnSiteA' \
  --itr 1 \
  --learning_rate 0.01 \
  --train_epochs 20 \
  --loss 'MSE' 

echo "Training finished. Start testing on SiteA..."

python -u ./run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --use_gpu True \
  --use_multi_gpu \
  --root_path ./dataset/ \
  --data_path TODO: add SiteA data path here... \
  --model_id SiteA_DLinear \
  --model $model_name \
  --data SiteA \
  --checkpoint ./checkpoints/DLinear_TrnOnSiteA/checkpoint.pth \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 133 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 512 \
  --d_model 512 \
  --des 'TstA2A' \
  --itr 1 \
  --learning_rate 0.01 \
  --train_epochs 20 \
  --loss 'MSE' 


echo "Start testing on SiteB..."


python -u ./run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --use_gpu True \
  --use_multi_gpu \
  --root_path ./dataset/ \
  --data_path TODO: add SiteB data path here... \
  --model_id SiteA_DLinear \
  --model $model_name \
  --data SiteB \
  --checkpoint ./checkpoints/DLinear_TrnOnSiteA/checkpoint.pth \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 133 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 512 \
  --d_model 512 \
  --des 'TstA2B' \
  --itr 1 \
  --learning_rate 0.01 \
  --train_epochs 20 \
  --loss 'MSE' 

echo "Start testing on SiteC..."

python -u ./run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --use_gpu True \
  --use_multi_gpu \
  --root_path ./dataset/ \
  --data_path TODO: add SiteC data path here... \
  --model_id SiteA_DLinear \
  --model $model_name \
  --data SiteC \
  --checkpoint ./checkpoints/DLinear_TrnOnSiteA/checkpoint.pth \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 133 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 512 \
  --d_model 512 \
  --des 'TstA2C' \
  --itr 1 \
  --learning_rate 0.01 \
  --train_epochs 20 \
  --loss 'MSE' 
