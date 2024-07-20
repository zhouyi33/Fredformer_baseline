export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

python -u train.py \
  --root_path ./dataset/global \
  --data_path wind.npy \
  --model_id v1 \
  --model $model_name \
  --data Meteorology \
  --features MS \
  --seq_len 168 \
  --label_len 1 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 37 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --des 'global_wind' \
  --learning_rate 0.0005 \
  --batch_size 8192 \
  --train_epochs 3