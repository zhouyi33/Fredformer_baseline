export CUDA_VISIBLE_DEVICES=0

model_name=Fredformer

python -u train_Fredformer.py \
  --root_path ./dataset/global \
  --data_path temp.npy \
  --model_id v1 \
  --model $model_name \
  --data Fredformer \
  --features MS \
  --seq_len 168 \
  --label_len 1 \
  --pred_len 24 \
  --e_layers 3 \
  --enc_in 37 \
  --d_model 8 \
  --d_ff 256 \
  --n_heads 16 \
  --des 'global_temp' \
  --learning_rate 0.001 \
  --batch_size 128 \
  --train_epochs 2
