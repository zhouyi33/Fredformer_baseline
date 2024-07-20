export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

seq_len=168
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.001
d_model=16
d_ff=32
batch_size=16
train_epochs=2
patience=10

python -u train.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/global \
  --data_path temp.npy \
  --model_id temp_168_24 \
  --model $model_name \
  --data Meteorology \
  --features MS \
  --seq_len $seq_len \
  --label_len 1 \
  --pred_len 24 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 37 \
  --dec_in 37 \
  --c_out 37 \
  --des 'temp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 512 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window