export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../run_flax_speech_recognition_seq2seq_streaming_debug.py \
    --model_name_or_path openai/whisper-large-v2 \
    --run_name "ScreamLarge - debug_beam5_long" \
    --run_description "A Large Whisper Scream model with 5 batch size. Trained with 5e-6 and linear decay on the all_v5-corpus." \
    --wandb_entity "nbailab" \
    --wandb_project "Scream - septimus" \
    --dataset_name NbAiLab/NCC_speech_all_v5 \
    --language Norwegian \
    --text_column_name text \
    --train_split_name train\
    --eval_split_name validation\
    --output_dir ../../scream_large_oct_debug\
    --overwrite_output_dir\
    --warmup_steps 10000 \
    --do_train \
    --do_eval \
    --num_train_steps 50000 \
    --lr_scheduler_type linear \
    --eval_steps 10000 \
    --learning_rate 5e-6 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate \
    --log_max_eval_predictions 100 \
    --log_eval_predictions_fn "log_predictions.write_predictions" \
    --streaming True \
    --use_auth_token True \
    --dtype bfloat16 \
    --hub_private_repo True \
    --hub_model_id NbAiLab/scream_large_oct_debug \
    --resume_from_checkpoint True \
    --num_beams 5 \
    --ignore_data_skip \
    --push_to_hub
