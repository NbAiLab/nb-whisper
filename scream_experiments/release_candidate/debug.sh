# Environment variables
export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000

# Running the Python script
python ../../run_nb_flax_speech_recognition_seq2seq_streaming_debug.py \
    --model_name_or_path NbAiLab/nb-whisper-small-RC1 \
    --run_name "Debug" \
    --run_description "A tiny NB-Whisper RC1" \
    --wandb_entity "nbailab" \
    --wandb_project "NB-Debug" \
    --dataset_name NbAiLab/ncc_speech_v7 \
    --language Norwegian \
    --text_column_name text \
    --train_split_name train \
    --dataset_load_fn "utils.dataset_load_rc_debug.load_dataset_nbwhisper" \
    --test_split_name "test_nst" \
    --eval_split_name "validation_norwegian_fleurs,validation_nst" \
    --hub_model_id NbAiLab/nb-whisper-debug2 \
    --output_dir ../../../nb-whisper-debug2 \
    --overwrite_output_dir \
    --do_train \
    --warmup_steps 100 \
    --num_train_steps 200000 \
    --eval_steps 10 \
    --lr_scheduler_type linear \
    --learning_rate 1.5e-4 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --bpe_dropout 0.2 \
    --activation_dropout 0.1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --preprocessing_num_workers 32 \
    --timestamp_column_name "timestamped_text" \
    --prev_column_name "previous_text" \
    --log_max_eval_predictions 50 \
    --log_eval_predictions_fn "utils.log_predictions.write_predictions" \
    --log_examples 1 \
    --streaming True \
    --use_auth_token True \
    --dtype bfloat16 \
    --hub_private_repo True \
    --resume_from_checkpoint True \
    --ignore_data_skip \
    --gradient_checkpointing True \
    --push_to_hub
