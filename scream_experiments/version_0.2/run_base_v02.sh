# Environment variables
export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000

# Running the Python script
python ../../run_nb_flax_speech_recognition_seq2seq_streaming_dev.py \
    --model_name_or_path NbAiLab/nb-whisper-base-RC1 \
    --run_name "NB-Whisper - base - dynamic eval v0.2" \
    --run_description "A NB-Whisper dynamic eval v0.2" \
    --wandb_entity "nbailab" \
    --wandb_project "NB-Whisper dynamic v0.2" \
    --dataset_name NbAiLab/ncc_speech_styling_v2 \
    --language Norwegian \
    --text_column_name text \
    --train_split_name train \
    --dataset_load_fn "utils.dataset_load_rc_dynamic_noellipses_turndupwords.load_dataset_nbwhisper_rc_dynamic_eval_semantic" \
    --test_split_name "test_audio_books_no,test_norwegian_fleurs,test_nrk_no,test_nst,test_stortinget_no,test_clean_audio_books_no,test_clean_stortinget_no" \
    --eval_split_name "validation_nst,validation_clean_stortinget_no" \
    --hub_model_id NbAiLab/nb-whisper-base-v0.2 \
    --output_dir ../../../nb-whisper-base-v0.2 \
    --overwrite_output_dir \
    --do_train \
    --do_predict \
    --do_eval \
    --predict_with_generate \
    --warmup_steps 100 \
    --num_train_steps 1000 \
    --eval_steps 100 \
    --lr_scheduler_type linear \
    --learning_rate 5e-5 \
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
    --log_max_eval_predictions 100 \
    --log_eval_predictions_fn "utils.log_predictions.write_predictions" \
    --log_examples 100 \
    --streaming True \
    --use_auth_token True \
    --dtype bfloat16 \
    --hub_private_repo True \
    --resume_from_checkpoint True \
    --ignore_data_skip \
    --gradient_checkpointing True \
    --pad_target_to_multiple_of 448 \
    --max_prev_length 120 \
    --push_to_hub
