# Environment variables
export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000

# Running the Python script
python ../../run_nb_flax_speech_recognition_seq2seq_streaming_dev.py \
    --model_name_or_path openai/whisper-large-v2 \
    --run_name "Scream - large_public_beta 100k" \
    --run_description "A Large NB-Whisper Public Beta" \
    --wandb_entity "nbailab" \
    --wandb_project "NB-Whisper Public Beta" \
    --dataset_name NbAiLab/ncc_speech3 \
    --language Norwegian \
    --text_column_name text \
    --train_split_name train \
    --dataset_load_fn "utils.dataset_load_public_beta.load_dataset_scream" \
    --test_split_name "test_fleurs,test_stortinget" \
    --eval_split_name "validation_fleurs,validation_stortinget" \
    --hub_model_id NbAiLab/nb-whisper-large-publicbeta-speech3-100k \
    --output_dir ../../../nb-whisper-large-publicbeta-speech3-100k \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --warmup_steps 5000 \
    --num_train_steps 100000 \
    --eval_steps 2500 \
    --lr_scheduler_type linear \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --bpe_dropout 0.1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --preprocessing_num_workers 32 \
    --timestamp_column_name "timestamped_text" \
    --prev_column_name "prompt" \
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
    --push_to_hub_auto_lfs_prune True \
    --push_to_hub
