export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../../run_nb_flax_speech_recognition_seq2seq_streaming_dev.py \
        --model_name_or_path openai/whisper-large-v2 \
        --run_name "Scream - scream_large_beta" \
        --run_description "A Large Scream model. Beta." \
        --wandb_entity "nbailab" \
        --wandb_project "Scream - beta" \
        --dataset_name NbAiLab/ncc_speech \
        --language Norwegian \
        --text_column_name text \
        --train_split_name train \
        --dataset_load_fn "utils.dataset_load_scream_nopretext.load_dataset_scream" \
        --eval_split_name "validation_fleurs,validation_stortinget,validation_nrk_tv" \
        --output_dir ../../../scream_large_beta \
        --overwrite_output_dir\
        --warmup_steps 2500 \
        --do_train \
        --do_eval \
        --num_train_steps 25000 \
        --lr_scheduler_type linear \
        --eval_steps 1000 \
        --learning_rate 2e-5 \
        --preprocessing_num_workers 32 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 8 \
        --predict_with_generate \
        --log_max_eval_predictions 100 \
        --log_eval_predictions_fn "utils.log_predictions.write_predictions" \
        --streaming True \
        --use_auth_token True \
        --dtype bfloat16 \
        --hub_private_repo True \
        --hub_model_id NbAiLab/scream_large_beta \
        --resume_from_checkpoint True \
        --ignore_data_skip \
        --gradient_checkpointing True \
        --log_examples 100 \
        --bpe_dropout 0.1 \
        --weight_decay 0.1 \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
	--timestamp_column_name "timestamped_text" \
        --prev_column_name "previous_text" \
        --push_to_hub
        
        
