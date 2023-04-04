export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../run_flax_speech_recognition_seq2seq_streaming_dev.py \
        --model_name_or_path openai/whisper-base \
        --run_name "ScreamBase - exp_test_base_2" \
        --run_description "A Large Scream model with 8*4*8=256 seq length. Trained with linear decay only on the all_v5 corpus. This version is trained with a learning rate of 2e6." \
        --wandb_entity "nbailab" \
        --wandb_project "Scream - septimus" \
        --dataset_name NbAiLab/NCC_speech_all_v5 \
        --language Norwegian \
        --text_column_name text \
        --train_split_name train \
        --eval_split_name validation \
        --output_dir ../../scream_base_test2\
        --overwrite_output_dir\
        --warmup_steps 5000 \
        --do_train \
        --do_eval \
        --num_train_steps 100000 \
        --lr_scheduler_type linear \
        --eval_steps 2500 \
        --learning_rate 2e-6 \
        --preprocessing_num_workers 8 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 8 \
        --predict_with_generate \
        --log_max_eval_predictions 100 \
        --log_eval_predictions_fn "log_predictions.write_predictions" \
        --streaming True \
        --use_auth_token True \
        --dtype bfloat16 \
        --hub_private_repo True \
        --hub_model_id NbAiLab/scream_base_test2 \
        --resume_from_checkpoint True \
	--ignore_data_skip \
        --push_to_hub
