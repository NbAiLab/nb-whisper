export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../run_nb_flax_speech_recognition_seq2seq_streaming_pere.py \
        --model_name_or_path openai/whisper-small \
        --run_name "Scream - prompt test v1 streaming" \
        --run_description "A Small Scream model. Prompt test 1 - streaming." \
        --wandb_entity "nbailab" \
        --wandb_project "Scream - septimus" \
        --dataset_name NbAiLab/NCC_speech_all_v5 \
        --language Norwegian \
        --text_column_name text \
        --train_split_name train \
        --eval_split_name validation \
        --output_dir ../../scream_prompt_test_v1_streaming\
        --overwrite_output_dir\
        --warmup_steps 1000 \
        --do_train \
        --do_eval \
        --num_train_steps 20000 \
        --lr_scheduler_type linear \
        --eval_steps 500 \
        --learning_rate 9e-6 \
        --preprocessing_num_workers 32 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --predict_with_generate \
        --log_max_eval_predictions 100 \
        --log_eval_predictions_fn "log_predictions.write_predictions" \
        --streaming True \
        --use_auth_token True \
        --dtype bfloat16 \
        --hub_private_repo True \
        --hub_model_id NbAiLab/scream_prompt_test_v1_streaming \
        --resume_from_checkpoint True \
        --num_beams 5 \
        --ignore_data_skip \
        --gradient_checkpointing True \
        --push_to_hub
