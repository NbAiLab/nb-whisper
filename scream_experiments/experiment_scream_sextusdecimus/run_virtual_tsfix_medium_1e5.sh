export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../../run_nb_flax_speech_recognition_seq2seq_streaming_dev.py \
        --model_name_or_path openai/whisper-medium \
        --run_name "Scream - scream_sextusdecimus_virtual_tsfix_medium" \
        --run_description "A Medium Scream model. Labels" \
        --wandb_entity "nbailab" \
        --wandb_project "Scream - duodecimus" \
        --dataset_name NbAiLab/ncc_speech \
        --language Norwegian \
        --text_column_name text \
        --train_split_name train \
        --dataset_load_fn "utils.dataset_load_scream.load_dataset_scream" \
        --eval_split_name validation \
        --output_dir ../../../scream_sextusdecimus_virtual_tsfix_medium_1e5\
        --overwrite_output_dir\
        --warmup_steps 2000 \
        --do_train \
        --do_eval \
        --num_train_steps 20000 \
        --lr_scheduler_type linear \
        --eval_steps 1000 \
        --learning_rate 1e-5 \
        --preprocessing_num_workers 32 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --predict_with_generate \
        --log_max_eval_predictions 50 \
        --log_eval_predictions_fn "utils.log_predictions.write_predictions" \
        --streaming True \
        --use_auth_token True \
        --dtype bfloat16 \
        --hub_private_repo True \
        --hub_model_id NbAiLab/scream_sextusdecimus_virtual_tsfix_medium_1e5 \
        --resume_from_checkpoint True \
        --num_beams 5 \
        --ignore_data_skip \
        --gradient_checkpointing True \
        --prev_column_name "prompt" \
        --log_examples 100 \
        --bpe_dropout 0.1 \
        --activation_dropout 0.1 \
        --push_to_hub
        
