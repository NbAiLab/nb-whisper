export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../run_flax_speech_recognition_seq2seq_streaming_dev.py \
        --model_name_or_path openai/whisper-tiny \
        --run_name "ScreamTiny - exp_sept_tpu32_nrk" \
        --run_description "A Tiny Whisper Scream model with 32*4*4=512 seq length. Trained with linear decay only on nrk_v5 corpus." \
        --wandb_entity "nbailab" \
        --wandb_project "Scream - septimus" \
        --dataset_name NbAiLab/NCC_speech_nrk_v5 \
        --language Norwegian \
        --text_column_name text \
        --train_split_name train \
        --eval_split_name validation \
        --output_dir ../../scream_tiny_sept_nrk\
        --overwrite_output_dir\
        --warmup_steps 10000 \
        --do_train \
        --do_eval \
        --num_train_steps 250000 \
        --lr_scheduler_type linear \
        --eval_steps 5000 \
        --learning_rate 2e-5 \
        --preprocessing_num_workers 64 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --predict_with_generate \
        --log_max_eval_predictions 100 \
        --log_eval_predictions_fn "log_predictions.write_predictions" \
        --streaming True \
        --use_auth_token True \
        --dtype bfloat16 \
        --hub_private_repo True \
        --hub_model_id NbAiLab/scream_tiny_sept_nrk \
        --resume_from_checkpoint \
        --push_to_hub
