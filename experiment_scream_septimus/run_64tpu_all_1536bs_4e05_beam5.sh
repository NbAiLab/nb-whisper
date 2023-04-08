export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../run_flax_speech_recognition_seq2seq_streaming.py \
        --model_name_or_path openai/whisper-tiny \
        --run_name "ScreamTiny - exp_sept_tpu64_all_1536bs_4e05_beam5" \
        --run_description "A Tiny Whisper Scream model with 96*4*4=1536 seq length. Trained with linear decay on the all_v5-corpus." \
        --wandb_entity "nbailab" \
        --wandb_project "Scream - septimus" \
        --dataset_name NbAiLab/NCC_speech_all_v5 \
        --language Norwegian \
        --text_column_name text \
        --train_split_name train \
        --eval_split_name validation \
        --output_dir ../../scream_tiny_sept_all_1536bs_4e05_64_beam5\
        --overwrite_output_dir\
        --warmup_steps 5000 \
        --do_train \
        --do_eval \
        --num_train_steps 50000 \
        --lr_scheduler_type linear \
        --eval_steps 2500 \
        --learning_rate 4e-5 \
        --preprocessing_num_workers 64 \
        --per_device_train_batch_size 48 \
        --per_device_eval_batch_size 16 \
        --predict_with_generate \
        --log_max_eval_predictions 100 \
        --log_eval_predictions_fn "log_predictions.write_predictions" \
        --streaming True \
        --use_auth_token True \
        --dtype bfloat16 \
        --hub_private_repo True \
        --hub_model_id NbAiLab/scream_tiny_sept_all_1536bs_4e05_64_beam5 \
        --resume_from_checkpoint True \
        --num_beams 5 \
	--push_to_hub
