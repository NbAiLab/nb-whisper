export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../run_flax_speech_recognition_seq2seq_streaming.py \
            --model_name_or_path openai/whisper-small \
            --dataset_name NbAiLab/NCC_S3_nrk \
	        --language Norwegian \
            --text_column_name text \
            --train_split_name train\
            --eval_split_name test\
            --output_dir ../../scream_prime_e5_ncc3nrk_linearlr_small\
            --overwrite_output_dir\
            --warmup_steps 500 \
            --do_train \
            --do_eval \
            --num_train_steps 10000 \
            --lr_scheduler_type linear \
            --eval_steps 500 \
            --learning_rate 5e-4 \
            --per_device_train_batch_size 20 \
            --per_device_eval_batch_size 4 \
            --predict_with_generate \
            --number_write_predictions 100 \
            --streaming True \
            --hub_private_repo True \
            --hub_model_id NbAiLab/scream_prime_e5_ncc3nrk_linearlr_small \
            --use_auth_token True \
            --dtype bfloat16 \
            --push_to_hub