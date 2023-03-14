export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../run_flax_speech_recognition_seq2seq_streaming.py \
            --model_name_or_path openai/whisper-tiny \
            --dataset_name NbAiLab/NCC_S3_NRK_SMALL \
	    --language Norwegian \
            --text_column_name text \
            --train_split_name train\
            --eval_split_name test\
            --output_dir ../../scream_nrk_gpu_stream_tiny\
            --overwrite_output_dir\
            --warmup_steps 100 \
            --do_train \
            --do_eval \
            --num_train_steps 585 \
	    --lr_scheduler_type linear \
            --eval_steps 100 \
            --learning_rate 0.75e-3 \
            --per_device_train_batch_size 256 \
            --per_device_eval_batch_size 12 \
            --predict_with_generate \
            --number_write_predictions 100 \
            --streaming True \
            --hub_private_repo True \
            --hub_model_id NbAiLab/scream_nrk_gpu_stream_tiny \
            --use_auth_token True \
            --dtype bfloat16 \
            --push_to_hub
