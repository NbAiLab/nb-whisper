export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../run_flax_speech_recognition_seq2seq_streaming.py \
            --model_name_or_path openai/whisper-tiny \
            --dataset_name NbAiLab/NCC_speech_nrk_v4 \
	    --language Norwegian \
            --text_column_name text \
            --train_split_name train\
            --eval_split_name test\
            --output_dir ../../scream_tertius_exp11_constant_5e6\
            --overwrite_output_dir\
            --warmup_steps 1000 \
            --do_train \
            --do_eval \
            --num_train_steps 100000 \
	    --lr_scheduler_type constant_with_warmup \
            --eval_steps 500 \
            --learning_rate 5e-6 \
	    --preprocessing_num_workers 50 \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64 \
            --predict_with_generate \
            --number_write_predictions 100 \
            --streaming True \
            --hub_private_repo True \
            --hub_model_id NbAiLab/scream_tertius_exp11_constant_5e6 \
            --use_auth_token True \
            --dtype bfloat16 \
            --push_to_hub
