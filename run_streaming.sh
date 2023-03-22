export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python run_flax_speech_recognition_seq2seq_streaming_dev.py \
            --model_name_or_path openai/whisper-tiny \
            --dataset_name NbAiLab/NCC_speech_nrk_v4 \
	        --language Norwegian \
            --text_column_name text \
            --train_split_name validation\
            --eval_split_name test\
            --output_dir ../../whisper_dummy\
            --overwrite_output_dir\
            --max_eval_samples=20\
            --warmup_steps=8 \
            --do_train \
            --do_eval \
            --num_train_steps 30 \
            --eval_steps 5 \
            --learning_rate=1e-5 \
            --preprocessing_num_workers 10 \
            --per_device_train_batch_size=2 \
            --per_device_eval_batch_size=2 \
            --streaming True \
            --hub_private_repo True \
            --hub_model_id NbAiLab/whisper_dummy \
            --use_auth_token True \
            --dtype bfloat16 \
            --predict_with_generate \
            --log_eval_predictions_fn log_predictions.write_predictions \
            --log_max_eval_predictions 100 \
            --streaming=True \
            --push_to_hub
            
