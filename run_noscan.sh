export PYTHONBREAKPOINT="ipdb.set_trace"
export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python run_nb_flax_speech_recognition_seq2seq_streaming_dev.py \
            --model_name_or_path openai/whisper-tiny \
            --dataset_name NbAiLab/ncc_speech \
            --dataset_load_fn utils.dataset_load_scream.load_dataset_scream \
            --language Norwegian \
            --text_column_name text \
            --prev_column_name previous_text \
            --timestamp_column_name  timestamped_text \
            --train_split_name train \
            --eval_split_name validation \
            --output_dir ../whisper_scan \
            --overwrite_output_dir\
            --warmup_steps=200 \
            --do_train \
            --do_eval \
            --do_predict \
            --num_train_steps 2000 \
            --eval_steps 200 \
            --learning_rate=1e-5 \
            --preprocessing_num_workers 50 \
            --per_device_train_batch_size=16 \
            --per_device_eval_batch_size=4 \
            --streaming True \
            --gradient_checkpointing True \
            --hub_private_repo True \
            --hub_model_id NbAiLab/whisper_dummy6 \
            --use_auth_token True \
            --dtype bfloat16 \
            --predict_with_generate \
            --log_eval_predictions_fn utils.log_predictions.write_predictions \
            --log_max_eval_predictions 100 \
            --streaming=True \
            --push_to_hub
