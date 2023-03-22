python run_flax_speech_recognition_seq2seq_streaming_dev.py \
            --model_name_or_path openai/whisper-tiny \
            --dataset_name mozilla-foundation/common_voice_11_0 \
            --dataset_config nn-NO \
	    --language Norwegian \
            --text_column_name sentence \
            --train_split_name test\
            --eval_split_name test\
            --output_dir ../../whisper-tiny-ft-dummy\
            --overwrite_output_dir\
            --max_eval_samples=16\
            --warmup_steps=8 \
            --do_train \
            --do_eval \
            --num_train_steps 30 \
            --eval_steps 5 \
            --learning_rate=2e-4 \
            --per_device_train_batch_size=2 \
            --per_device_eval_batch_size=2 \
            --predict_with_generate \
            --number_write_predictions 10 \
            --log_eval_predictions_fn log_predictions.write_predictions \
            --log_max_eval_predictions 100 \
            --streaming=True
