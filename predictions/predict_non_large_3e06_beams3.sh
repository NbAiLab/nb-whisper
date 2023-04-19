export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
export TF_CPP_MIN_LOG_LEVEL=2
python ../run_flax_speech_recognition_seq2seq_streaming_dev.py \
        --model_name_or_path NbAiLab/scream_non_large_3e06_beams3 \
        --run_name "Scream - non_small_64pod_3e06_beam5" \
        --run_description "A Small Scream model. Trained with linear decay only on the all_v5 corpus. This version is trained with a learning rate of 6e6." \
        --wandb_entity "nbailab" \
        --wandb_project "Scream - septimus" \
        --dataset_name NbAiLab/NCC_speech_all_v5 \
        --language Norwegian \
        --text_column_name text \
        --eval_split_name validation \
        --do_train False\
        --do_eval False \
        --do_predict True \
        --predict_with_generate \
        --per_device_eval_batch_size 4 \
        --log_max_eval_predictions 100 \
        --log_eval_predictions_fn "log_predictions.write_predictions" \
        --streaming True \
        --use_auth_token True \
        --dtype bfloat16 \
        --output_dir output \
        --num_beams 3 \
        --push_to_hub False \

        