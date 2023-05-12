export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../run_nb_flax_speech_recognition_seq2seq_streaming_test.py \
    --model_name_or_path NbAiLab/scream_non_large_1e06_beams5_constantlr_long \
    --run_name "Scream - non_large_1e06_verbosity3" \
    --run_description "A Large Whisper Scream model with 5 batch size. Trained with 1e-6 and linear decay on the all_v5-corpus." \
    --wandb_entity "nbailab" \
    --wandb_project "Scream - septimus" \
    --dataset_name NbAiLab/NCC_speech_all_v5 \
    --language Norwegian \
    --text_column_name text \
    --train_split_name train\
    --eval_split_name validation\
    --output_dir ../../scream_non_large_debug4\
    --overwrite_output_dir\
    --warmup_steps 2500 \
    --do_train \
    --do_eval \
    --num_train_steps 25000 \
    --lr_scheduler_type linear \
    --eval_steps 2500 \
    --learning_rate 1e-6 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --log_max_eval_predictions 100 \
    --log_eval_predictions_fn "log_predictions.write_predictions" \
    --streaming True \
    --use_auth_token True \
    --dtype bfloat16 \
    --hub_private_repo True \
    --hub_model_id NbAiLab/scream_non_large_debug4 \
    --resume_from_checkpoint True \
    --ignore_data_skip \
    --num_beams 5 \
    --push_to_hub
