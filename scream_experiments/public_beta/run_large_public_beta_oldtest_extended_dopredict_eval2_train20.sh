export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../../run_nb_flax_speech_recognition_seq2seq_streaming_dev.py \
    --model_name_or_path openai/whisper-large-v2 \
    --run_name "Scream - deci_64bs_1e06lr" \
    --run_description "A Large Whisper Scream model - Dynamic Style Prompting. With bug fix for Fleurs." \
    --wandb_entity "nbailab" \
    --wandb_project "Scream - decimus" \
    --dataset_name NbAiLab/ncc_speech2 \
    --language Norwegian \
    --text_column_name text \
    --train_split_name train \
    --eval_split_name validation_fleurs \
    --test_split_name "test_fleurs,test_stortinget" \
    --output_dir ../../public_beta_large_oldtest_extended_dopredict_eval2 \
    --overwrite_output_dir\
    --warmup_steps 1000 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_steps 5000 \
    --lr_scheduler_type linear \
    --eval_steps 1000 \
    --learning_rate 1e-6 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate \
    --log_max_eval_predictions 100 \
    --log_eval_predictions_fn "utils.log_predictions.write_predictions" \
    --streaming True \
    --use_auth_token True \
    --dtype bfloat16 \
    --hub_private_repo True \
    --hub_model_id NbAiLab/public_beta_large_oldtest_extended_dopredict_eval2 \
    --resume_from_checkpoint True \
    --ignore_data_skip \
    --num_beams 5 \
    --overwrite_output_dir \
    --gradient_checkpointing True \
    --push_to_hub