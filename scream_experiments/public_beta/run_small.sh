# Environment variables
export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000

# Running the Python script
python ../../run_nb_flax_speech_recognition_seq2seq_streaming_dev.py \
    # Model details
    --model_name_or_path openai/whisper-small \
    --run_name "Scream - small_public_beta" \
    --run_description "A Small NB-Whisper Public Beta" \

    # Weights and Biases (W&B) settings
    --wandb_entity "nbailab" \
    --wandb_project "NB-Whisper Public Beta" \
    
    # Dataset settings
    --dataset_name NbAiLab/ncc_speech \
    --language Norwegian \
    --text_column_name text \
    --train_split_name train \
    --dataset_load_fn "utils.dataset_load_public_beta.load_dataset_scream" \
    --test_split_name "test_fleurs,test_stortinget,test_nrk_tv,test_audio_books" \
    --eval_split_name "validation_fleurs,validation_stortinget,validation_nrk_tv,validation_audio_books" \
    
    # Output settings
    --hub_model_id NbAiLab/nb-whisper_public_beta_small_testrun \
    --output_dir ../../../nb-whisper_public_beta_small_testrun \
    --overwrite_output_dir \

    #Action
    --do_train \
    --do_test \
    --do_eval \
    --predict_with_generate \

    # Number of Steps
    --warmup_steps 1000 \
    --num_train_steps 10000 \
    --eval_steps 1000 \

    # Learning rate and optimization settings
    --lr_scheduler_type linear \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --bpe_dropout 0.1 \
    --activation_dropout 0.1 \

    # Batch size
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \

    # Data preprocessing settings
    --preprocessing_num_workers 32 \
    --timestamp_column_name "timestamped_text" \
    --prev_column_name "previous_text" \

    # Logging and prediction settings
    --log_max_eval_predictions 100 \
    --log_eval_predictions_fn "utils.log_predictions.write_predictions" \
    --log_examples 100 \

    # Misc settings
    --streaming True \
    --use_auth_token True \
    --dtype bfloat16 \
    --num_beams 5 \
    --hub_private_repo True \
    --resume_from_checkpoint True \
    --ignore_data_skip \
    --gradient_checkpointing True \
    --push_to_hub


