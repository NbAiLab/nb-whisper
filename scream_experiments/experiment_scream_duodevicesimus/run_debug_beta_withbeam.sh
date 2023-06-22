# Original gives worse scores - Added beam search - Apart from that identical to scream_small_beta
export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../../run_nb_flax_speech_recognition_seq2seq_streaming_dev.py \
        --model_name_or_path openai/whisper-small \
        --run_name "Scream - scream_duodevicesimus_beta_beam" \
        --run_description "A Small Scream model. Beta." \
        --wandb_entity "nbailab" \
        --wandb_project "Scream - beta" \
        --dataset_name NbAiLab/ncc_speech \
        --language Norwegian \
        --text_column_name text \
        --train_split_name train \
        --test_split_name "test_fleurs,test_stortinget,test_nrk_tv,test_audio_books" \
        --eval_split_name "validation_fleurs,validation_stortinget,validation_nrk_tv,validation_audio_books" \
        --dataset_load_fn "utils.dataset_load_scream.load_dataset_scream" \
        --output_dir ../../../scream_duodevicesimus_working \
        --overwrite_output_dir\
        --warmup_steps 2000 \
        --do_train \
        --do_eval \
        --do_predict \
        --num_train_steps 25000 \
        --lr_scheduler_type linear \
        --eval_steps 1000 \
        --learning_rate 5e-5 \
        --preprocessing_num_workers 32 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 8 \
        --predict_with_generate \
        --log_max_eval_predictions 100 \
        --log_eval_predictions_fn "utils.log_predictions.write_predictions" \
        --streaming True \
        --use_auth_token True \
        --dtype bfloat16 \
        --hub_private_repo True \
        --hub_model_id NbAiLab/scream_duodevicesimus_beta_beam \
        --resume_from_checkpoint True \
        --num_beams 5 \
        --ignore_data_skip \
        --gradient_checkpointing True \
        --log_examples 100 \
        --bpe_dropout 0.1 \
        --weight_decay 0.1 \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
	--timestamp_column_name "timestamped_text" \
        --prev_column_name "prompt" \
        --push_to_hub
        
        
