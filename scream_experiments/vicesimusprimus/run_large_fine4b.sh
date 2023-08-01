# Environment variables
export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000

# Running the Python script
python ../../run_nb_flax_speech_recognition_seq2seq_streaming_dev.py \
    --model_name_or_path NbAiLab/nb-whisper-large-beta \
    --run_name "Scream - large-fine4a-npsc-norm-nohes-beta" \
    --run_description "A Medium NB-Whisper Public Beta NPSC" \
    --wandb_entity "nbailab" \
    --wandb_project "NB-Whisper Public Beta" \
    --dataset_name NbAiLab/NPSC \
    --dataset_config_name "16K_mp3_bokmaal" \
    --language Norwegian \
    --text_column_name normsentence_text \
    --train_split_name train \
    --dataset_load_fn "utils.dataset_load_npsc.load_dataset_npsc_norm_nohes" \
    --test_split_name "test" \
    --eval_split_name "validation" \
    --hub_model_id NbAiLab/nb-whisper-large-fine4b-npsc-norm-nohes-beta \
    --output_dir ../../../nb-whisper-large-fine4b-npsc-norm-nohes-beta \
    --overwrite_output_dir \
    --do_train \
    --do_predict \
    --do_eval \
    --predict_with_generate \
    --warmup_steps 40 \
    --num_train_steps 200 \
    --eval_steps 20 \
    --max_eval_samples 2048 \
    --lr_scheduler_type linear \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --bpe_dropout 0.2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --preprocessing_num_workers 32 \
    --log_max_eval_predictions 100 \
    --log_eval_predictions_fn "utils.log_predictions.write_predictions" \
    --log_examples 100 \
    --streaming True \
    --use_auth_token True \
    --dtype bfloat16 \
    --hub_private_repo True \
    --resume_from_checkpoint True \
    --ignore_data_skip \
    --num_beams 5 \
    --gradient_checkpointing True \
    --push_to_hub_auto_lfs_prune True \
    --push_to_hub
