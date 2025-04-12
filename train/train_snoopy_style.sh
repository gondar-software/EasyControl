export MODEL_DIR="./huggingface/black-forest-labs/FLUX.1-dev" # your flux path
export OUTPUT_DIR="./models/style_model"  # your save path
export CONFIG="./snoopy_style_config.yaml"
export TRAIN_DATA="./datasets/dataset.jsonl"  # your data jsonl file
export LOG_PATH="$OUTPUT_DIR/log"

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --cond_size=512 \
    --noise_size=1024 \
    --subject_column="None" \
    --spatial_column="source" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 128 \
    --network_alphas 128 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "Snoopy style, Charming hand-drawn anime-style illustration" \
    --num_train_epochs=960 \
    --validation_steps=2000 \
    --checkpointing_steps=2000 \
    --pretrained_lora_path ./pretrained_lora_models/lora.safetensors \
    --spatial_test_images ./datasets/Testing/{1.jpg,2.jpg,3.jpg,4.jpg,5.jpg} \
    --subject_test_images None \
    --test_h 1024 \
    --test_w 1024 \
    --num_validation_images=1
# 
