#!/bin/bash

# Train both text and latent projectors simultaneously
# This script runs the multimodal training that updates both projectors

set -e

cd "$(dirname "$0")"

# Activate environment (optional; adjust path as needed)
# source ../env/bin/activate

# Run training with multimodal flag
python pts/train/train.py \
    --answer_model_id "meta-llama/Llama-3.2-3B-Instruct" \
    --llada_model_id "GSAI-ML/LLaDA-8B-Instruct" \
    --per_dataset_train_samples 5000 \
    --max_test_samples 700 \
    --max_length 512 \
    --draft_plan_max_new_tokens 96 \
    --answer_max_new_tokens 64 \
    --output_dir "models/dart-multimodal-projector" \
    --run_name "llada-multimodal-projector-llama" \
    --seed 42 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --multimodal

echo "Training complete! Both text_projector and latent_projector saved to models/dart-multimodal-projector/"
