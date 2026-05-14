#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate project venv if present.
if [[ -f "llada_env/bin/activate" ]]; then
	source llada_env/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Run the plan-conditioned LoRA distillation trainer.
python pts/train/train.py "$@"


