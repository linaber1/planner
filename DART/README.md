# DART

DART now supports a plan-refinement training pipeline implemented in `pts/train/train.py`.

## Training Pipeline (Notebook-Aligned)

`pts/train/train.py` trains a `text_projector` with this flow:

1. Build a uniform mixed training set from 7 datasets:
	 ARC-Easy, ARC-Challenge, DART-1, DART-2, DART-3, DART-4, DART-5.
2. Sample `5,000` examples per dataset by default (`35,000` total).
3. Generate draft plans from question-only prompts using LLaDA.
4. Project the draft-plan token embeddings using `model.text_projector`.
5. Train against final-answer tokens with token-level cross-entropy
	 (prompt labels are masked with `-100`) using Qwen 2.5-0.5B.

By default Qwen is frozen and only `text_projector` is trained.
Use `--lightly_tune_qwen` to also tune a small subset of Qwen weights.

## Scripts

- `train.sh`: runs `pts/train/train.py` (passes through extra CLI args).
- `generate_latents.sh`: legacy compatibility script for old plan generation flow.
- `eval_ours.sh`: legacy dual pipeline evaluation script.

## Quick Start

```bash
./train.sh \
	--per_dataset_train_samples 5000 \
	--max_test_samples 700
```

To reduce memory usage:

```bash
./train.sh \
	--per_dataset_train_samples 1000 \
	--max_test_samples 140 \
	--learning_rate 1e-4
```