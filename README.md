# Planner

A unified workspace for training and evaluating multimodal projectors that bridge diffusion models, language models, and their latent representations.

## Repository Structure

- **DARM/** — Diffusion-based Adapter with Reasoning Module. Trains a latent projector that maps diffusion model hidden states to LLM embeddings.
- **DART/** — Diffusion-Aided Reasoning Text-projector. Trains a text-based projector that projects LLaDA draft plan embeddings into LLM space. Now includes **multimodal training** capability.
- **LLaDA/** — LLaDA model integration and evaluation utilities.

## Quick Start: Train Both Projectors

To train the text-based projector and latent-based projector simultaneously:

```bash
cd DART
./train_both.sh
```

This runs `MultiModalTrainer`, which:
1. Loads LLaDA to generate draft plans (text pathway)
2. Obtains final hidden states from LLaDA's denoising pass (latent pathway)
3. Projects both through separate projectors (`text_projector` and `latent_projector`) into Llama embeddings
4. Computes combined loss and optimizes both projectors jointly
5. Saves checkpoints for both projectors to `models/dart-multimodal-projector/`

### Configuration

Edit `DART/train_both.sh` to customize:
- Model IDs, batch size, learning rate, epochs
- LLaDA generation parameters (steps, gen_length, block_length, temperature, cfg_scale)
- Output directory and run name

## Training Options

### Text-only Projector (original)
```bash
cd DART
python pts/train/train.py  # no --multimodal flag
```

### Both Projectors (new)
```bash
cd DART
./train_both.sh
# or manually:
python pts/train/train.py --multimodal
```

## Architecture Overview

### DART Multimodal Trainer

```
Input Question
    ↓
LLaDA Model
    ├─ Text Output (draft_plan) ──→ [text_projector] ──→ Llama
    └─ Hidden States (last layer) ──→ [latent_projector] ──→ Llama
    ↓
Combined Loss (text_loss + λ * latent_loss)
    ↓
Backprop & Update both projectors
```

## Key Files

- `DART/train_both.sh` — Bash script for convenient multimodal training
- `DART/pts/train/train.py` — Main training script; contains `PlanProjectorTrainer`, `MultiModalProjectorTrainer`, and `MultiModalTrainer`
- `DARM/pts/train/generate_latents.py` — Denoising generator used by latent pathway (dynamically imported)

## Output

After training, checkpoints are saved to `models/dart-multimodal-projector/`:
- `final_merged/text_projector.pth` — Trained text embedding projector
- `final_merged/latent_projector.pth` — Trained latent-to-embedding projector
- `final_merged/` — Tokenizer and metadata
- `checkpoint-*/` — Intermediate checkpoints

## References

- **DART** uses LLaDA for planning and text generation
- **DARM** contributes the denoising generation routine for obtaining final latent representations
- **LLaDA** provides the base instruction-tuned language model

## License

See individual project READMEs.
