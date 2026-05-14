import os

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from torch import nn


def add_text_projector(model, config):
    model.bottleneck_dim = 1024
    hidden_size = config.hidden_size
    model.text_projector = nn.Sequential(
        nn.Linear(hidden_size, model.bottleneck_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(model.bottleneck_dim, model.bottleneck_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(model.bottleneck_dim, hidden_size),
        LlamaRMSNorm(hidden_size, eps=config.rms_norm_eps),
    ).to(model.device)
    return model


def init_text_projector(model):
    with torch.no_grad():
        nn.init.xavier_uniform_(model.text_projector[0].weight)
        nn.init.zeros_(model.text_projector[0].bias)
        nn.init.xavier_uniform_(model.text_projector[2].weight)
        nn.init.zeros_(model.text_projector[2].bias)
        nn.init.xavier_uniform_(model.text_projector[4].weight)
        nn.init.zeros_(model.text_projector[4].bias)


print("Loading base model...")
base_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

print("Adding text projector...")
base_model = add_text_projector(base_model, base_model.config)
print("Initializing text projector...")
init_text_projector(base_model)

print("Loading LoRA checkpoint...")
checkpoint_path = "/home/berrayan/planner/planner_executor_DiscreteDiffusion/models/dart-text-projector/checkpoint-100"
model = PeftModel.from_pretrained(base_model, checkpoint_path)

projector = None
if hasattr(model.base_model, "text_projector"):
    projector = model.base_model.text_projector
elif hasattr(model.base_model, "model") and hasattr(model.base_model.model, "text_projector"):
    projector = model.base_model.model.text_projector

if projector is None:
    raise RuntimeError("No text_projector found in checkpoint.")

output_path = "/home/berrayan/planner/planner_executor_DiscreteDiffusion/models/dart-text-projector/final_merged/text_projector.pth"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save(projector.state_dict(), output_path)
print(f"Saved text projector to: {output_path}")
