import argparse
import gc
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def free_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def split_gsm8k_answer(answer_text: str) -> tuple[str, str]:
    parts = answer_text.split("####")
    final_ans = parts[-1].strip() if len(parts) > 1 else answer_text.strip()
    final_ans = final_ans.replace(",", "")
    return answer_text, final_ans


ARC_QUESTION_PROMPT_TEMPLATE = """Question: {question}\n{choices_text}"""
ARC_QUESTION_POSTFIX = (
    "\nAnswer with a single letter (A, B, C, or D) and no explanation. "
    "Your answer should start with \"Answer: \" and be followed by the letter "
    "of the answer you choose. Do not include any other text in your response."
)


def normalize_answer_text(answer) -> str:
    if answer is None:
        return ""
    return str(answer).strip().replace(",", "")


def prepare_arc_sample(item: Dict) -> Dict[str, str]:
    question = item["question"]
    choices = item["choices"]
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])])
    input_text = ARC_QUESTION_PROMPT_TEMPLATE.format(question=question, choices_text=choices_text) + ARC_QUESTION_POSTFIX
    answer_key = normalize_answer_text(item["answerKey"])
    return {"question": input_text, "gold_final": answer_key}


def prepare_dart_sample(item: Dict) -> Dict[str, str]:
    question = item["query"]
    answer_key = normalize_answer_text(item["gt_ans"])
    return {"question": f"Question: {question}", "gold_final": answer_key}


def _uniform_sample_indices(total: int, sample_count: int, rng: random.Random) -> List[int]:
    if total <= 0:
        return []
    if total >= sample_count:
        return rng.sample(range(total), sample_count)
    # Keep exact per-dataset count even for small splits by sampling with replacement when needed.
    return [rng.randrange(total) for _ in range(sample_count)]


def build_uniform_mixed_train_set(per_dataset_samples: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    mixed_records: List[Dict[str, str]] = []

    dataset_defs: List[Tuple[str, str, str, int]] = [
        ("arc_easy", "allenai/ai2_arc", "ARC-Easy", 0),
        ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", 0),
        ("dart_1", "hkust-nlp/dart-math-pool-math", "", 1),
        ("dart_2", "hkust-nlp/dart-math-pool-math", "", 2),
        ("dart_3", "hkust-nlp/dart-math-pool-math", "", 3),
        ("dart_4", "hkust-nlp/dart-math-pool-math", "", 4),
        ("dart_5", "hkust-nlp/dart-math-pool-math", "", 5),
    ]

    for name, dataset_id, config_name, dart_level in dataset_defs:
        if name.startswith("arc"):
            ds = load_dataset(dataset_id, config_name, split="train")
            indices = _uniform_sample_indices(len(ds), per_dataset_samples, rng)
            for idx in indices:
                mixed_records.append(prepare_arc_sample(ds[int(idx)]))
        else:
            ds = load_dataset(dataset_id, split="train")
            ds = ds.filter(lambda x: x["query_metadata"]["level"] == dart_level)
            indices = _uniform_sample_indices(len(ds), per_dataset_samples, rng)
            for idx in indices:
                mixed_records.append(prepare_dart_sample(ds[int(idx)]))

    rng.shuffle(mixed_records)
    return mixed_records


def llada_plan_prompt(question: str) -> str:
    return (
        "You are a careful planning assistant.\n"
        "Given the math question, produce only a concise step-by-step PLAN.\n"
        "Do not output the final numeric answer.\n\n"
        f"Question: {question}\n\n"
        "Plan:"
    )


def build_executor_prompt_parts(question: str) -> tuple[str, str]:
    prefix = (
        "You are a math solver.\n"
        "Use the refined plan to solve the question.\n"
        "Return only the final numeric answer.\n\n"
        f"Question: {question}\n\n"
        "Refined plan:\n"
    )
    suffix = "\n\nAnswer:"
    return prefix, suffix


def extract_last_number(text: str) -> str:
    import re

    nums = re.findall(r"[-+]?\d+[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else ""


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
    ).to(next(model.parameters()).device)
    return model


def init_text_projector(model) -> None:
    with torch.no_grad():
        nn.init.xavier_uniform_(model.text_projector[0].weight)
        nn.init.zeros_(model.text_projector[0].bias)
        nn.init.xavier_uniform_(model.text_projector[2].weight)
        nn.init.zeros_(model.text_projector[2].bias)
        nn.init.xavier_uniform_(model.text_projector[4].weight)
        nn.init.zeros_(model.text_projector[4].bias)


@dataclass
class Config:
    answer_model_id: str
    llada_model_id: str
    per_dataset_train_samples: int
    max_test_samples: int
    max_length: int
    draft_plan_max_new_tokens: int
    answer_max_new_tokens: int
    enable_oom_fallback: bool
    output_dir: str
    run_name: str
    seed: int
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    logging_steps: int
    save_steps: int
    lightly_tune_qwen: bool
    multimodal: bool


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="LLaDA -> text_projector -> Llama training")
    ap.add_argument("--answer_model_id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--llada_model_id", default="GSAI-ML/LLaDA-8B-Instruct")

    ap.add_argument("--per_dataset_train_samples", type=int, default=500)
    ap.add_argument("--max_test_samples", type=int, default=70)
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--draft_plan_max_new_tokens", type=int, default=96)
    ap.add_argument("--answer_max_new_tokens", type=int, default=64)
    ap.add_argument("--disable_oom_fallback", action="store_true")

    ap.add_argument("--output_dir", default="models/dart-text-projector")
    ap.add_argument("--run_name", default="llada-projector-llama")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--lightly_tune_qwen", action="store_true")
    ap.add_argument("--multimodal", action="store_true", help="Train both text and latent projectors together")

    args = ap.parse_args()
    return Config(
        answer_model_id=args.answer_model_id,
        llada_model_id=args.llada_model_id,
        per_dataset_train_samples=args.per_dataset_train_samples,
        max_test_samples=args.max_test_samples,
        max_length=args.max_length,
        draft_plan_max_new_tokens=args.draft_plan_max_new_tokens,
        answer_max_new_tokens=args.answer_max_new_tokens,
        enable_oom_fallback=not args.disable_oom_fallback,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        lightly_tune_qwen=args.lightly_tune_qwen,
        multimodal=args.multimodal,
    )


def load_split(cfg: Config):
    mixed_train = build_uniform_mixed_train_set(
        per_dataset_samples=cfg.per_dataset_train_samples,
        seed=cfg.seed,
    )
    if cfg.max_test_samples > 0:
        eval_records = mixed_train[: min(cfg.max_test_samples, len(mixed_train))]
    else:
        eval_records = []
    return mixed_train, eval_records


def build_draft_generator(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(cfg.llada_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.llada_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=cfg.draft_plan_max_new_tokens,
    )


def generate_draft_plan(question: str, draft_generator, cfg: Config) -> str:
    prompt = llada_plan_prompt(question)
    try:
        out = draft_generator(prompt, return_full_text=False, use_cache=False)[0]["generated_text"]
        out = out.strip()
        return out if out else "Parse the quantities and solve step-by-step."
    except torch.cuda.OutOfMemoryError:
        if not cfg.enable_oom_fallback:
            raise
        free_cuda_memory()
        return "Parse the quantities and solve step-by-step."


def build_records(cfg: Config, raw_train_records: List[Dict[str, str]], eval_seed_records: List[Dict[str, str]]):
    print("Loading LLaDA draft generator...")
    draft_generator = build_draft_generator(cfg)

    train_records: List[Dict[str, str]] = []
    eval_records: List[Dict[str, str]] = []

    print("Generating LLaDA draft plans...")
    for ex in raw_train_records:
        question = ex["question"]
        gold_final = ex["gold_final"]
        draft_plan = generate_draft_plan(question, draft_generator, cfg)
        train_records.append(
            {
                "question": question,
                "draft_plan": draft_plan,
                "gold_final": gold_final,
            }
        )

    for ex in eval_seed_records:
        question = ex["question"]
        gold_final = ex["gold_final"]
        draft_plan = generate_draft_plan(question, draft_generator, cfg)
        eval_records.append(
            {
                "question": question,
                "draft_plan": draft_plan,
                "gold_final": gold_final,
            }
        )

    del draft_generator
    free_cuda_memory()
    return train_records, eval_records


class PlanProjectorTrainer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.answer_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.qwen = AutoModelForCausalLM.from_pretrained(
            cfg.answer_model_id,
            trust_remote_code=True,
            torch_dtype=self.model_dtype,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qwen = self.qwen.to(self.device)
        self.qwen = add_text_projector(self.qwen, self.qwen.config)
        init_text_projector(self.qwen)
        self.qwen.text_projector = self.qwen.text_projector.to(device=self.device, dtype=self.model_dtype)

        self._configure_trainable_params(lightly_tune_qwen=cfg.lightly_tune_qwen)

    def _configure_trainable_params(self, lightly_tune_qwen: bool) -> None:
        for param in self.qwen.parameters():
            param.requires_grad = False

        for param in self.qwen.text_projector.parameters():
            param.requires_grad = True

        if lightly_tune_qwen:
            if hasattr(self.qwen, "lm_head"):
                for param in self.qwen.lm_head.parameters():
                    param.requires_grad = True
            if hasattr(self.qwen, "model") and hasattr(self.qwen.model, "norm"):
                for param in self.qwen.model.norm.parameters():
                    param.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.qwen.parameters() if p.requires_grad]

    def _tokenize_no_special(self, text: str, max_len: int) -> torch.Tensor:
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )["input_ids"]
        return ids.to(self.device)

    def compute_loss(self, question: str, draft_plan: str, gold_final: str, max_length: int) -> torch.Tensor:
        prefix, suffix = build_executor_prompt_parts(question)
        target = gold_final.strip() + self.tokenizer.eos_token

        prefix_ids = self._tokenize_no_special(prefix, max_length)
        draft_ids = self._tokenize_no_special(draft_plan, max_length)
        suffix_ids = self._tokenize_no_special(suffix, max_length)
        target_ids = self._tokenize_no_special(target, max_length)

        max_total = max_length
        keep_target = min(target_ids.shape[1], max_total)
        target_ids = target_ids[:, -keep_target:]

        available_context = max_total - keep_target
        if available_context < 1:
            available_context = 1

        context_ids = torch.cat([prefix_ids, draft_ids, suffix_ids], dim=1)
        if context_ids.shape[1] > available_context:
            context_ids = context_ids[:, -available_context:]

        prefix_len = min(prefix_ids.shape[1], context_ids.shape[1])
        draft_len = min(draft_ids.shape[1], max(context_ids.shape[1] - prefix_len, 0))

        embeddings = self.qwen.get_input_embeddings()
        context_embeds = embeddings(context_ids)
        # avoid in-place modifications on a view returned by embeddings(...) which
        # can break autograd (versioning). Work on a clone instead.
        context_embeds = context_embeds.clone()

        if draft_len > 0:
            draft_start = prefix_len
            draft_end = min(draft_start + draft_len, context_embeds.shape[1])
            draft_embeds = context_embeds[:, draft_start:draft_end, :]
            projected = self.qwen.text_projector(draft_embeds.to(self.model_dtype)).to(context_embeds.dtype)
            context_embeds[:, draft_start:draft_end, :] = projected

        target_embeds = embeddings(target_ids)
        inputs_embeds = torch.cat([context_embeds, target_embeds], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device, dtype=torch.long)
        labels = torch.full(
            (1, inputs_embeds.shape[1]),
            -100,
            device=self.device,
            dtype=torch.long,
        )
        labels[:, context_embeds.shape[1] :] = target_ids

        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate_answer(self, question: str, draft_plan: str, max_length: int, max_new_tokens: int) -> str:
        self.qwen.eval()
        prefix, suffix = build_executor_prompt_parts(question)

        prefix_ids = self._tokenize_no_special(prefix, max_length)
        draft_ids = self._tokenize_no_special(draft_plan, max_length)
        suffix_ids = self._tokenize_no_special(suffix, max_length)

        context_ids = torch.cat([prefix_ids, draft_ids, suffix_ids], dim=1)
        if context_ids.shape[1] > max_length:
            context_ids = context_ids[:, -max_length:]

        prefix_len = min(prefix_ids.shape[1], context_ids.shape[1])
        draft_len = min(draft_ids.shape[1], max(context_ids.shape[1] - prefix_len, 0))

        embeddings = self.qwen.get_input_embeddings()
        context_embeds = embeddings(context_ids)
        # clone to prevent in-place write on a view (fixes autograd version error)
        context_embeds = context_embeds.clone()

        if draft_len > 0:
            draft_start = prefix_len
            draft_end = min(draft_start + draft_len, context_embeds.shape[1])
            draft_embeds = context_embeds[:, draft_start:draft_end, :]
            projected = self.qwen.text_projector(draft_embeds.to(self.model_dtype)).to(context_embeds.dtype)
            context_embeds[:, draft_start:draft_end, :] = projected

        output_ids = self.qwen.generate(
            inputs_embeds=context_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_only = output_ids[:, context_embeds.shape[1] :]
        return self.tokenizer.decode(gen_only[0], skip_special_tokens=True).strip()


class MultiModalProjectorTrainer(PlanProjectorTrainer):
    """Train both the text-based projector and a latent-based projector simultaneously.

    Workflow (per sample):
    - Use the same question input to obtain LLaDA draft text (already provided in training records)
    - Also obtain the last hidden-layer activations from a separate "llada" model for the draft tokens
    - Project (a) draft token embeddings through `text_projector` (inherited) and (b) llada hidden states
      through `latent_projector` into the Qwen embedding space
    - For each pathway construct `inputs_embeds` for Qwen and compute a cross-entropy loss
    - Return the (optionally weighted) sum of the two losses so both projectors are trained together
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

        # Load a LLaDA model (used to produce hidden-layer latents for the same input)
        # We load the raw model (not a pipeline) so we can extract hidden states.
        self.llada_tokenizer = AutoTokenizer.from_pretrained(cfg.llada_model_id, trust_remote_code=True)
        if self.llada_tokenizer.pad_token is None:
            self.llada_tokenizer.pad_token = self.llada_tokenizer.eos_token

        self.llada = AutoModelForCausalLM.from_pretrained(
            cfg.llada_model_id,
            trust_remote_code=True,
            torch_dtype=self.model_dtype,
            output_hidden_states=True,
        )
        self.llada = self.llada.to(self.device)

        # Build a latent projector that maps llada hidden dim -> qwen hidden dim
        llada_hidden = getattr(self.llada.config, "hidden_size", None)
        if llada_hidden is None:
            llada_hidden = self.llada.config.hidden_size if hasattr(self.llada.config, "hidden_size") else 4096

        qwen_hidden = getattr(self.qwen.config, "hidden_size", self.qwen.config.hidden_size)

        # Bottleneck reuse: use same bottleneck dim as text projector
        bottleneck = getattr(self.qwen, "bottleneck_dim", 1024)

        self.latent_projector = nn.Sequential(
            nn.Linear(llada_hidden, bottleneck),
            nn.GELU(approximate="tanh"),
            nn.Linear(bottleneck, qwen_hidden),
            LlamaRMSNorm(qwen_hidden, eps=self.qwen.config.rms_norm_eps),
        ).to(device=self.device)

        # Initialize latent projector
        with torch.no_grad():
            for idx, mod in enumerate(self.latent_projector):
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    nn.init.zeros_(mod.bias)

        # Make sure latent projector is trainable
        for p in self.latent_projector.parameters():
            p.requires_grad = True

    def compute_dual_loss(self, question: str, draft_plan: str, gold_final: str, max_length: int, latent_loss_weight: float = 1.0) -> torch.Tensor:
        """Compute combined loss from (A) text-projector pathway and (B) latent-projector pathway.

        - The text pathway reuses `compute_loss` from the parent class but with the draft_plan
        - The latent pathway obtains hidden states from `self.llada` for the draft tokens, projects
          them with `self.latent_projector`, and computes a second loss by feeding the projected
          latents into Qwen in place of the draft token embeddings.
        """
        # ---- Text pathway loss (reuse existing compute_loss implementation) ----
        text_loss = super().compute_loss(question=question, draft_plan=draft_plan, gold_final=gold_final, max_length=max_length)

        # ---- Latent pathway loss ----
        # Build prefix/suffix and tokenize using qwen tokenizer for target processing
        prefix, suffix = build_executor_prompt_parts(question)
        target = gold_final.strip() + self.tokenizer.eos_token

        # Tokenize draft with llada tokenizer and build generation prompt
        m = [{"role": "user", "content": draft_plan}]
        # use the LLaDA tokenizer's chat template to add generation prompt tokens (same as in DARM)
        try:
            gen_prompt = self.llada_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        except Exception:
            # Fallback: use raw draft_plan
            gen_prompt = draft_plan

        draft_input_ids = self.llada_tokenizer(gen_prompt)["input_ids"]
        draft_input_ids = torch.tensor(draft_input_ids, device=self.device).unsqueeze(0)

        # Dynamically import the DARM generate routine (which runs the denoising loop and returns hidden states)
        import importlib.util
        from pathlib import Path

        darm_path = Path(__file__).resolve().parents[1] / "DARM" / "pts" / "train" / "generate_latents.py"
        latent_module = None
        if darm_path.exists():
            spec = importlib.util.spec_from_file_location("darm_generate_latents", str(darm_path))
            latent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(latent_module)

        # Default: do a simple forward if we cannot import the custom generator
        if latent_module is None or not hasattr(latent_module, "generate"):
            draft_ids_ll = self.llada_tokenizer(draft_plan, add_special_tokens=False, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].to(self.device)
            with torch.no_grad():
                ll_out = self.llada(input_ids=draft_ids_ll)
                last_hidden = ll_out.hidden_states[-1]
        else:
            # Parameters can be tuned; use sensible defaults or cfg values if present
            steps = getattr(self.cfg, "llada_gen_steps", 128)
            gen_length = getattr(self.cfg, "llada_gen_length", 128)
            block_length = getattr(self.cfg, "llada_block_length", 32)
            temperature = getattr(self.cfg, "llada_temperature", 0.0)
            cfg_scale = getattr(self.cfg, "llada_cfg_scale", 0.0)
            remasking = getattr(self.cfg, "llada_remasking", "low_confidence")

            # run the denoising generator: returns (sequences, hidden_states)
            with torch.no_grad():
                out, hidden_states = latent_module.generate(
                    self.llada,
                    draft_input_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=remasking,
                )

            input_length = draft_input_ids.shape[1]
            # hidden_states shape: (batch, seq_len, hidden_dim) per iteration; the helper returns final hidden states
            # take the hidden states corresponding to generated/unmasked positions
            last_hidden = hidden_states[:, input_length:, :]

        # Project llada hidden states to qwen hidden dim
        projected = self.latent_projector(last_hidden.to(self.model_dtype)).to(dtype=self.qwen.get_input_embeddings().weight.dtype)

        # Now construct qwen inputs: prefix_ids + projected (as draft) + suffix + target
        prefix_ids = self._tokenize_no_special(prefix, max_length)
        suffix_ids = self._tokenize_no_special(suffix, max_length)
        target_ids = self._tokenize_no_special(target, max_length)

        # Prepare context embeddings using qwen input embeddings for prefix/suffix
        embeddings = self.qwen.get_input_embeddings()
        context_prefix = embeddings(prefix_ids)
        context_suffix = embeddings(suffix_ids)

        # Ensure projected has same batch dim and dtype as embeddings
        # projected shape: (1, L_draft, qwen_hidden)
        # Concatenate prefix + projected + suffix
        context_embeds = torch.cat([context_prefix, projected, context_suffix], dim=1)

        # Target embeds
        target_embeds = embeddings(target_ids)
        inputs_embeds = torch.cat([context_embeds, target_embeds], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device, dtype=torch.long)
        labels = torch.full((1, inputs_embeds.shape[1]), -100, device=self.device, dtype=torch.long)
        labels[:, context_embeds.shape[1] :] = target_ids

        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        latent_loss = outputs.loss

        # Combined loss (weighted)
        return text_loss + latent_loss_weight * latent_loss


class MultiModalTrainer:
    """Training loop manager for simultaneous text+latent projectors."""

    def __init__(self, cfg: Config, model_wrapper: MultiModalProjectorTrainer):
        self.cfg = cfg
        self.model = model_wrapper
        self.device = model_wrapper.device
        self.optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def train(self, train_records: List[Dict[str, str]]) -> Dict[str, float]:
        self.model.qwen.train()

        global_step = 0
        losses: List[float] = []
        running_loss = 0.0

        for epoch in range(self.cfg.num_train_epochs):
            random.shuffle(train_records)
            self.optimizer.zero_grad(set_to_none=True)

            for idx, rec in enumerate(train_records, start=1):
                loss = self.model.compute_dual_loss(
                    question=rec["question"],
                    draft_plan=rec["draft_plan"],
                    gold_final=rec["gold_final"],
                    max_length=self.cfg.max_length,
                )

                loss_to_backward = loss / self.cfg.gradient_accumulation_steps
                loss_to_backward.backward()

                running_loss += loss.item()

                if idx % self.cfg.gradient_accumulation_steps == 0 or idx == len(train_records):
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    step_loss = running_loss / self.cfg.gradient_accumulation_steps
                    losses.append(step_loss)
                    running_loss = 0.0

                    if global_step % self.cfg.logging_steps == 0:
                        print(f"epoch={epoch + 1} step={global_step} loss={step_loss:.6f}")

                    if global_step % self.cfg.save_steps == 0:
                        ckpt_dir = Path(self.cfg.output_dir) / f"checkpoint-{global_step}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(self.model.qwen.text_projector.state_dict(), ckpt_dir / "text_projector.pth")
                        if hasattr(self.model, "latent_projector"):
                            torch.save(self.model.latent_projector.state_dict(), ckpt_dir / "latent_projector.pth")
                        print(f"Saved projector checkpoint to: {ckpt_dir}")

        mean_loss = float(np.mean(losses)) if losses else 0.0
        return {"global_steps": global_step, "mean_loss": mean_loss}


def train_projector(cfg: Config, train_records: List[Dict[str, str]], model_wrapper: PlanProjectorTrainer) -> Dict[str, float]:
    model_wrapper.qwen.train()

    optimizer = torch.optim.AdamW(
        model_wrapper.trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    global_step = 0
    losses: List[float] = []
    running_loss = 0.0

    for epoch in range(cfg.num_train_epochs):
        random.shuffle(train_records)
        optimizer.zero_grad(set_to_none=True)

        for idx, rec in enumerate(train_records, start=1):
            loss = model_wrapper.compute_loss(
                question=rec["question"],
                draft_plan=rec["draft_plan"],
                gold_final=rec["gold_final"],
                max_length=cfg.max_length,
            )
            loss_to_backward = loss / cfg.gradient_accumulation_steps
            loss_to_backward.backward()

            running_loss += loss.item()

            if idx % cfg.gradient_accumulation_steps == 0 or idx == len(train_records):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                step_loss = running_loss / cfg.gradient_accumulation_steps
                losses.append(step_loss)
                running_loss = 0.0

                if global_step % cfg.logging_steps == 0:
                    print(f"epoch={epoch + 1} step={global_step} loss={step_loss:.6f}")

                if global_step % cfg.save_steps == 0:
                    ckpt_dir = Path(cfg.output_dir) / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model_wrapper.qwen.text_projector.state_dict(), ckpt_dir / "text_projector.pth")
                    print(f"Saved projector checkpoint to: {ckpt_dir / 'text_projector.pth'}")

    mean_loss = float(np.mean(losses)) if losses else 0.0
    return {"global_steps": global_step, "mean_loss": mean_loss}


def quick_eval(cfg: Config, eval_records: List[Dict[str, str]], model_wrapper: PlanProjectorTrainer) -> float:
    correct = 0
    total = len(eval_records)

    for ex in eval_records:
        pred_text = model_wrapper.generate_answer(
            question=ex["question"],
            draft_plan=ex["draft_plan"],
            max_length=cfg.max_length,
            max_new_tokens=cfg.answer_max_new_tokens,
        )
        pred_num = extract_last_number(pred_text)
        correct += int(pred_num == ex["gold_final"])

    return correct / total if total else 0.0


def main() -> None:
    cfg = parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    set_seed(cfg.seed)

    raw_train, raw_test = load_split(cfg)
    print(f"Train samples: {len(raw_train)} | Eval samples: {len(raw_test)}")

    train_records, eval_records = build_records(cfg, raw_train, raw_test)
    print(f"Prepared projector training records: {len(train_records)}")

    if cfg.multimodal:
        print("Loading Qwen + text_projector + latent_projector...")
        model_wrapper = MultiModalProjectorTrainer(cfg)
    else:
        print("Loading Qwen + text_projector...")
        model_wrapper = PlanProjectorTrainer(cfg)

    trainable_count = sum(p.numel() for p in model_wrapper.trainable_parameters())
    print(f"Trainable parameters: {trainable_count}")

    print("Starting projector training...")
    if cfg.multimodal:
        trainer = MultiModalTrainer(cfg, model_wrapper)
        train_stats = trainer.train(train_records)
    else:
        train_stats = train_projector(cfg, train_records, model_wrapper)

    final_dir = Path(cfg.output_dir) / "final_merged"
    final_dir.mkdir(parents=True, exist_ok=True)
    projector_path = final_dir / "text_projector.pth"
    torch.save(model_wrapper.qwen.text_projector.state_dict(), projector_path)
    
    if cfg.multimodal and hasattr(model_wrapper, "latent_projector"):
        latent_projector_path = final_dir / "latent_projector.pth"
        torch.save(model_wrapper.latent_projector.state_dict(), latent_projector_path)
        print(f"Saved latent projector to: {latent_projector_path}")

    model_wrapper.tokenizer.save_pretrained(str(final_dir))

    print("Training finished.")
    print(f"Saved text projector to: {projector_path}")

    acc = quick_eval(cfg, eval_records, model_wrapper)
    print(f"Quick eval samples: {len(eval_records)}")
    print(f"Exact-number accuracy: {acc:.3f}")

    meta_path = Path(cfg.output_dir) / "training_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "num_train_records": len(train_records),
                "num_eval_records": len(eval_records),
                "train_global_steps": train_stats["global_steps"],
                "train_mean_loss": train_stats["mean_loss"],
                "quick_eval_exact_number_accuracy": acc,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
