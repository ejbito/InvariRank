from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve_dtype(dtype_name: str):
    import torch

    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def select_device(requested: str):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_tokenizer(cfg: Any):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
    )
    special_tokens = [
        cfg.span_start_token,
        cfg.span_end_token,
        cfg.item_start_token,
        cfg.item_end_token,
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    validate_special_tokens(tokenizer, cfg)
    return tokenizer


def validate_special_tokens(tokenizer: Any, cfg: Any) -> None:
    tokens = [
        cfg.span_start_token,
        cfg.span_end_token,
        cfg.item_start_token,
        cfg.item_end_token,
    ]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    missing = []
    split = []
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or token_id == unk_id:
            missing.append(token)
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            split.append(token)
    if missing:
        raise ValueError(f"Special token(s) missing from tokenizer vocabulary: {missing}")
    if split:
        raise ValueError(f"Special token(s) do not tokenize as single tokens: {split}")


def load_base_model(cfg: Any, tokenizer: Any, device: Any):
    from transformers import AutoModelForCausalLM

    dtype = resolve_dtype(getattr(cfg, "dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=dtype,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
    )
    model.resize_token_embeddings(len(tokenizer))
    return model.to(device)


def build_lora_model(cfg: Any, tokenizer: Any, device: Any):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    model = load_base_model(cfg, tokenizer, device)
    resume = getattr(cfg, "resume_checkpoint_path", None)
    if resume:
        return PeftModel.from_pretrained(model, resume, is_trainable=True)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        target_modules=list(cfg.lora_target_modules),
    )
    return get_peft_model(model, lora_cfg)


def load_model_for_ranking(cfg: Any, tokenizer: Any, device: Any):
    model = load_base_model(cfg, tokenizer, device)
    adapter_path = getattr(cfg, "adapter_path", None) or getattr(cfg, "checkpoint_path", None)
    if adapter_path:
        adapter_path = Path(adapter_path)
        if adapter_path.exists() and adapter_path.is_dir() and not (adapter_path / "adapter_config.json").exists():
            raise ValueError(
                f"Configured adapter_path exists but is not a PEFT adapter directory: {adapter_path}. "
                "Remove or unset adapter_path to use the original base model for zero-shot ranking."
            )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model


def model_dtype(model: Any):
    return next(model.parameters()).dtype
