"""Model loading module for the RSA project.

Loads Qwen2.5-3B-Instruct (or a configured alternative) from
HuggingFace Hub with support for fp16 and 4-bit (NF4) quantization.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

logger = logging.getLogger(__name__)


def _build_quantization_config(config: dict[str, Any]):
    """Create a ``BitsAndBytesConfig`` for 4-bit quantization.

    Returns ``None`` when quantization is not requested.
    """
    quantization = config.get("quantization") or (
        "4bit" if config.get("load_in_4bit") else None
    )
    if quantization != "4bit":
        return None

    from transformers import BitsAndBytesConfig

    compute_dtype_str = config.get("bnb_4bit_compute_dtype", "float16")
    compute_dtype = getattr(torch, compute_dtype_str, torch.float16)
    quant_type = config.get("bnb_4bit_quant_type", "nf4")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=True,
    )
    logger.info(
        "4-bit quantization enabled (type=%s, compute_dtype=%s)",
        quant_type,
        compute_dtype_str,
    )
    return bnb_config


def _resolve_dtype(config: dict[str, Any]) -> Optional[torch.dtype]:
    """Map the config's ``dtype`` string to a :class:`torch.dtype`."""
    dtype_str = config.get("dtype", "float16")
    if dtype_str in ("auto", None):
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = mapping.get(dtype_str)
    if dtype is None:
        logger.warning("Unknown dtype '%s', falling back to float16", dtype_str)
        return torch.float16
    return dtype


def load_model_and_tokenizer(
    config: dict[str, Any],
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal-LM model and its tokenizer from HuggingFace.

    Parameters
    ----------
    config:
        Configuration dictionary.  Accepts either the full project
        config (with a top-level ``"model"`` key) or the ``model``
        sub-dict directly.  Relevant keys:

        * ``name`` -- HuggingFace model identifier
          (default: ``"Qwen/Qwen2.5-3B-Instruct"``).
        * ``dtype`` -- ``"float16"``, ``"bfloat16"``, ``"float32"``,
          or ``"auto"`` (default: ``"float16"``).
        * ``quantization`` -- ``null`` for full precision, ``"4bit"``
          for NF4 quantization via *bitsandbytes*.
        * ``load_in_4bit`` -- shorthand flag; equivalent to setting
          ``quantization: "4bit"``.
        * ``trust_remote_code`` -- passed through to HuggingFace
          (default: ``True``).
        * ``max_memory`` -- optional dict for explicit per-device VRAM
          limits (e.g. ``{0: "14GiB"}``).

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        The loaded model (on ``device_map="auto"``) and tokenizer.
    """
    # Allow passing the full config or just the model sub-dict.
    model_cfg: dict[str, Any] = config.get("model", config)

    model_name: str = model_cfg.get("name", "Qwen/Qwen2.5-3B-Instruct")
    trust_remote: bool = model_cfg.get("trust_remote_code", True)
    dtype = _resolve_dtype(model_cfg)
    quantization_config = _build_quantization_config(model_cfg)

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("pad_token set to eos_token (%s)", tokenizer.eos_token)

    # Build keyword arguments for model loading.
    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_name,
        "device_map": "auto",
        "trust_remote_code": trust_remote,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        # Only set torch_dtype when not using bitsandbytes (bnb
        # controls dtype internally).
        model_kwargs["torch_dtype"] = dtype

    max_memory = model_cfg.get("max_memory")
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory

    logger.info(
        "Loading model: %s (dtype=%s, quantization=%s)",
        model_name,
        dtype,
        "4bit" if quantization_config else "none",
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    except Exception:
        try:
            # Gemma4 and other multimodal models need their specific class
            from transformers import AutoModelForImageTextToText
            logger.info("Retrying with AutoModelForImageTextToText")
            model = AutoModelForImageTextToText.from_pretrained(**model_kwargs)
        except Exception:
            from transformers import AutoModel
            logger.info("Retrying with AutoModel")
            model = AutoModel.from_pretrained(**model_kwargs)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded: %s (%.1fM params, device_map=auto)",
        model_name,
        num_params / 1e6,
    )

    return model, tokenizer


def get_model_layers(model) -> "torch.nn.ModuleList":
    """Return the transformer decoder layers for any supported model architecture.

    Tries common layer paths in order:
      1. ``model.model.layers``                       — Qwen, LLaMA style
      2. ``model.model.language_model.layers``         — Gemma4 multimodal style
      3. ``model.language_model.model.layers``         — alternative multimodal style

    Raises
    ------
    AttributeError
        If none of the known paths exist on the model.
    """
    # Qwen / LLaMA style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Base decoder models loaded via AutoModel, e.g. Qwen2Model.
    if hasattr(model, "layers"):
        return model.layers
    # Gemma4 style: model.model.language_model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    # Alternative multimodal style
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        return model.language_model.model.layers
    raise AttributeError(
        f"Cannot locate transformer layers on {type(model).__name__}. "
        "Expected model.model.layers, model.layers, "
        "model.model.language_model.layers, or model.language_model.model.layers."
    )
