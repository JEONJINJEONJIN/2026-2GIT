"""Conditional log-likelihood action selection."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _prompt_to_text(tokenizer, prompt: str | Sequence[dict]) -> str:
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
    return str(prompt)


def _score_action(model, tokenizer, prompt_text: str, action_text: str) -> tuple[float, float]:
    device = _get_model_device(model)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
    full_inputs = tokenizer(prompt_text + action_text, return_tensors="pt")

    prompt_ids = prompt_inputs["input_ids"].to(device)
    input_ids = full_inputs["input_ids"].to(device)
    attention_mask = full_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_len = prompt_ids.shape[1]
    seq_len = input_ids.shape[1]
    action_len = seq_len - prompt_len
    if action_len <= 0:
        return float("-inf"), float("-inf")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    action_logits = logits[:, prompt_len - 1 : seq_len - 1, :]
    action_targets = input_ids[:, prompt_len:seq_len]
    log_probs = F.log_softmax(action_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, action_targets.unsqueeze(-1)).squeeze(-1)
    loglik = float(token_log_probs.sum().item())
    normalized = loglik / action_len
    return loglik, normalized


def select_action_by_loglik(
    model,
    tokenizer,
    prompt: str,
    action_texts: list[str],
    steering_hook=None,
    length_normalize: bool = True,
) -> dict:
    """Select the action candidate with highest conditional log-likelihood.

    Parameters
    ----------
    model, tokenizer:
        HuggingFace-style causal language model and matching tokenizer.
    prompt:
        Prompt text, or chat messages accepted by ``tokenizer.apply_chat_template``.
    action_texts:
        Candidate action continuations to score as ``p(action | prompt)``.
    steering_hook:
        Optional context manager or callable context manager that registers
        steering hooks around the scoring forward passes.
    length_normalize:
        If true, select by average token log-likelihood rather than summed
        log-likelihood.
    """
    prompt_text = _prompt_to_text(tokenizer, prompt)

    def run_scoring():
        logliks = []
        normalized_logliks = []
        for action_text in action_texts:
            loglik, normalized = _score_action(
                model,
                tokenizer,
                prompt_text,
                action_text,
            )
            logliks.append(loglik)
            normalized_logliks.append(normalized)
        return logliks, normalized_logliks

    if steering_hook is None:
        logliks, normalized_logliks = run_scoring()
    elif hasattr(steering_hook, "__enter__") and hasattr(steering_hook, "__exit__"):
        with steering_hook:
            logliks, normalized_logliks = run_scoring()
    elif callable(steering_hook):
        with steering_hook():
            logliks, normalized_logliks = run_scoring()
    else:
        logliks, normalized_logliks = run_scoring()

    selection_scores = normalized_logliks if length_normalize else logliks
    score_tensor = torch.tensor(selection_scores, dtype=torch.float32)
    selected_idx = int(torch.argmax(score_tensor).item())
    softmax_probs = torch.softmax(score_tensor, dim=0).tolist()

    return {
        "selected_idx": selected_idx,
        "logliks": logliks,
        "normalized_logliks": normalized_logliks,
        "softmax_probs": [float(prob) for prob in softmax_probs],
    }
