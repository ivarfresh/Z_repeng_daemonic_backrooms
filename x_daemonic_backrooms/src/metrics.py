"""
Token Metrics Computation
==========================

Utilities for computing derived metrics from model logits and token outputs.
"""

import torch
import numpy as np


def compute_token_metrics(logits: torch.Tensor, chosen_ids: torch.Tensor, k: int = 10) -> dict:
    """
    Compute derived metrics from raw logits.

    Args:
        logits: (num_tokens, vocab_size) - Raw logits per generated token
        chosen_ids: (num_tokens,) - The actually generated token IDs
        k: Number of top alternatives to track

    Returns:
        Dictionary containing:
            - top_k_tokens: (num_tokens, k) - Top-k token IDs
            - top_k_probs: (num_tokens, k) - Top-k probabilities
            - entropy: (num_tokens,) - Shannon entropy per token
            - chosen_prob: (num_tokens,) - Probability of chosen token
            - chosen_rank: (num_tokens,) - Rank of chosen token in distribution
    """
    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Top-k tokens and probabilities
    top_k_probs, top_k_tokens = torch.topk(probs, k, dim=-1)

    # Entropy: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    # Probability of chosen tokens
    chosen_prob = probs[torch.arange(len(chosen_ids)), chosen_ids]

    # Rank of chosen token (0 = highest probability)
    sorted_indices = torch.argsort(probs, dim=-1, descending=True)
    chosen_rank = (sorted_indices == chosen_ids.unsqueeze(-1)).nonzero(as_tuple=True)[1]

    return {
        "top_k_tokens": top_k_tokens.numpy(),
        "top_k_probs": top_k_probs.numpy(),
        "entropy": entropy.numpy(),
        "chosen_prob": chosen_prob.numpy(),
        "chosen_rank": chosen_rank.numpy()
    }


def analyze_token_distribution(logits: torch.Tensor, vocab_size: int = None) -> dict:
    """
    Analyze the full probability distribution for debugging/visualization.

    Args:
        logits: (num_tokens, vocab_size)
        vocab_size: Optional vocab size for validation

    Returns:
        Dictionary with distribution statistics
    """
    probs = torch.softmax(logits, dim=-1)

    return {
        "mean_entropy": float(torch.mean(-torch.sum(probs * torch.log(probs + 1e-10), dim=-1))),
        "max_entropy": float(torch.max(-torch.sum(probs * torch.log(probs + 1e-10), dim=-1))),
        "min_entropy": float(torch.min(-torch.sum(probs * torch.log(probs + 1e-10), dim=-1))),
        "mean_max_prob": float(torch.mean(torch.max(probs, dim=-1)[0])),
        "min_max_prob": float(torch.min(torch.max(probs, dim=-1)[0])),
    }
