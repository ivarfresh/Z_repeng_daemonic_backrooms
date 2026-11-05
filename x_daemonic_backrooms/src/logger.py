"""
Conversation Logger
===================

Comprehensive logging system for backroom conversations, capturing:
- Per-token logits
- Derived metrics (entropy, top-k, etc.)
- Optional attention weights
- Session metadata and manifest
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from .metrics import compute_token_metrics


@dataclass
class TurnLog:
    """Data class for a single conversation turn."""
    turn_id: int
    agent: str
    timestamp: str
    prompt: str
    response: str
    control_coeff: float
    token_count: int

    # File references (not inline data)
    logits_file: str
    metrics_file: str
    attention_file: Optional[str]


class ConversationLogger:
    """
    Logger for capturing all data from a backroom conversation session.

    Creates a structured directory with:
    - manifest.json: Session metadata and turn index
    - conversation.txt: Human-readable transcript
    - vocab.json: Token ID → string mapping
    - logits/: Raw logits per turn (large, binary)
    - metrics/: Derived metrics per turn (smaller, compressed)
    - attention/: Attention weights per turn (optional, very large)
    """

    def __init__(self, session_dir: Path, save_attention: bool = False):
        """
        Initialize logger for a new session.

        Args:
            session_dir: Directory to store session data
            save_attention: Whether to save attention weights (warning: large!)
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.save_attention = save_attention
        self.turns: list[TurnLog] = []
        self.tokenizer = None  # set during init

        # Create subdirectories
        (self.session_dir / "logits").mkdir(exist_ok=True)
        (self.session_dir / "metrics").mkdir(exist_ok=True)
        if save_attention:
            (self.session_dir / "attention").mkdir(exist_ok=True)

        print(f"Initialized logger: {self.session_dir}")

    def set_tokenizer(self, tokenizer):
        """
        Store tokenizer and save vocabulary for visualization.

        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer

        # Save vocab for visualization (token_id → string)
        vocab = {v: k for k, v in tokenizer.get_vocab().items()}
        with open(self.session_dir / "vocab.json", "w") as f:
            json.dump(vocab, f, indent=2)

    def log_turn(
        self,
        turn_id: int,
        agent: str,
        prompt: str,
        response: str,
        metadata: dict,
        control_coeff: float
    ):
        """
        Log a single conversation turn with all associated data.

        Args:
            turn_id: Turn number in conversation
            agent: Agent identifier ("A", "B", etc.)
            prompt: Input prompt for this turn
            response: Generated response text
            metadata: Dict containing input_ids, output_ids, logits, etc.
            control_coeff: Control vector coefficient used
        """
        timestamp = datetime.now().isoformat()

        # Compute metrics from logits
        metrics = {}
        if "logits" in metadata and "output_ids" in metadata:
            metrics = compute_token_metrics(
                metadata["logits"],
                metadata["output_ids"],
                k=10
            )

        # Save logits (large, binary)
        logits_file = f"logits/turn_{turn_id:04d}.npy"
        if "logits" in metadata:
            np.save(
                self.session_dir / logits_file,
                metadata["logits"].numpy()
            )

        # Save metrics (derived, smaller)
        metrics_file = f"metrics/turn_{turn_id:04d}.npz"
        if metrics:
            np.savez(
                self.session_dir / metrics_file,
                **metrics,
                output_ids=metadata["output_ids"].numpy()
            )

        # Save attention (optional, very large)
        attention_file = None
        if self.save_attention and "attentions" in metadata:
            attention_file = f"attention/turn_{turn_id:04d}.npz"
            # Attention is nested: tuple of (layer, batch, heads, seq, seq)
            # Flatten to dict for saving
            att_dict = {}
            for layer_idx, layer_att in enumerate(metadata["attentions"]):
                if layer_att is not None:
                    att_dict[f"layer_{layer_idx}"] = layer_att[0].cpu().numpy()
            np.savez_compressed(self.session_dir / attention_file, **att_dict)

        # Create turn log entry
        turn_log = TurnLog(
            turn_id=turn_id,
            agent=agent,
            timestamp=timestamp,
            prompt=prompt,
            response=response,
            control_coeff=control_coeff,
            token_count=len(metadata.get("output_ids", [])),
            logits_file=logits_file,
            metrics_file=metrics_file,
            attention_file=attention_file
        )

        self.turns.append(turn_log)

        # Append to running conversation log (text only, for quick review)
        with open(self.session_dir / "conversation.txt", "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"TURN {turn_id} | {agent} | {timestamp}\n")
            f.write(f"{'='*60}\n")
            f.write(f"PROMPT:\n{prompt}\n\n")
            f.write(f"RESPONSE:\n{response}\n")

    def finalize(self, config: dict = None):
        """
        Save session manifest and close logger.

        Args:
            config: Optional configuration dict to include in manifest
        """
        manifest = {
            "session_id": self.session_dir.name,
            "created_at": datetime.now().isoformat(),
            "total_turns": len(self.turns),
            "config": config or {},
            "turns": [asdict(t) for t in self.turns]
        }

        with open(self.session_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Session saved to: {self.session_dir}")
        print(f"Total turns: {len(self.turns)}")
        print(f"{'='*60}")

    def get_storage_usage(self) -> dict:
        """
        Calculate actual storage usage of the session.

        Returns:
            Dictionary with size information in bytes
        """
        total_size = 0
        file_counts = {"logits": 0, "metrics": 0, "attention": 0, "other": 0}

        for item in self.session_dir.rglob("*"):
            if item.is_file():
                size = item.stat().st_size
                total_size += size

                if "logits" in str(item):
                    file_counts["logits"] += 1
                elif "metrics" in str(item):
                    file_counts["metrics"] += 1
                elif "attention" in str(item):
                    file_counts["attention"] += 1
                else:
                    file_counts["other"] += 1

        return {
            "total_bytes": total_size,
            "total_mb": total_size / (1024 * 1024),
            "total_gb": total_size / (1024 * 1024 * 1024),
            "file_counts": file_counts
        }
