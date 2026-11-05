"""
Control Vector Training
========================

Train mystical/void control vector from contrastive pairs.
"""

import json
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path to import repeng
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from repeng import ControlVector, DatasetEntry


def load_divination_pairs(pairs_path: str) -> List[tuple]:
    """
    Load contrastive pairs from JSON file.

    Args:
        pairs_path: Path to divination_pairs.json

    Returns:
        List of (positive, negative) tuples
    """
    with open(pairs_path, 'r') as f:
        data = json.load(f)

    pairs = []

    # Load base pairs
    for pair in data.get("base_pairs", []):
        pairs.append((pair["positive"], pair["negative"]))

    # Load expanded pairs
    for pair in data.get("expanded_pairs", []):
        pairs.append((pair["positive"], pair["negative"]))

    print(f"Loaded {len(pairs)} contrastive pairs")
    return pairs


def expand_pairs_with_suffixes(
    pairs: List[tuple],
    suffixes: List[str],
    tokenizer,
    max_tokens: int = 20
) -> List[DatasetEntry]:
    """
    Expand pairs by appending truncated suffixes to create more training samples.

    This helps create a larger training dataset from a small set of base pairs.

    Args:
        pairs: List of (positive, negative) tuples
        suffixes: List of suffix strings to append
        tokenizer: HuggingFace tokenizer
        max_tokens: Maximum suffix length in tokens

    Returns:
        List of DatasetEntry objects for training
    """
    expanded = []

    for positive, negative in pairs:
        # Add base pair without suffix
        expanded.append(DatasetEntry(
            positive=positive,
            negative=negative
        ))

        # Add pairs with truncated suffixes
        for suffix in suffixes[:50]:  # Limit to first 50 suffixes to keep dataset manageable
            tokens = tokenizer.encode(suffix, add_special_tokens=False)

            # Create multiple entries with different truncation lengths
            for length in range(5, min(len(tokens), max_tokens), 3):
                truncated = tokenizer.decode(tokens[:length])
                expanded.append(DatasetEntry(
                    positive=f"{positive} {truncated}",
                    negative=f"{negative} {truncated}"
                ))

    print(f"Expanded to {len(expanded)} training samples")
    return expanded


def create_dataset_from_pairs(pairs: List[tuple]) -> List[DatasetEntry]:
    """
    Convert simple pairs to DatasetEntry objects.

    Args:
        pairs: List of (positive, negative) tuples

    Returns:
        List of DatasetEntry objects
    """
    return [DatasetEntry(positive=pos, negative=neg) for pos, neg in pairs]


def train_divination_vector(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    pairs_path: str = "data/divination_pairs.json",
    suffixes_path: str = None,
    output_path: str = "vectors/divination_v1.gguf",
    device: str = "cuda",
    batch_size: int = 8,
    method: str = "pca_diff",
    use_expansion: bool = False
) -> ControlVector:
    """
    Main training function for divination control vector.

    Args:
        model_name: HuggingFace model identifier
        pairs_path: Path to contrastive pairs JSON
        suffixes_path: Optional path to suffixes for dataset expansion
        output_path: Where to save trained vector
        device: Device to use ("cuda", "mps", or "cpu")
        batch_size: Training batch size
        method: Training method ("pca_diff" or "pca_center")
        use_expansion: Whether to expand dataset with suffixes

    Returns:
        Trained ControlVector
    """
    print("="*60)
    print("MULCURSIO Control Vector Training")
    print("="*60)

    # 1. Load model + tokenizer
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load pairs
    print(f"\nLoading pairs from: {pairs_path}")
    pairs = load_divination_pairs(pairs_path)

    # 3. Prepare dataset
    if use_expansion and suffixes_path:
        print(f"\nExpanding dataset with suffixes from: {suffixes_path}")
        with open(suffixes_path, 'r') as f:
            suffixes_data = json.load(f)
            suffixes = suffixes_data if isinstance(suffixes_data, list) else suffixes_data.get("suffixes", [])

        dataset = expand_pairs_with_suffixes(pairs, suffixes, tokenizer)
    else:
        print("\nUsing base pairs only (no expansion)")
        dataset = create_dataset_from_pairs(pairs)

    # 4. Train vector
    print(f"\nTraining control vector...")
    print(f"  Method: {method}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {len(dataset)}")

    vector = ControlVector.train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=batch_size,
        method=method
    )

    # 5. Export
    print(f"\nExporting to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vector.export_gguf(output_path)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    return vector


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train mystical divination control vector")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.1",
                        help="Model to train on")
    parser.add_argument("--pairs", default="data/divination_pairs.json",
                        help="Path to contrastive pairs JSON")
    parser.add_argument("--suffixes", default=None,
                        help="Path to suffixes JSON for expansion")
    parser.add_argument("--output", default="vectors/divination_v1.gguf",
                        help="Output path for trained vector")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="Device to use")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--method", default="pca_diff",
                        choices=["pca_diff", "pca_center"],
                        help="Training method")
    parser.add_argument("--expand", action="store_true",
                        help="Use suffix expansion")

    args = parser.parse_args()

    train_divination_vector(
        model_name=args.model,
        pairs_path=args.pairs,
        suffixes_path=args.suffixes,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        method=args.method,
        use_expansion=args.expand
    )
