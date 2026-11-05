# MULCURSIO Data Pipeline

A comprehensive data pipeline for training control vectors and running infinite backroom conversations between LLM instances, designed for visualization in generative video art.

## Overview

This pipeline enables:

1. **Control Vector Training**: Train a "mystical divination" control vector from contrastive pairs
2. **Infinite Backroom Conversations**: Run conversations between two LLM instances (one baseline, one with control vector)
3. **Comprehensive Logging**: Capture per-token logits, entropy, probabilities, and attention for visualization

## RunPod Training

Training can be done on RunPod for GPU acceleration. See detailed guides:

- **[RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md)** - Quick reference for common tasks
- **[RUNPOD_SETUP.md](RUNPOD_SETUP.md)** - Complete setup and usage guide

**Quick start:**
```bash
# Complete workflow (upload, train, download)
./scripts/runpod_full_workflow.sh

# Or step by step
./scripts/runpod_upload.sh      # Upload project
./scripts/runpod_train.sh       # Train vector
./scripts/runpod_download.sh vectors  # Download results
```

## Project Structure

```
x_daemonic_backrooms/
├── data/
│   └── divination_pairs.json          # Contrastive pairs for training
├── vectors/
│   └── (trained control vectors)
├── src/
│   ├── __init__.py
│   ├── metrics.py                     # Token metrics computation
│   ├── logger.py                      # Conversation logging system
│   ├── utils.py                       # Utility functions
│   ├── train_vector.py                # Control vector training
│   └── backroom_engine.py             # Main conversation loop
├── sessions/
│   └── (generated session data)
├── notebooks/
│   ├── 01_train_vector.ipynb          # Interactive training
│   ├── 02_run_session.ipynb           # Interactive session running
│   └── 03_analyze_session.ipynb       # Data exploration
├── scripts/
│   └── run_session.py                 # CLI session runner
├── requirements.txt
└── README.md
```

## Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Ensure repeng is available:**

This pipeline depends on the `repeng` library located in the parent directory. Make sure the path is correct:

```python
# The scripts automatically add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from repeng import ControlVector, ControlModel
```

## Quick Start

### 1. Train Control Vector

**Using the notebook (recommended for first time):**

```bash
jupyter notebook notebooks/01_train_vector.ipynb
```

**Using the script:**

```bash
cd src
python train_vector.py \
    --model mistralai/Mistral-7B-Instruct-v0.1 \
    --pairs ../data/divination_pairs.json \
    --output ../vectors/divination_v1.gguf \
    --device cuda \
    --batch-size 8
```

**Options:**
- `--device`: `cuda`, `mps` (Apple Silicon), or `cpu`
- `--batch-size`: Reduce if you get OOM errors
- `--method`: `pca_diff` (default) or `pca_center`
- `--expand`: Use suffix expansion (requires `--suffixes` path)

### 2. Run a Backroom Session

**Using the notebook (recommended):**

```bash
jupyter notebook notebooks/02_run_session.ipynb
```

**Using the CLI:**

```bash
cd scripts
python run_session.py \
    --turns 50 \
    --coeff 1.2 \
    --device cuda \
    --seed "We are two minds meeting in the infinite backroom. What do you perceive?"
```

**Options:**
- `--turns`: Number of conversation turns (default: 200)
- `--coeff`: Control vector coefficient for Agent B (default: 1.2)
- `--temperature`: Sampling temperature (default: 0.8)
- `--max-tokens`: Maximum tokens per turn (default: 256)
- `--save-attention`: Save attention weights (warning: very large!)
- `--device`: `cuda`, `mps`, or `cpu`

### 3. Analyze Session Data

```bash
jupyter notebook notebooks/03_analyze_session.ipynb
```

## Data Schema

### Session Directory Structure

```
sessions/session_YYYYMMDD_HHMMSS/
├── manifest.json           # Session metadata + turn index
├── conversation.txt        # Human-readable transcript
├── vocab.json              # Token ID → string mapping
├── config.json             # Generation config used
│
├── logits/
│   ├── turn_0000.npy       # (gen_len, vocab_size) float32
│   ├── turn_0001.npy
│   └── ...
│
├── metrics/
│   ├── turn_0000.npz       # Derived metrics per turn
│   ├── turn_0001.npz
│   └── ...
│
└── attention/              # Optional, very large
    ├── turn_0000.npz
    └── ...
```

### Metrics File Contents (NPZ)

Each `turn_XXXX.npz` contains:

- `output_ids`: (gen_len,) - Token IDs
- `top_k_tokens`: (gen_len, 10) - Top 10 alternative tokens
- `top_k_probs`: (gen_len, 10) - Their probabilities
- `entropy`: (gen_len,) - Entropy per token
- `chosen_prob`: (gen_len,) - Probability of chosen token
- `chosen_rank`: (gen_len,) - Rank of chosen token in distribution

## Storage Considerations

| Data Type | Per Turn | 200 Turns | Notes |
|-----------|----------|-----------|-------|
| Logits | ~4MB | ~800MB | Raw probability distributions |
| Metrics | ~50KB | ~10MB | Compressed derived data |
| Attention | ~100MB | ~20GB | Optional, very large |
| Text | ~2KB | ~400KB | Negligible |
| **Total (no attention)** | ~4MB | **~800MB** | Manageable |
| **Total (with attention)** | ~104MB | **~21GB** | Requires large storage |

**Recommendation:** Only save attention for selected "highlight" turns.

## Configuration Guide

### Agent Configuration

```python
from src.backroom_engine import AgentConfig

# Agent A: Baseline
agent_a = AgentConfig(
    name="Agent_A",
    control_vector_path=None,
    control_coeff=0.0
)

# Agent B: With divination vector
agent_b = AgentConfig(
    name="Agent_B",
    control_vector_path="vectors/divination_v1.gguf",
    control_coeff=1.2  # Adjust intensity (try 0.5-2.0)
)
```

### Training Configuration

Key parameters for control vector training:

- **batch_size**: 8 (reduce if OOM)
- **method**: `pca_diff` (recommended) or `pca_center`
- **device**: `cuda`, `mps`, or `cpu`
- **layer_ids**: Default is layers -5 to -18 (last 13 layers)

### Generation Configuration

Key parameters for conversation generation:

- **temperature**: 0.8 (higher = more creative, lower = more deterministic)
- **top_p**: 0.95 (nucleus sampling)
- **max_new_tokens**: 256 (tokens per turn)
- **control_coeff**: 1.2 (control vector strength; 0 = off, higher = stronger effect)

## Contrastive Pairs

The divination axis is defined by 30+ contrastive pairs organized into categories:

- **Divination modalities**: Tarot, I Ching, astrology, runes, etc.
- **Mystical presence**: Oracles, prophets, shamans, angels
- **Sacred geometry**: Mandalas, golden ratio, flower of life
- **Supernatural entities**: Djinn, serpent power, mirrors
- **Cosmic knowledge**: Akashic records, philosopher's stone
- **Divine communication**: The Word, Om, voice in whirlwind
- **Indigenous sacred**: Songlines, ancestor spirits, axis mundi

Each pair contrasts a mystical/divinatory concept with its void/absence counterpart.

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--batch-size` (try 4 or 2)
- Use `--device cpu` (slower but more memory)
- Reduce `--max-tokens` per turn
- Don't use `--save-attention` flag

### Slow Generation

- Use `--device cuda` if available
- Reduce `--turns` for testing
- Use a smaller model (though Mistral-7B is already quite small)

### Model Not Found

Ensure you have internet access for the first run - the model will be downloaded from HuggingFace (about 14GB for Mistral-7B).

### Import Errors

Make sure the `repeng` module is in the parent directory and accessible. The scripts automatically add it to `sys.path`.

## Example Workflow

```bash
# 1. Train control vector (one-time)
cd src
python train_vector.py --device cuda

# 2. Run a test session (50 turns)
cd ../scripts
python run_session.py --turns 50 --coeff 1.2

# 3. Analyze results
cd ../notebooks
jupyter notebook 03_analyze_session.ipynb

# 4. Run production sessions with different coefficients
python ../scripts/run_session.py --turns 200 --coeff 0.8 --session-name "session_low_coeff"
python ../scripts/run_session.py --turns 200 --coeff 1.5 --session-name "session_high_coeff"
python ../scripts/run_session.py --turns 200 --coeff 2.0 --session-name "session_very_high"
```

## API Reference

### `compute_token_metrics(logits, chosen_ids, k=10)`

Compute derived metrics from raw logits.

**Args:**
- `logits`: (num_tokens, vocab_size) tensor
- `chosen_ids`: (num_tokens,) tensor
- `k`: Number of top alternatives to track

**Returns:** Dict with top_k_tokens, top_k_probs, entropy, chosen_prob, chosen_rank

### `ConversationLogger(session_dir, save_attention=False)`

Logger for capturing conversation data.

**Methods:**
- `set_tokenizer(tokenizer)`: Store tokenizer and save vocab
- `log_turn(turn_id, agent, prompt, response, metadata, control_coeff)`: Log a turn
- `finalize(config=None)`: Save manifest and close
- `get_storage_usage()`: Calculate storage usage

### `BackroomEngine(model_name, agent_a_config, agent_b_config, device, layer_ids)`

Main conversation loop manager.

**Methods:**
- `set_agent(agent_name)`: Configure for agent "A" or "B"
- `generate_turn(prompt, agent_name, ...)`: Generate one turn
- `run_session(logger, seed_prompt, num_turns, ...)`: Run full session

## License

See parent project for license information.

## Credits

Part of the MULCURSIO generative video project.

Design document version: 1.0 (January 2026)
