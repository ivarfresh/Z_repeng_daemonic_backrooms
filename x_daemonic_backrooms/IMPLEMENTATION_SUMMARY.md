# MULCURSIO Implementation Summary

## What Has Been Implemented

The complete MULCURSIO data pipeline as specified in `mulcursio_data_pipeline_design.md` has been implemented, with full RunPod support added.

## File Structure

```
x_daemonic_backrooms/
├── README.md                          # Main documentation
├── RUNPOD_SETUP.md                    # RunPod detailed guide
├── RUNPOD_QUICKSTART.md               # RunPod quick reference
├── IMPLEMENTATION_SUMMARY.md          # This file
├── mulcursio_data_pipeline_design.md  # Original design doc
├── requirements.txt                   # Python dependencies
│
├── data/
│   └── divination_pairs.json          # 30+ contrastive pairs
│
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── metrics.py                     # Token metrics (entropy, top-k, etc.)
│   ├── logger.py                      # ConversationLogger class
│   ├── utils.py                       # Utility functions
│   ├── train_vector.py                # Control vector training
│   └── backroom_engine.py             # BackroomEngine class
│
├── scripts/
│   ├── run_session.py                 # CLI session runner
│   ├── export_for_viz.py              # Data export utilities
│   ├── runpod_upload.sh               # Upload to RunPod
│   ├── runpod_train.sh                # Train on RunPod
│   ├── runpod_download.sh             # Download from RunPod
│   └── runpod_full_workflow.sh        # Complete workflow
│
├── notebooks/
│   ├── 00_runpod_train_vector.ipynb   # RunPod training (remote kernel)
│   ├── 01_train_vector.ipynb          # Local training
│   ├── 02_run_session.ipynb           # Session running
│   └── 03_analyze_session.ipynb       # Data analysis
│
├── vectors/                           # Trained control vectors
└── sessions/                          # Generated session data
```

## Core Functionality

### 1. Control Vector Training (`src/train_vector.py`)

**Features:**
- Loads contrastive pairs from JSON
- Supports dataset expansion with suffixes
- Trains using repeng's ControlVector
- Exports to GGUF format
- CLI and notebook interfaces

**Usage:**
```bash
# Local training
python src/train_vector.py --device cuda

# RunPod training
./scripts/runpod_train.sh
```

### 2. Backroom Engine (`src/backroom_engine.py`)

**Features:**
- Manages conversation between two agents
- Alternating baseline/controlled responses
- Comprehensive metadata capture
- Per-token logits and metrics

**Key Classes:**
- `AgentConfig` - Agent configuration
- `BackroomEngine` - Main conversation loop
- `generate_with_logging()` - Generation with full metadata

### 3. Logging System (`src/logger.py`)

**Features:**
- Per-turn logging with all metadata
- Structured session directories
- Human-readable transcripts
- Compressed metrics storage
- Optional attention weights

**Key Classes:**
- `TurnLog` - Single turn data
- `ConversationLogger` - Session logger

### 4. Metrics Computation (`src/metrics.py`)

**Features:**
- Entropy calculation
- Top-k token alternatives
- Probability tracking
- Token ranking

**Functions:**
- `compute_token_metrics()` - Main metrics function
- `analyze_token_distribution()` - Distribution analysis

## RunPod Integration

### Automated Scripts

1. **`runpod_upload.sh`** - Uploads project to pod
   - Uses rsync for efficient transfer
   - Creates necessary directories
   - Preserves file permissions

2. **`runpod_train.sh`** - Runs training on pod
   - Checks dependencies
   - Configurable parameters
   - Real-time progress output

3. **`runpod_download.sh`** - Downloads results
   - Vectors, sessions, or all
   - Selective downloads
   - Resume capability

4. **`runpod_full_workflow.sh`** - Complete automation
   - Upload → Train → Download
   - Single command execution
   - Progress tracking

### Interactive Jupyter

**Notebook:** `00_runpod_train_vector.ipynb`

**Features:**
- Connects to remote RunPod kernel
- Step-by-step training
- Interactive testing
- Result verification

**Setup:**
```bash
# On RunPod
cd ~/mulcursio/notebooks
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# On local machine
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -L 8888:localhost:8888 sfghgd9eq41tsn-64411155@ssh.runpod.io

# Connect Cursor to http://localhost:8888
```

## Usage Workflows

### Workflow 1: RunPod Training (Recommended)

```bash
# 1. Complete automated workflow
./scripts/runpod_full_workflow.sh

# 2. Test locally
jupyter notebook notebooks/01_train_vector.ipynb

# 3. Run session locally or on RunPod
python scripts/run_session.py --turns 50

# 4. Analyze
jupyter notebook notebooks/03_analyze_session.ipynb
```

### Workflow 2: Local Training

```bash
# 1. Train locally
cd src
python train_vector.py --device cuda

# 2. Run session
cd ../scripts
python run_session.py --turns 50

# 3. Analyze
cd ../notebooks
jupyter notebook 03_analyze_session.ipynb
```

### Workflow 3: Interactive (Jupyter)

```bash
# 1. Train interactively
jupyter notebook notebooks/01_train_vector.ipynb

# 2. Run session interactively
jupyter notebook notebooks/02_run_session.ipynb

# 3. Analyze
jupyter notebook notebooks/03_analyze_session.ipynb
```

## Data Output

### Session Directory Structure

```
sessions/session_20260127_143022/
├── manifest.json           # Metadata + turn index
├── conversation.txt        # Human-readable transcript
├── vocab.json              # Token mapping
├── config.json             # Generation config
├── logits/
│   ├── turn_0000.npy       # Raw logits
│   └── ...
├── metrics/
│   ├── turn_0000.npz       # Derived metrics
│   └── ...
└── attention/              # Optional
    └── ...
```

### Metrics Per Turn

Each `metrics/turn_XXXX.npz` contains:
- `output_ids` - Token IDs
- `top_k_tokens` - Top 10 alternatives
- `top_k_probs` - Their probabilities
- `entropy` - Per-token entropy
- `chosen_prob` - Probability of chosen token
- `chosen_rank` - Rank in distribution

## Analysis Tools

### Export for Visualization

```bash
# Export session summary
python scripts/export_for_viz.py summary sessions/session_XXXX

# Export specific turn
python scripts/export_for_viz.py turn sessions/session_XXXX --turn-id 42

# Export highlights
python scripts/export_for_viz.py highlights sessions/session_XXXX
```

### Jupyter Analysis

`notebooks/03_analyze_session.ipynb` provides:
- Entropy over time plots
- Confidence tracking
- Agent comparison
- Token-level inspection
- Top-k alternatives visualization
- Highlight extraction

## Configuration

### Training Config

```python
CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
    "device": "cuda",
    "batch_size": 8,
    "method": "pca_diff",
    "use_expansion": False
}
```

### Agent Config

```python
# Baseline agent
AgentConfig(name="A", control_vector_path=None, control_coeff=0.0)

# Divination agent
AgentConfig(name="B", control_vector_path="vectors/divination_v1.gguf", control_coeff=1.2)
```

### Session Config

```python
{
    "num_turns": 200,
    "max_tokens": 256,
    "temperature": 0.8,
    "save_attention": False
}
```

## Key Parameters

### Control Coefficient

Adjusts control vector strength:
- `0.0` - No effect (baseline)
- `0.5-1.0` - Subtle effect
- `1.2` - Recommended default
- `1.5-2.0` - Strong effect
- `>2.0` - Very strong (may degrade coherence)

### Batch Size

Training batch size:
- `8` - Default (works on most GPUs)
- `4` - Reduce if OOM
- `2` - Minimal memory
- `16+` - If you have large GPU memory

### Temperature

Sampling temperature:
- `0.7` - More focused/deterministic
- `0.8` - Recommended balance
- `1.0` - More creative/diverse
- `1.2+` - Very creative (may be incoherent)

## Storage Requirements

### Training
- Model download: ~14GB (one-time)
- Training output: ~100MB

### Sessions (200 turns)
- Without attention: ~800MB
- With attention: ~20GB

### Recommendations
- Save attention only for highlights
- Use compression where possible
- Delete old sessions after analysis

## Performance

### Training Time (RTX 4090)
- Model download: ~5-10 min (first time)
- Training 30 pairs: ~10-20 min

### Session Time
- ~0.5-2 min per turn (depends on token length)
- 200 turns: ~2-4 hours

### Cost (RunPod)
- Training: $0.50 - $2.00 (one-time)
- Session (200 turns): $2.00 - $8.00
- Use spot instances for savings

## Next Steps

1. **Train Control Vector**
   ```bash
   ./scripts/runpod_full_workflow.sh
   ```

2. **Run Test Session**
   ```bash
   python scripts/run_session.py --turns 10 --coeff 1.2
   ```

3. **Analyze Results**
   ```bash
   jupyter notebook notebooks/03_analyze_session.ipynb
   ```

4. **Run Production Sessions**
   ```bash
   # Try different coefficients
   python scripts/run_session.py --turns 200 --coeff 0.8 --session-name low
   python scripts/run_session.py --turns 200 --coeff 1.5 --session-name high
   python scripts/run_session.py --turns 200 --coeff 2.0 --session-name extreme
   ```

5. **Export for Visualization**
   ```bash
   python scripts/export_for_viz.py highlights sessions/session_NAME
   ```

## Troubleshooting

See README.md and RUNPOD_SETUP.md for detailed troubleshooting.

**Common issues:**
- OOM: Reduce batch size or use CPU
- Connection: Check pod status and SSH key
- Import errors: Ensure repeng is installed
- Slow generation: Use GPU, reduce tokens

## Documentation

- `README.md` - Main project documentation
- `RUNPOD_SETUP.md` - RunPod detailed guide
- `RUNPOD_QUICKSTART.md` - Quick reference
- `mulcursio_data_pipeline_design.md` - Original design spec
- Notebooks - Interactive documentation with examples

## Support

For issues:
1. Check README.md troubleshooting section
2. Review RUNPOD_SETUP.md for RunPod issues
3. Check notebook examples for usage patterns
4. Review design doc for architectural questions

## Implementation Status

✅ **Complete** - All components from design doc implemented
✅ **RunPod Integration** - Full automation scripts
✅ **Documentation** - Comprehensive guides
✅ **Examples** - Interactive notebooks
✅ **Analysis Tools** - Visualization exports

Ready for production use!
