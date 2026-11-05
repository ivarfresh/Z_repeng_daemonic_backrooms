# MULCURSIO Training on RunPod

This guide walks through training the divination control vector on a RunPod GPU instance.

## Prerequisites

- RunPod account with GPU pod access
- SSH key (already configured): `~/.ssh/id_ed25519`
- Active pod (Host: `sfghgd9eq41tsn-64411155@ssh.runpod.io`)

## Quick Start

### 1. Upload Project to RunPod

```bash
# From the x_daemonic_backrooms directory
cd /Users/ivar/Desktop/current_projects/AI_projects/Z_repeng/x_daemonic_backrooms

# Upload entire project
./scripts/runpod_upload.sh
```

This will upload:
- All source code (`src/`)
- Data files (`data/divination_pairs.json`)
- Training scripts
- Notebooks

### 2. Connect and Set Up Environment

```bash
# SSH into pod
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io

# Navigate to project
cd ~/mulcursio

# Install dependencies
pip install torch transformers accelerate numpy gguf

# Install repeng (from parent repo)
cd ~/repeng
pip install -e .
cd ~/mulcursio
```

### 3. Run Training

**Option A: Direct Script Execution**

```bash
# On the pod
cd ~/mulcursio/src
python train_vector.py \
    --model mistralai/Mistral-7B-Instruct-v0.1 \
    --pairs ../data/divination_pairs.json \
    --output ../vectors/divination_v1.gguf \
    --device cuda \
    --batch-size 8
```

**Option B: Via Jupyter Notebook**

```bash
# On the pod
cd ~/mulcursio/notebooks
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then in a **new local terminal**:

```bash
# Create SSH tunnel
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -L 8888:localhost:8888 sfghgd9eq41tsn-64411155@ssh.runpod.io
```

Open `http://localhost:8888` in your browser or connect Cursor to the remote kernel, then run `01_train_vector.ipynb`.

### 4. Download Trained Vector

```bash
# From your local machine
scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 \
    sfghgd9eq41tsn-64411155@ssh.runpod.io:~/mulcursio/vectors/divination_v1.gguf \
    ./vectors/
```

## Running Sessions on RunPod

You can also run backroom conversation sessions on RunPod for faster generation.

### Upload and Run

```bash
# SSH to pod
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io

# Navigate to project
cd ~/mulcursio/scripts

# Run session
python run_session.py \
    --turns 200 \
    --coeff 1.2 \
    --device cuda \
    --output-dir ~/mulcursio/sessions
```

### Download Session Data

```bash
# From your local machine
scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -r \
    sfghgd9eq41tsn-64411155@ssh.runpod.io:~/mulcursio/sessions/session_YYYYMMDD_HHMMSS \
    ./sessions/
```

## Automated Workflow Scripts

### `scripts/runpod_upload.sh`

Uploads all necessary files to RunPod.

```bash
./scripts/runpod_upload.sh
```

### `scripts/runpod_train.sh`

Connects to RunPod and runs training.

```bash
./scripts/runpod_train.sh
```

### `scripts/runpod_download.sh`

Downloads trained vectors and session data.

```bash
./scripts/runpod_download.sh
```

## Tips

### Monitor Training Progress

```bash
# On the pod
watch -n 2 nvidia-smi  # Monitor GPU usage

# In Python script, output is shown in real-time
```

### Keep Session Alive

Use `tmux` or `screen` to keep processes running if SSH disconnects:

```bash
# On the pod
tmux new -s training

# Run training inside tmux
python src/train_vector.py --device cuda

# Detach with: Ctrl+B, then D
# Reattach later with: tmux attach -t training
```

### Storage Management

Check available space before large operations:

```bash
# On the pod
df -h
du -sh ~/mulcursio/sessions/*  # Check session sizes
```

## Troubleshooting

### Connection Issues

```bash
# Test SSH connection
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io "echo 'Connected!'"
```

### CUDA Out of Memory

```bash
# Reduce batch size
python train_vector.py --batch-size 4  # or 2

# Or use CPU (slower)
python train_vector.py --device cpu
```

### Model Download Issues

```bash
# Authenticate with Hugging Face
huggingface-cli login

# Or set token in environment
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

### File Upload Failures

```bash
# Use rsync for resume capability
rsync -avz -e "ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519" \
    ./ sfghgd9eq41tsn-64411155@ssh.runpod.io:~/mulcursio/ \
    --exclude sessions --exclude vectors --exclude .git
```

## Quick Reference

```bash
# Connect to pod
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io

# Upload project
./scripts/runpod_upload.sh

# Start training
./scripts/runpod_train.sh

# Download results
./scripts/runpod_download.sh

# Set up SSH tunnel for Jupyter
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -L 8888:localhost:8888 sfghgd9eq41tsn-64411155@ssh.runpod.io
```

## File Locations on RunPod

```
~/repeng/                      # Parent repeng library
~/mulcursio/                   # MULCURSIO project
├── data/
│   └── divination_pairs.json
├── src/
│   ├── train_vector.py
│   └── ...
├── vectors/
│   └── divination_v1.gguf     # Trained vector (download this)
├── sessions/
│   └── session_*/             # Session data (download for analysis)
└── notebooks/
    └── *.ipynb
```

## Expected Training Time

On a typical RunPod GPU (e.g., RTX 4090):

- **Training time**: ~10-20 minutes for 30+ pairs
- **Model download**: ~5-10 minutes (first time only, ~14GB)
- **Session generation**: ~1-2 minutes per turn (depends on token length)

## Cost Estimation

RunPod pricing varies by GPU type. For a typical session:

- **Training**: $0.50 - $2.00 (one-time, 20 minutes)
- **200-turn session**: $2.00 - $8.00 (4-8 hours)
- **Storage**: Minimal (sessions are ~1GB each)

Use spot instances for significant savings on long-running sessions.
