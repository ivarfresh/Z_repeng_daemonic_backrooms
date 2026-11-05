# MULCURSIO RunPod Quick Start

## One-Line Complete Workflow

```bash
# Upload, train, and download in one go
./scripts/runpod_full_workflow.sh
```

## Step-by-Step Workflow

### 1. Upload Project to RunPod

```bash
./scripts/runpod_upload.sh
```

### 2. Train Control Vector

```bash
./scripts/runpod_train.sh
```

**Options:**
```bash
./scripts/runpod_train.sh --batch-size 4    # Reduce if OOM
./scripts/runpod_train.sh --device cpu      # Use CPU (slower)
./scripts/runpod_train.sh --method pca_center
```

### 3. Download Results

```bash
./scripts/runpod_download.sh vectors        # Download trained vectors
./scripts/runpod_download.sh sessions       # Download session data
./scripts/runpod_download.sh all            # Download everything
```

## Interactive Training (via Jupyter)

### Start Jupyter on RunPod

```bash
# SSH to pod
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io

# Start Jupyter
cd ~/mulcursio/notebooks
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Connect from Local Machine

```bash
# In a new terminal, create SSH tunnel
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -L 8888:localhost:8888 sfghgd9eq41tsn-64411155@ssh.runpod.io
```

Open `http://localhost:8888` in browser or connect Cursor to remote kernel, then run:
- `notebooks/00_runpod_train_vector.ipynb` - Training
- `notebooks/02_run_session.ipynb` - Sessions

## Direct SSH Commands

### Connect to Pod

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io
```

### Run Training Manually

```bash
cd ~/mulcursio/src
python train_vector.py --device cuda
```

### Run Session Manually

```bash
cd ~/mulcursio/scripts
python run_session.py --turns 50 --coeff 1.2 --device cuda
```

### Check GPU Status

```bash
nvidia-smi
watch -n 1 nvidia-smi  # Live monitoring
```

## Using tmux for Long-Running Tasks

```bash
# Start tmux session
tmux new -s training

# Run training (will continue if SSH disconnects)
cd ~/mulcursio/src && python train_vector.py --device cuda

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

## Configuration

**RunPod Details:**
- Host: `sfghgd9eq41tsn-64411155@ssh.runpod.io`
- User: `root`
- SSH Key: `~/.ssh/id_ed25519`
- Project Dir: `~/mulcursio`

**Typical Training Time:**
- Model download (first time): ~5-10 min (14GB)
- Training: ~10-20 min
- Session (200 turns): ~2-4 hours

## Troubleshooting

### Connection Failed

```bash
# Test connection
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io "echo 'OK'"

# Check pod is running in RunPod dashboard
```

### Out of Memory

```bash
# Reduce batch size
./scripts/runpod_train.sh --batch-size 4

# Or use CPU (slower)
./scripts/runpod_train.sh --device cpu
```

### Files Not Found

```bash
# Re-upload project
./scripts/runpod_upload.sh
```

### Check What's on Pod

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io "ls -lh ~/mulcursio/"
```

## File Locations

**Local:**
- Project: `/Users/ivar/Desktop/current_projects/AI_projects/Z_repeng/x_daemonic_backrooms/`
- Scripts: `./scripts/`
- Notebooks: `./notebooks/`

**RunPod:**
- Project: `~/mulcursio/`
- Vectors: `~/mulcursio/vectors/`
- Sessions: `~/mulcursio/sessions/`

## Quick Reference

| Task | Command |
|------|---------|
| Full workflow | `./scripts/runpod_full_workflow.sh` |
| Upload only | `./scripts/runpod_upload.sh` |
| Train only | `./scripts/runpod_train.sh` |
| Download vectors | `./scripts/runpod_download.sh vectors` |
| Download sessions | `./scripts/runpod_download.sh sessions` |
| SSH to pod | `ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io` |
| SSH tunnel | `ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -L 8888:localhost:8888 sfghgd9eq41tsn-64411155@ssh.runpod.io` |
| Check GPU | `ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io "nvidia-smi"` |

## Cost Saving Tips

1. **Use spot instances** for long training/sessions (cheaper)
2. **Stop pod** when not in use
3. **Download results** then terminate pod
4. **Test locally first** with small datasets
5. **Use tmux** to avoid SSH timeout issues
