# Acid Trip Notebook - Complete Setup Guide

This guide will walk you through setting up and running the `acid_trip.ipynb` notebook on a PrimeIntellect compute pod.

## Prerequisites

- PrimeIntellect account with GPU pod access
- SSH key pair generated
- Local machine with SSH client

## Step 1: Generate SSH Key (if not already done)

```bash
# Generate SSH key pair
ssh-keygen -t ed25519 -C "your-email@example.com" -f ~/.ssh/primeintellect_ed25519

# Add to SSH agent
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/primeintellect_ed25519

# Display public key to add to PrimeIntellect
cat ~/.ssh/primeintellect_ed25519.pub
```

## Step 2: Set Up PrimeIntellect Pod

1. **Log into PrimeIntellect Dashboard**
   - Go to [PrimeIntellect.com](https://primeintellect.com)
   - Navigate to "My Pods" or "Compute" section

2. **Add SSH Key**
   - Find "SSH Keys" or "Access Keys" section
   - Add your public key (from step 1)
   - Wait for approval (usually instant)

3. **Create/Start Pod**
   - Create a new GPU pod or start existing one
   - Wait for status to change from "Pending" to "Running"
   - Note the IP address and username (usually `ubuntu` or `root`)

## Step 3: Upload Project Files

```bash
# Upload the repeng project to your pod
scp -i ~/.ssh/primeintellect_ed25519 -r /path/to/Z_repeng/repeng/ ubuntu@YOUR_POD_IP:~/

# Upload data files (if not included in repeng folder)
scp -i ~/.ssh/primeintellect_ed25519 /path/to/data/all_truncated_outputs.json ubuntu@YOUR_POD_IP:~/repeng/notebooks/data/
scp -i ~/.ssh/primeintellect_ed25519 /path/to/data/true_facts.json ubuntu@YOUR_POD_IP:~/repeng/notebooks/data/
```

## Step 4: Connect to Pod and Set Up Environment

```bash
# SSH into your pod
ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@YOUR_POD_IP

# Navigate to project directory
cd ~/repeng

# Install repeng package and dependencies
pip install -e .

# Install additional required packages
pip install jupyterlab ipykernel

# Navigate to notebooks directory
cd notebooks

# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Step 5: Set Up Local Development Environment

### Option A: SSH Tunneling (Recommended)

```bash
# In a new terminal on your local machine
ssh -i ~/.ssh/primeintellect_ed25519 -L 8888:localhost:8888 ubuntu@YOUR_POD_IP
```

### Option B: Direct Access

Use the Jupyter URL provided in the pod terminal:
```
http://YOUR_POD_IP:8888/lab?token=YOUR_TOKEN
```

## Step 6: Connect Cursor to Remote Kernel

1. **Open the notebook in Cursor**
   - Open `acid_trip.ipynb` in Cursor

2. **Connect to remote kernel**
   - Click kernel selector (top right)
   - Select "Select Another Kernel" → "Existing Jupyter Server"
   - Enter: `http://localhost:8888` (if using SSH tunnel)
   - Enter token from Jupyter Lab output

3. **Alternative: SSH Interpreter**
   - Command Palette (Cmd+Shift+P)
   - "Python: Select Interpreter"
   - Choose "SSH" → `ubuntu@YOUR_POD_IP`
   - Use key: `~/.ssh/primeintellect_ed25519`

## Step 7: Authenticate with Hugging Face

The notebook uses gated models that require authentication:

1. **Get Hugging Face Token**
   - Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Create new token with "Read" permissions

2. **Add authentication cell to notebook**
   ```python
   from huggingface_hub import login
   login()  # Enter your token when prompted
   ```

## Step 8: Run the Notebook

1. **Install dependencies** (first cell)
   ```python
   %pip uninstall -y huggingface_hub pytorch transformers datasets sentencepiece accelerate bitsandbytes peft
   %pip install huggingface_hub torch transformers datasets sentencepiece accelerate bitsandbytes peft
   ```

2. **Restart kernel** after installation

3. **Run cells in order** to:
   - Load Mistral-7B-Instruct model
   - Create various control vectors (happy, honest, trippy, lazy, etc.)
   - Generate text with different personality traits

## Expected Results

The notebook will create control vectors for various personality traits:
- **Happy/Sad**: Emotional responses
- **Honest/Untruthful**: Truthfulness in responses  
- **Trippy/Sober**: Psychedelic vs normal responses
- **Lazy/Hardworking**: Response length and effort
- **Self-aware/Unaware**: Self-knowledge level
- **Creative/Uncreative**: Creativity in writing
- **Future/Past**: Temporal perspective
- **Dealership-focused**: Car dealership responses

## Troubleshooting

### Connection Issues
- Verify pod IP address and username
- Check SSH key is properly added to PrimeIntellect
- Ensure pod is in "Running" state

### Model Access Issues
- Request access to Mistral-7B-Instruct on Hugging Face
- Authenticate with `huggingface-cli login`
- Check token has proper permissions

### Kernel Issues
- Restart kernel after installing packages
- Verify SSH tunnel is active
- Check Jupyter Lab is running on pod

### Memory Issues
- Monitor GPU usage with `nvidia-smi`
- Reduce batch sizes if needed
- Use smaller model variants if necessary

## File Structure

```
Z_repeng/repeng/
├── notebooks/
│   ├── acid_trip.ipynb
│   ├── emotion.ipynb
│   └── data/
│       ├── all_truncated_outputs.json
│       └── true_facts.json
├── repeng/
│   └── __init__.py
└── pyproject.toml
```

## Quick Start Commands

```bash
# 1. Connect to pod
ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@YOUR_POD_IP

# 2. Start Jupyter
cd ~/repeng/notebooks && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# 3. SSH tunnel (new terminal)
ssh -i ~/.ssh/primeintellect_ed25519 -L 8888:localhost:8888 ubuntu@YOUR_POD_IP

# 4. Open http://localhost:8888 in browser or connect via Cursor
```

This setup allows you to run the notebook with full GPU acceleration while developing in your local Cursor environment!
