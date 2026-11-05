# Running Acid Trip Notebook on Existing Pod

This guide is for when you want to run the `acid_trip.ipynb` notebook on a PrimeIntellect pod you've already set up.

## Prerequisites

- Existing PrimeIntellect pod that's already configured
- SSH access to the pod
- Project files already uploaded

## Step 1: Connect to Your Existing Pod

```bash
# SSH into your existing pod
ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@154.54.100.122
```

## Step 2: Start Jupyter Lab

```bash
# Navigate to notebooks directory
cd ~/repeng/notebooks

# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Step 3: Set Up SSH Tunnel (Local Machine)

In a **new terminal** on your local machine:

```bash
# Create SSH tunnel
ssh -i ~/.ssh/primeintellect_ed25519 -L 8888:localhost:8888 ubuntu@154.54.100.122
```

## Step 4: Connect Cursor to Remote Kernel

1. **Open the notebook in Cursor**
   - Open `acid_trip.ipynb` in Cursor

2. **Connect to remote kernel**
   - Click kernel selector (top right)
   - Select "Select Another Kernel" → "Existing Jupyter Server"
   - Enter: http://localhost:8888
   - Enter token from Jupyter Lab output

## Step 5: Run the Notebook

1. **Install/Update dependencies** (if needed)
   ```python
   %pip install huggingface_hub torch transformers datasets sentencepiece accelerate bitsandbytes peft
   ```

2. **Authenticate with Hugging Face** (if needed)
   ```python
   from huggingface_hub import login
   login()  # Enter your token when prompted
   ```

3. **Run cells in order** to create control vectors and generate text

## Quick Commands

```bash
# 1. Connect to pod
ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@YOUR_POD_IP

# 2. Start Jupyter
cd ~/repeng/notebooks && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# 3. SSH tunnel (new terminal)
ssh -i ~/.ssh/primeintellect_ed25519 -L 8888:localhost:8888 ubuntu@YOUR_POD_IP
```

## Troubleshooting

### Jupyter Not Starting
- Check if port 8888 is already in use: `lsof -i :8888`
- Kill existing Jupyter processes: `pkill -f jupyter`
- Try a different port: `jupyter lab --ip=0.0.0.0 --port=8889 --no-browser --allow-root`

### Connection Issues
- Verify SSH tunnel is active
- Check Jupyter Lab is running on pod
- Restart kernel in Cursor if needed

### Model Issues
- Ensure Hugging Face authentication is working
- Check GPU availability: `nvidia-smi`
- Restart kernel after installing packages

## Notes

- Keep the SSH session open while using Jupyter
- The pod will maintain your installed packages between sessions
- Control vectors are trained fresh each time you run the notebook
- GPU memory is shared, so close other processes if needed
