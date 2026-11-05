#!/bin/bash
# Connect to RunPod and run training

set -e  # Exit on error

# Configuration
SSH_KEY="$HOME/.ssh/id_ed25519"
POD_HOST="sfghgd9eq41tsn-64411155@ssh.runpod.io"
REMOTE_DIR="/root/mulcursio"

# Parse command line arguments
BATCH_SIZE=8
DEVICE="cuda"
METHOD="pca_diff"

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--batch-size SIZE] [--device DEVICE] [--method METHOD]"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "MULCURSIO Training on RunPod"
echo "======================================"
echo "Pod:        $POD_HOST"
echo "Device:     $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Method:     $METHOD"
echo "======================================"
echo ""

# Check connection
echo "Checking connection..."
if ! ssh -o IdentitiesOnly=yes -i "$SSH_KEY" -o ConnectTimeout=5 "$POD_HOST" "echo 'Connected'" &>/dev/null; then
    echo "Error: Cannot connect to RunPod at $POD_HOST"
    echo "Please check:"
    echo "  1. Pod is running"
    echo "  2. IP address is correct"
    echo "  3. SSH key is correct"
    exit 1
fi
echo "✓ Connected"
echo ""

# Check if project exists
echo "Checking project directory..."
if ! ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "test -d $REMOTE_DIR" &>/dev/null; then
    echo "Error: Project directory not found on RunPod"
    echo "Please run ./scripts/runpod_upload.sh first"
    exit 1
fi
echo "✓ Project found"
echo ""

# Check if repeng is installed
echo "Checking dependencies..."
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" bash <<'EOF'
# Use python3 if available, otherwise python
PYTHON_CMD=$(command -v python3 || command -v python)

if ! $PYTHON_CMD -c "import repeng" 2>/dev/null; then
    echo "Installing repeng..."
    cd /root/repeng && pip3 install -e . -q || pip install -e . -q
fi
if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
    echo "Installing dependencies..."
    cd /root/mulcursio && pip3 install -r requirements.txt -q || pip install -r requirements.txt -q
fi
echo "✓ Dependencies ready"
EOF
echo ""

# Run training
echo "======================================"
echo "Starting training..."
echo "======================================"
echo ""
echo "This will take 10-20 minutes."
echo "Press Ctrl+C to cancel."
echo ""

ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" bash <<EOF
cd $REMOTE_DIR/src

# Use python3 if available, otherwise python
PYTHON_CMD=\$(command -v python3 || command -v python)

echo "Running training with:"
echo "  Python: \$PYTHON_CMD"
echo "  Model: mistralai/Mistral-7B-Instruct-v0.1"
echo "  Pairs: ../data/divination_pairs.json"
echo "  Output: ../vectors/divination_v1.gguf"
echo "  Device: $DEVICE"
echo "  Batch size: $BATCH_SIZE"
echo "  Method: $METHOD"
echo ""

\$PYTHON_CMD train_vector.py \
    --model mistralai/Mistral-7B-Instruct-v0.1 \
    --pairs ../data/divination_pairs.json \
    --output ../vectors/divination_v1.gguf \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --method $METHOD

echo ""
echo "Training complete! Vector saved to: $REMOTE_DIR/vectors/divination_v1.gguf"
echo ""
echo "To download the vector, run:"
echo "  ./scripts/runpod_download.sh vectors"
EOF

echo ""
echo "======================================"
echo "Training complete!"
echo "======================================"
echo ""
echo "To download the trained vector:"
echo "  ./scripts/runpod_download.sh vectors"
echo ""
