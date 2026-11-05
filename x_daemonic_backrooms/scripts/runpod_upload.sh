#!/bin/bash
# Upload MULCURSIO project to RunPod

set -e  # Exit on error

# Configuration
SSH_KEY="$HOME/.ssh/id_ed25519"
POD_HOST="sfghgd9eq41tsn-64411155@ssh.runpod.io"
REMOTE_DIR="/root/mulcursio"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "======================================"
echo "MULCURSIO RunPod Upload"
echo "======================================"
echo "Local dir:  $LOCAL_DIR"
echo "Remote:     $POD_HOST:$REMOTE_DIR"
echo "======================================"

# Create remote directory
echo "Creating remote directory..."
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "mkdir -p $REMOTE_DIR"

# Upload project using tar+ssh (more reliable for RunPod)
echo "Uploading project files..."
echo "  - Packing files..."
cd "$LOCAL_DIR"

# Create tar archive and upload
tar czf - \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    src \
    data \
    notebooks \
    scripts \
    requirements.txt \
    RUNPOD_SETUP.md \
    | \
    ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
    "mkdir -p $REMOTE_DIR && tar xzf - -C $REMOTE_DIR"

if [ $? -eq 0 ]; then
    echo "✓ Project files uploaded successfully"
else
    echo "✗ Upload failed"
    exit 1
fi

# Create remote directories for outputs
echo "Creating output directories..."
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "mkdir -p $REMOTE_DIR/vectors $REMOTE_DIR/sessions"

echo ""
echo "======================================"
echo "Upload complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Connect to pod:"
echo "     ssh -o IdentitiesOnly=yes -i $SSH_KEY $POD_HOST"
echo ""
echo "  2. Install dependencies:"
echo "     cd $REMOTE_DIR"
echo "     pip install -r requirements.txt"
echo ""
echo "  3. Train vector:"
echo "     cd src && python train_vector.py --device cuda"
echo ""
echo "Or use the automated script:"
echo "  ./scripts/runpod_train.sh"
echo ""
