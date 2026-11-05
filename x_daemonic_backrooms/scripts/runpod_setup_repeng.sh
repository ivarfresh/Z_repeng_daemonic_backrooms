#!/bin/bash
# Upload and install repeng library on RunPod

set -e

# Configuration
SSH_KEY="$HOME/.ssh/id_ed25519"
POD_HOST="sfghgd9eq41tsn-64411155@ssh.runpod.io"
LOCAL_REPENG_DIR="$(cd "$(dirname "$0")/../.." && pwd)/repeng"

echo "======================================"
echo "MULCURSIO - Setup repeng on RunPod"
echo "======================================"
echo ""
echo "Local repeng:  $LOCAL_REPENG_DIR"
echo "Remote pod:    $POD_HOST"
echo "======================================"
echo ""

# Check if local repeng exists
if [ ! -d "$LOCAL_REPENG_DIR" ]; then
    echo "Error: repeng directory not found at: $LOCAL_REPENG_DIR"
    echo ""
    echo "Expected location: /Users/ivar/Desktop/current_projects/AI_projects/Z_repeng/repeng"
    echo ""
    echo "Please ensure the repeng library is in the parent directory."
    exit 1
fi

echo "✓ Local repeng found"
echo ""

# Upload repeng to pod using tar+ssh (more reliable for RunPod)
echo "Uploading repeng to RunPod..."
echo "  - Packing repeng directory..."
cd "$LOCAL_REPENG_DIR/.."

# Create tar and upload
tar czf - repeng \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude '*.egg-info' \
    2>/dev/null | \
    ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "tar xzf - -C ~/"

if [ $? -eq 0 ]; then
    echo "✓ repeng uploaded successfully"
else
    echo "✗ Upload failed"
    exit 1
fi
echo ""

# Install repeng on pod
echo "Installing repeng on RunPod..."
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" bash <<'INSTALL_EOF' 2>&1 | grep -v "PTY"
# Use python3 if available
PYTHON_CMD=$(command -v python3 || command -v python)
PIP_CMD=$(command -v pip3 || command -v pip)

echo "Using Python: $PYTHON_CMD"
echo "Using pip: $PIP_CMD"

cd ~/repeng
$PIP_CMD install -e . -q 2>&1

# Verify installation
if $PYTHON_CMD -c "import repeng; print('repeng', repeng.__version__)" 2>/dev/null; then
    echo "✓ repeng installed successfully"
else
    echo "✗ repeng installation failed"
    exit 1
fi
INSTALL_EOF

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "repeng is now available on the RunPod instance."
echo ""
echo "Next steps:"
echo "  1. Upload MULCURSIO project: ./scripts/runpod_upload.sh"
echo "  2. Train vector: ./scripts/runpod_train.sh"
echo ""
