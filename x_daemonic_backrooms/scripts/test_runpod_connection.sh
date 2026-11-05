#!/bin/bash
# Test RunPod connection and environment

set -e

# Configuration
SSH_KEY="$HOME/.ssh/id_ed25519"
POD_HOST="sfghgd9eq41tsn-64411155@ssh.runpod.io"

echo "======================================"
echo "MULCURSIO RunPod Connection Test"
echo "======================================"
echo ""
echo "Testing connection to: $POD_HOST"
echo ""
echo "Note: 'PTY not supported' warnings are normal for RunPod"
echo ""

# Test 1: SSH Key exists
echo "[1/6] Checking SSH key..."
if [ -f "$SSH_KEY" ]; then
    echo "  ✓ SSH key found: $SSH_KEY"
else
    echo "  ✗ SSH key not found: $SSH_KEY"
    echo ""
    echo "  Please ensure the key exists. If needed, create it with:"
    echo "  ssh-keygen -t ed25519 -C 'your-email@example.com' -f $SSH_KEY"
    exit 1
fi

# Test 2: Connection
echo ""
echo "[2/6] Testing SSH connection..."
if ssh -o IdentitiesOnly=yes -i "$SSH_KEY" -o ConnectTimeout=5 -o BatchMode=yes "$POD_HOST" "echo 'Connected'" &>/dev/null; then
    echo "  ✓ SSH connection successful"
else
    echo "  ✗ Cannot connect to RunPod"
    echo ""
    echo "  Please check:"
    echo "    1. Pod is running in RunPod dashboard"
    echo "    2. Pod host is correct: $POD_HOST"
    echo "    3. SSH key is added to RunPod account"
    exit 1
fi

# Test 3: Python
echo ""
echo "[3/6] Checking Python..."
# Try python3 first (most common on RunPod), then python
PYTHON_VERSION=$(ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "python3 --version 2>&1 || python --version 2>&1" || echo "Not found")
if [[ "$PYTHON_VERSION" == *"Python"* ]]; then
    echo "  ✓ Python: $PYTHON_VERSION"
    # Set Python command for later use
    PYTHON_CMD=$(ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "command -v python3 || command -v python")
    echo "  ✓ Python command: $PYTHON_CMD"
else
    echo "  ✗ Python not found"
    echo "  Note: This may be okay if Python is in a conda/venv environment"
fi

# Test 4: CUDA
echo ""
echo "[4/6] Checking CUDA..."
CUDA_CHECK=$(ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader 2>&1" || echo "Not found")
if [[ "$CUDA_CHECK" == *"Not found"* ]]; then
    echo "  ⚠ CUDA not available (will use CPU)"
else
    echo "  ✓ CUDA available"
    echo "  GPU Info:"
    while IFS= read -r line; do
        echo "    $line"
    done <<< "$CUDA_CHECK"
fi

# Test 5: Check if repeng exists
echo ""
echo "[5/6] Checking repeng library..."
REPENG_CHECK=$(ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "test -d ~/repeng && echo 'found' || echo 'not found'")
if [[ "$REPENG_CHECK" == "found" ]]; then
    echo "  ✓ repeng directory found at ~/repeng"

    # Check if installed (try python3 first, then python)
    REPENG_INSTALLED=$(ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "python3 -c 'import repeng; print(repeng.__version__)' 2>/dev/null || python -c 'import repeng; print(repeng.__version__)' 2>/dev/null || echo 'not installed'")
    if [[ "$REPENG_INSTALLED" != "not installed" ]]; then
        echo "  ✓ repeng installed (version: $REPENG_INSTALLED)"
    else
        echo "  ⚠ repeng not installed (will install during setup)"
    fi
else
    echo "  ⚠ repeng directory not found (will need to upload parent repo)"
fi

# Test 6: Check if mulcursio project exists
echo ""
echo "[6/6] Checking MULCURSIO project..."
MULCURSIO_CHECK=$(ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "test -d ~/mulcursio && echo 'found' || echo 'not found'")
if [[ "$MULCURSIO_CHECK" == "found" ]]; then
    echo "  ✓ MULCURSIO project found at ~/mulcursio"

    # List contents
    echo ""
    echo "  Project structure:"
    ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "cd ~/mulcursio && find . -maxdepth 2 -type d | sort" | while read -r line; do
        echo "    $line"
    done
else
    echo "  ⚠ MULCURSIO project not found"
    echo "    Run: ./scripts/runpod_upload.sh"
fi

# Summary
echo ""
echo "======================================"
echo "Test Summary"
echo "======================================"
echo ""

if [[ "$CUDA_CHECK" != *"Not found"* ]] && [[ "$REPENG_CHECK" == "found" ]]; then
    echo "✓ Environment ready for training!"
    echo ""
    if [[ "$MULCURSIO_CHECK" == "not found" ]]; then
        echo "Next step:"
        echo "  ./scripts/runpod_upload.sh"
    else
        echo "Next steps:"
        echo "  ./scripts/runpod_train.sh       # Train control vector"
        echo "  ./scripts/runpod_full_workflow.sh  # Complete workflow"
    fi
else
    echo "⚠ Environment needs setup"
    echo ""
    echo "Actions needed:"
    [ "$REPENG_CHECK" == "not found" ] && echo "  - Upload repeng library to ~/repeng"
    [ "$CUDA_CHECK" == *"Not found"* ] && echo "  - CUDA not available (training will use CPU)"
    [ "$MULCURSIO_CHECK" == "not found" ] && echo "  - Upload MULCURSIO project: ./scripts/runpod_upload.sh"
fi

echo ""
