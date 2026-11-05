#!/bin/bash
# Verify RunPod setup - Simple version

SSH_KEY="$HOME/.ssh/id_ed25519"
POD_HOST="sfghgd9eq41tsn-64411155@ssh.runpod.io"

echo "======================================"
echo "Verifying RunPod Setup"
echo "======================================"
echo ""

echo "1. Checking directories..."
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
    'test -d ~/repeng && echo "  ✓ ~/repeng exists" || echo "  ✗ ~/repeng missing"' 2>&1 | grep -v PTY

ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
    'test -d ~/mulcursio && echo "  ✓ ~/mulcursio exists" || echo "  ✗ ~/mulcursio missing"' 2>&1 | grep -v PTY

echo ""
echo "2. Checking Python..."
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
    'python3 --version 2>&1 || python --version 2>&1' 2>&1 | grep -E "Python|not found" | head -1

echo ""
echo "3. Checking repeng..."
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
    'python3 -c "import repeng; print(\"  ✓ repeng\", repeng.__version__)" 2>&1 || echo "  ✗ repeng not installed"' 2>&1 | grep -v PTY

echo ""
echo "4. Checking torch..."
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
    'python3 -c "import torch; print(\"  ✓ torch\", torch.__version__)" 2>&1 || echo "  ✗ torch not installed"' 2>&1 | grep -v PTY

echo ""
echo "5. Checking CUDA..."
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
    'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1' 2>&1 | grep -v PTY | sed 's/^/  ✓ GPU: /'

echo ""
echo "6. Project files..."
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
    'ls -1 ~/mulcursio 2>&1 | head -10' 2>&1 | grep -v PTY | sed 's/^/  /'

echo ""
echo "======================================"
echo "Verification complete"
echo "======================================"
echo ""
echo "Next steps:"
echo "  - If repeng ✓: run ./scripts/runpod_upload.sh"
echo "  - If mulcursio ✓: run ./scripts/runpod_train.sh"
