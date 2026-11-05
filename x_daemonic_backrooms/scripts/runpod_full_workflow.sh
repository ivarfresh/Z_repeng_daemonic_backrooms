#!/bin/bash
# Complete workflow: Upload, train, and download

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "======================================"
echo "MULCURSIO Full RunPod Workflow"
echo "======================================"
echo ""
echo "This script will:"
echo "  1. Upload project to RunPod"
echo "  2. Run training"
echo "  3. Download trained vector"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Step 1: Upload
echo ""
echo "======================================"
echo "STEP 1: Uploading to RunPod"
echo "======================================"
"$SCRIPT_DIR/runpod_upload.sh"

# Step 2: Train
echo ""
echo "======================================"
echo "STEP 2: Training Control Vector"
echo "======================================"
"$SCRIPT_DIR/runpod_train.sh" "$@"

# Step 3: Download
echo ""
echo "======================================"
echo "STEP 3: Downloading Results"
echo "======================================"
"$SCRIPT_DIR/runpod_download.sh" vectors

echo ""
echo "======================================"
echo "Workflow Complete!"
echo "======================================"
echo ""
echo "Trained vector is ready at: vectors/divination_v1.gguf"
echo ""
echo "Next steps:"
echo "  1. Test the vector locally with notebook 01_train_vector.ipynb"
echo "  2. Run a backroom session with scripts/run_session.py"
echo "  3. Analyze results with notebook 03_analyze_session.ipynb"
echo ""
