#!/bin/bash
# Download results from RunPod

set -e  # Exit on error

# Configuration
SSH_KEY="$HOME/.ssh/id_ed25519"
POD_HOST="sfghgd9eq41tsn-64411155@ssh.runpod.io"
REMOTE_DIR="/root/mulcursio"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Parse arguments
DOWNLOAD_TYPE="${1:-all}"

echo "======================================"
echo "MULCURSIO RunPod Download"
echo "======================================"
echo "Download type: $DOWNLOAD_TYPE"
echo "Remote:        $POD_HOST:$REMOTE_DIR"
echo "Local dir:     $LOCAL_DIR"
echo "======================================"
echo ""

download_vectors() {
    echo "Downloading trained vectors..."
    mkdir -p "$LOCAL_DIR/vectors"

    # Use tar+ssh for reliable download
    ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
        "cd $REMOTE_DIR && tar czf - vectors/*.gguf vectors/*.json 2>/dev/null || echo 'No files found'" \
        | tar xzf - -C "$LOCAL_DIR" 2>/dev/null

    if [ -d "$LOCAL_DIR/vectors" ] && [ "$(ls -A $LOCAL_DIR/vectors)" ]; then
        echo "✓ Vectors downloaded to: $LOCAL_DIR/vectors/"
    else
        echo "⚠ No vector files found on pod"
    fi
}

download_sessions() {
    echo "Downloading session data..."
    mkdir -p "$LOCAL_DIR/sessions"

    # List available sessions
    echo ""
    echo "Available sessions on RunPod:"
    ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" "ls -1 $REMOTE_DIR/sessions/ 2>/dev/null || echo '(none)'"
    echo ""

    # Ask which session to download
    if [ "$2" ]; then
        SESSION_NAME="$2"
    else
        read -p "Enter session name to download (or 'all' for all sessions): " SESSION_NAME
    fi

    # Use tar+ssh for reliable download
    if [ "$SESSION_NAME" = "all" ]; then
        echo "Downloading all sessions..."
        ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
            "cd $REMOTE_DIR && tar czf - sessions/ 2>/dev/null" \
            | tar xzf - -C "$LOCAL_DIR" 2>/dev/null
    else
        echo "Downloading session: $SESSION_NAME"
        ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
            "cd $REMOTE_DIR && tar czf - sessions/$SESSION_NAME/ 2>/dev/null" \
            | tar xzf - -C "$LOCAL_DIR" 2>/dev/null
    fi

    echo "✓ Sessions downloaded to: $LOCAL_DIR/sessions/"
}

download_logs() {
    echo "Downloading logs..."
    mkdir -p "$LOCAL_DIR/logs"

    # Use tar+ssh for reliable download
    ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" \
        "cd $REMOTE_DIR && tar czf - *.log 2>/dev/null || echo 'No logs found'" \
        | tar xzf - -C "$LOCAL_DIR/logs/" 2>/dev/null

    if [ -d "$LOCAL_DIR/logs" ] && [ "$(ls -A $LOCAL_DIR/logs 2>/dev/null)" ]; then
        echo "✓ Logs downloaded"
    else
        echo "⚠ No log files found on pod"
    fi
}

case "$DOWNLOAD_TYPE" in
    vectors)
        download_vectors
        ;;
    sessions)
        download_sessions "$@"
        ;;
    logs)
        download_logs
        ;;
    all)
        download_vectors
        download_sessions "$@"
        download_logs
        ;;
    *)
        echo "Usage: $0 [vectors|sessions|logs|all] [session_name]"
        echo ""
        echo "Examples:"
        echo "  $0 vectors                              # Download trained vectors"
        echo "  $0 sessions session_20260127_143022     # Download specific session"
        echo "  $0 sessions all                         # Download all sessions"
        echo "  $0 all                                  # Download everything"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "Download complete!"
echo "======================================"
echo ""
echo "Files are in: $LOCAL_DIR"
echo ""
