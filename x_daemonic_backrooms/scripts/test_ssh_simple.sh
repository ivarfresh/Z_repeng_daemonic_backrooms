#!/bin/bash
# Simple SSH test

SSH_KEY="/Users/ivar/.ssh/id_ed25519"
POD_HOST="sfghgd9eq41tsn-64411155@ssh.runpod.io"

echo "Testing SSH connection..."
echo ""

# Test 1: Simple echo
echo "Test 1: Echo command"
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" 'echo "Hello from pod"' 2>&1

echo ""
echo "Test 2: PWD"
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" 'pwd' 2>&1

echo ""
echo "Test 3: List home"
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" 'ls -1 ~/' 2>&1

echo ""
echo "Test 4: Check mulcursio"
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "$POD_HOST" 'test -d ~/mulcursio && echo "EXISTS" || echo "NOT_FOUND"' 2>&1

echo ""
echo "Done"
