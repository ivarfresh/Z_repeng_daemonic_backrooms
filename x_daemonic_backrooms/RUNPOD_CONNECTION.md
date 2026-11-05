# RunPod Connection Details

## Current Configuration

**SSH Key:** `~/.ssh/id_ed25519`
**Host:** `sfghgd9eq41tsn-64411155@ssh.runpod.io`
**User:** `root`
**Project Directory:** `~/mulcursio`

## Quick Commands

### Connect to Pod

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io
```

### SSH Tunnel for Jupyter

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -L 8888:localhost:8888 sfghgd9eq41tsn-64411155@ssh.runpod.io
```

### Test Connection

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io "echo 'Connected!'"
```

### Check GPU

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 sfghgd9eq41tsn-64411155@ssh.runpod.io "nvidia-smi"
```

## Automated Scripts

All scripts have been configured with the correct connection details:

- `./scripts/test_runpod_connection.sh` - Test connection and environment
- `./scripts/runpod_upload.sh` - Upload project to RunPod
- `./scripts/runpod_train.sh` - Train on RunPod
- `./scripts/runpod_download.sh` - Download results
- `./scripts/runpod_full_workflow.sh` - Complete workflow

## Testing the Connection

Run the connection test script:

```bash
./scripts/test_runpod_connection.sh
```

This will verify:
1. SSH key exists
2. Connection works
3. Python is available
4. CUDA/GPU status
5. repeng library status
6. MULCURSIO project status

## Important Notes

- **Always use `-o IdentitiesOnly=yes`** to ensure the correct SSH key is used
- The pod ID (`sfghgd9eq41tsn-64411155`) is unique to your current pod
- If you create a new pod, update this file and the scripts
- The SSH key (`~/.ssh/id_ed25519`) was generated with:
  ```bash
  ssh-keygen -t ed25519 -C "runpod" -f ~/.ssh/id_ed25519
  ```

## Updating Connection Details

If you need to connect to a different pod:

1. Update the configuration in each script:
   - `scripts/test_runpod_connection.sh`
   - `scripts/runpod_upload.sh`
   - `scripts/runpod_train.sh`
   - `scripts/runpod_download.sh`

2. Look for these variables:
   ```bash
   SSH_KEY="$HOME/.ssh/id_ed25519"
   POD_HOST="YOUR_NEW_POD_ID@ssh.runpod.io"
   POD_USER="root"
   ```

3. Also update:
   - `RUNPOD_SETUP.md`
   - `RUNPOD_QUICKSTART.md`
   - `IMPLEMENTATION_SUMMARY.md`
   - `notebooks/00_runpod_train_vector.ipynb`
   - This file (`RUNPOD_CONNECTION.md`)

## Getting Pod Details

To find your current pod connection details:

1. Log into RunPod dashboard
2. Go to your pod
3. Look for "SSH" or "Connection" section
4. Copy the SSH command (format: `ssh USER@POD_ID@ssh.runpod.io`)
5. Extract the POD_ID part (e.g., `sfghgd9eq41tsn-64411155`)

## Troubleshooting

### "Permission denied (publickey)"

1. Verify SSH key is added to RunPod dashboard
2. Check key permissions: `chmod 600 ~/.ssh/id_ed25519`
3. Test key: `ssh-keygen -y -f ~/.ssh/id_ed25519`

### "Connection refused"

1. Check pod is running in RunPod dashboard
2. Verify pod ID is correct
3. Try without `-o IdentitiesOnly=yes` if other keys exist

### "Host key verification failed"

```bash
ssh-keygen -R sfghgd9eq41tsn-64411155@ssh.runpod.io
```

Then try connecting again and accept the new host key.
