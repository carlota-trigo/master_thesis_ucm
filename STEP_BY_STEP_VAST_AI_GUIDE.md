# Complete Step-by-Step Guide: Training Self-Supervised Learning on Vast.ai

This guide will take you from logging into vast.ai to having a fully trained self-supervised model.

## 🚀 Phase 1: Initial Setup (Local Machine)

### Step 1: Prepare Your Local Environment

```bash
# 1. Navigate to your project directory
cd /path/to/your/master_thesis_ucm

# 2. Run the automated setup
python run_vast_setup.py

# 3. Verify setup completed successfully
ls -la
# You should see: requirements_ssl.txt, vast_training_config.yaml, scripts/ folder
```

### Step 2: Configure Your Settings

```bash
# Edit the configuration file
nano vast_training_config.yaml

# Key settings to configure:
# - vast_ai.api_key: Your vast.ai API key (optional)
# - monitoring.wandb.project: Your wandb project name
# - data.local_data_path: Path to your dataset
```

**Important Settings to Update:**
```yaml
monitoring:
  wandb:
    enabled: true
    project: "ssl-dermatology-yourname"  # Change this
    entity: "your-wandb-username"        # Your wandb username

data:
  local_data_path: "../data"  # Path to your dataset
```

## 🌐 Phase 2: Vast.ai Account Setup

### Step 3: Create/Login to Vast.ai Account

1. **Go to vast.ai website:**
   - Visit: https://vast.ai/
   - Click "Sign Up" (if new user) or "Login" (if existing)

2. **Create Account:**
   - Use your email address
   - Create a strong password
   - Verify your email address

3. **Add Payment Method:**
   - Click on your profile/account settings
   - Go to "Billing" or "Payment"
   - Add a credit card or PayPal account
   - **Minimum deposit**: $5-10 recommended

### Step 4: Generate SSH Key (Recommended)

```bash
# On your local machine, generate SSH key
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"

# When prompted:
# - Press Enter for default file location (~/.ssh/id_rsa)
# - Enter a passphrase (optional but recommended)
# - Confirm passphrase

# Copy your public key
cat ~/.ssh/id_rsa.pub
# Copy this entire output - you'll need it for vast.ai
```

## 🖥️ Phase 3: Launch Vast.ai Instance

### Step 5: Create GPU Instance

1. **Go to vast.ai Dashboard:**
   - Login to vast.ai
   - Click "Create" or "Launch Instance"

2. **Configure Instance Settings:**

   **Template Selection:**
   - Choose: `CUDA:Devel-Ubuntu20.04`
   - This includes CUDA, Python, and common ML libraries

   **GPU Selection:**
   - **Recommended**: RTX 3090 or RTX 4090
   - **Minimum**: 8GB VRAM
   - **Budget**: RTX 3080
   - **High-end**: A100

   **Instance Configuration:**
   ```
   Storage: 100 GB (adjust based on dataset size)
   Duration: 24 hours (adjust based on training time)
   Secure Cloud: ✅ Enable (for persistent storage)
   ```

3. **Advanced Settings:**
   - **SSH Key**: Paste your public SSH key from Step 4
   - **Port Forwarding**: Enable if you want to access TensorBoard
   - **Auto-shutdown**: Set to prevent over-billing

4. **Review and Launch:**
   - Check estimated cost
   - Click "Rent" to launch instance
   - **Wait 2-5 minutes** for instance to start

### Step 6: Get Connection Details

After instance launches, you'll see:
```
SSH Command: ssh -p XXXXX root@sshX.vast.ai
Password: [random password]
```

**Save these details!** You'll need them for the next steps.

## 🔗 Phase 4: Connect to Instance

### Step 7: Test SSH Connection

```bash
# Test connection (replace with your actual details)
ssh -p XXXXX root@sshX.vast.ai

# If prompted for password, use the password provided
# If using SSH key, it should connect automatically

# Once connected, you should see:
# root@vast-XXXX:~#
```

### Step 8: Update System and Install Dependencies

```bash
# On the vast.ai instance:

# Update system
apt update && apt upgrade -y

# Install additional tools
apt install -y htop tree git vim

# Check GPU availability
nvidia-smi
# You should see your GPU listed

# Check Python and CUDA
python3 --version
nvcc --version
```

## 📁 Phase 5: Upload Your Project

### Step 9: Upload Project Files

**Option A: Using the Data Transfer Script (Recommended)**

```bash
# On your LOCAL machine:
cd /path/to/your/master_thesis_ucm

# Upload project files
python scripts/vast_ai/data_transfer.py \
    --hostname sshX.vast.ai \
    --port XXXXX \
    --action upload \
    --local-path . \
    --remote-path /workspace/project

# Upload dataset
python scripts/vast_ai/data_transfer.py \
    --hostname sshX.vast.ai \
    --port XXXXX \
    --action upload \
    --local-path ../data \
    --remote-path /workspace/data
```

**Option B: Manual Upload with SCP**

```bash
# On your LOCAL machine:
# Upload project
scp -P XXXXX -r . root@sshX.vast.ai:/workspace/project

# Upload dataset
scp -P XXXXX -r ../data root@sshX.vast.ai:/workspace/data
```

### Step 10: Setup Project on Instance

```bash
# SSH into your instance
ssh -p XXXXX root@sshX.vast.ai

# Navigate to project
cd /workspace/project

# Install Python dependencies
pip install -r requirements_ssl.txt

# Make scripts executable
chmod +x scripts/vast_ai/*.py

# Verify setup
python3 -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
# Should show your GPU
```

## 🏃‍♂️ Phase 6: Start Training

### Step 11: Configure Training

```bash
# On the vast.ai instance:

# Edit configuration for the instance
nano vast_training_config.yaml

# Key changes for vast.ai:
# - Set remote paths correctly
# - Enable wandb logging
# - Set appropriate batch sizes
```

### Step 12: Start Self-Supervised Training

```bash
# On the vast.ai instance:

# Option 1: Run training directly
python self_supervised_model.py

# Option 2: Run with monitoring (recommended)
# Terminal 1 - Start training
python scripts/vast_ai/self_supervised_training.py

# Terminal 2 - Start monitoring (in another SSH session)
ssh -p XXXXX root@sshX.vast.ai
cd /workspace/project
python scripts/vast_ai/remote_monitor.py --interval 300
```

### Step 13: Monitor Training Progress

**Weights & Biases Dashboard:**
1. Go to https://wandb.ai/
2. Login to your account
3. Find your project: "ssl-dermatology-yourname"
4. Monitor real-time metrics

**TensorBoard (Optional):**
```bash
# On the vast.ai instance:
tensorboard --logdir=/workspace/outputs/tensorboard_logs --port=6006

# Access via browser (if port forwarding enabled):
# http://localhost:6006
```

**SSH Monitoring:**
```bash
# Check training logs
tail -f outputs/ssl_simclr/ssl_history.csv

# Monitor system resources
htop
nvidia-smi
```

## 📊 Phase 7: Training Progress

### Step 14: Understand Training Phases

Your training will go through two phases:

**Phase 1: Self-Supervised Pre-training (SimCLR)**
- Duration: ~12-18 hours (depending on GPU)
- Epochs: 25
- Purpose: Learn general image representations
- Output: Pre-trained encoder

**Phase 2: Supervised Fine-tuning**
- Duration: ~6-12 hours
- Epochs: 25
- Purpose: Adapt to dermatology classification
- Output: Final classification model

### Step 15: Monitor Key Metrics

**SSL Training Metrics:**
- `loss`: Contrastive loss (should decrease)
- `learning_rate`: Current learning rate
- `gpu_utilization`: Should be 80-95%

**Fine-tuning Metrics:**
- `coarse_output_loss`: Coarse classification loss
- `fine_output_loss`: Fine-grained classification loss
- `coarse_output_sparse_categorical_accuracy`: Coarse accuracy
- `fine_output_sparse_categorical_accuracy`: Fine accuracy

## 💾 Phase 8: Download Results

### Step 16: Download Trained Models

```bash
# On your LOCAL machine:

# Download all outputs
python scripts/vast_ai/data_transfer.py \
    --hostname sshX.vast.ai \
    --port XXXXX \
    --action download \
    --remote-path /workspace/outputs \
    --local-path ./downloaded_outputs

# Download specific model
python scripts/vast_ai/data_transfer.py \
    --hostname sshX.vast.ai \
    --port XXXXX \
    --action download \
    --remote-path /workspace/outputs/ssl_finetuned \
    --local-path ./ssl_finetuned_model
```

### Step 17: Verify Downloaded Models

```bash
# On your LOCAL machine:

# Check downloaded files
ls -la downloaded_outputs/
# Should see: ssl_simclr/, ssl_finetuned/, etc.

# Test model loading
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('downloaded_outputs/ssl_finetuned/ssl_finetuned_best_model.keras')
print('Model loaded successfully!')
print(f'Input shape: {model.input_shape}')
print(f'Output shapes: {[output.shape for output in model.outputs]}')
"
```

## 🧹 Phase 9: Cleanup

### Step 18: Terminate Instance

```bash
# On vast.ai website:
# 1. Go to "Instances" tab
# 2. Find your running instance
# 3. Click "Stop" or "Terminate"
# 4. Confirm termination

# This stops billing immediately
```

### Step 19: Backup Important Files

```bash
# On your LOCAL machine:

# Create backup of trained models
tar -czf ssl_model_backup_$(date +%Y%m%d).tar.gz downloaded_outputs/

# Upload to cloud storage (optional)
# - Google Drive, Dropbox, etc.
# - GitHub (if repository is private)
```

## 🔧 Troubleshooting Common Issues

### Issue 1: SSH Connection Failed

```bash
# Solutions:
# 1. Check instance is running on vast.ai dashboard
# 2. Verify port number and hostname
# 3. Try different SSH client
# 4. Check firewall settings

# Test connection
telnet sshX.vast.ai XXXXX
```

### Issue 2: Out of Memory Error

```bash
# Solutions:
# 1. Reduce batch size in config
nano vast_training_config.yaml
# Change batch_size from 64 to 32 or 16

# 2. Enable gradient checkpointing
# Already enabled in config

# 3. Use smaller image size
# Change img_size from 224 to 192
```

### Issue 3: Training Stuck/Slow

```bash
# Check GPU utilization
nvidia-smi
# Should show 80-95% utilization

# Check data loading
htop
# Should show Python process using CPU

# Solutions:
# 1. Increase num_parallel_calls in config
# 2. Use faster GPU instance
# 3. Enable mixed precision (already enabled)
```

### Issue 4: Wandb Connection Issues

```bash
# On vast.ai instance:
# 1. Check internet connection
ping google.com

# 2. Login to wandb
wandb login
# Enter your API key

# 3. Test wandb
python -c "import wandb; wandb.init(mode='offline')"
```

## 📈 Expected Results

### Training Timeline

| Phase | Duration | GPU Utilization | Memory Usage |
|-------|----------|----------------|--------------|
| SSL Pre-training | 12-18 hours | 85-95% | 6-8 GB |
| Fine-tuning | 6-12 hours | 80-90% | 4-6 GB |

### Expected Performance

- **SSL Training Loss**: Should decrease from ~7.0 to ~2.0
- **Fine-tuning Accuracy**: 
  - Coarse: 85-92%
  - Fine-grained: 75-85%
- **Total Cost**: $3-9 depending on GPU choice

## 🎯 Next Steps After Training

1. **Evaluate Model**: Run evaluation scripts on test set
2. **Compare Results**: Compare with baseline models
3. **Analyze Performance**: Check per-class accuracy
4. **Document Results**: Update thesis with findings
5. **Deploy Model**: Set up inference pipeline

## 💡 Pro Tips

1. **Cost Optimization**:
   - Use spot instances (50% cheaper)
   - Monitor training to stop when complete
   - Download results immediately

2. **Performance Optimization**:
   - Enable mixed precision (2x speedup)
   - Use larger batch sizes if memory allows
   - Pre-process data on local machine

3. **Reliability**:
   - Save checkpoints frequently
   - Monitor training progress
   - Have backup plans for failures

---

**🎉 Congratulations!** You've successfully set up and trained a self-supervised learning model on vast.ai. The trained model should show improved performance on your dermatology classification task, especially for minority classes.

For any issues or questions, refer to the troubleshooting section or check the logs in your downloaded outputs.
