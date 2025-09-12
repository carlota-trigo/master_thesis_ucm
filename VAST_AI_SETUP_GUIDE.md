# Vast.ai Setup Guide for Self-Supervised Learning

This comprehensive guide will help you set up and train your self-supervised learning model on vast.ai GPU instances.

## 🚀 Quick Start

1. **Run the setup script:**
   ```bash
   python vast_ai_setup.py --action setup
   ```

2. **Configure your settings:**
   - Edit `vast_training_config.yaml` with your preferences
   - Add your vast.ai API key and wandb credentials

3. **Launch training:**
   - Follow the detailed steps below

## 📋 Prerequisites

- [ ] Vast.ai account with payment method added
- [ ] Weights & Biases account (recommended for monitoring)
- [ ] SSH key pair (optional but recommended)
- [ ] Your dataset prepared and accessible locally
- [ ] Python 3.8+ with required packages

## 🔧 Initial Setup

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements_ssl.txt

# Install additional monitoring tools
pip install wandb paramiko scp pyyaml
```

### 2. Configure Settings

Edit `vast_training_config.yaml` with your specific requirements:

```yaml
# Key settings to configure:
vast_ai:
  template: "CUDA:Devel-Ubuntu20.04"
  storage_size: 100  # Adjust based on your dataset size
  max_duration: 24   # Hours - set based on expected training time

monitoring:
  wandb:
    enabled: true
    project: "your-project-name"
    entity: "your-wandb-username"  # Optional
```

## 🖥️ Vast.ai Instance Setup

### 1. Launch GPU Instance

1. Go to [vast.ai](https://vast.ai/)
2. Click "Create" to launch a new instance
3. Configure your instance:

   **Recommended Settings:**
   - **Template**: `CUDA:Devel-Ubuntu20.04`
   - **GPU**: RTX 3090/4090 or A100 (8GB+ VRAM)
   - **Storage**: 100GB+ (adjust for your dataset)
   - **Duration**: 24+ hours
   - **Enable**: "Secure Cloud" for persistent storage

4. **GPU Selection Guide:**
   - **RTX 4090**: Best performance/cost ratio (~$0.50/hour)
   - **RTX 3090**: Good balance (~$0.30/hour)
   - **A100**: High-end training (~$1.20/hour)
   - **RTX 3080**: Budget option (~$0.20/hour)

### 2. Connect to Instance

After launching, you'll get SSH connection details:

```bash
# SSH into your instance
ssh -p YOUR_PORT root@YOUR_HOSTNAME

# Example:
ssh -p 12345 root@ssh4.vast.ai
```

## 📁 Data Transfer

### Upload Your Project

```bash
# Upload project files
python scripts/vast_ai/data_transfer.py \
    --hostname YOUR_HOSTNAME \
    --port YOUR_PORT \
    --action upload \
    --local-path . \
    --remote-path /workspace/project

# Upload your dataset
python scripts/vast_ai/data_transfer.py \
    --hostname YOUR_HOSTNAME \
    --port YOUR_PORT \
    --action upload \
    --local-path ../data \
    --remote-path /workspace/data
```

### Alternative: Manual Upload

```bash
# Using SCP
scp -P YOUR_PORT -r . root@YOUR_HOSTNAME:/workspace/project
scp -P YOUR_PORT -r ../data root@YOUR_HOSTNAME:/workspace/data
```

## 🏃‍♂️ Running Training

### 1. Setup on Instance

```bash
# SSH into your instance
ssh -p YOUR_PORT root@YOUR_HOSTNAME

# Navigate to project
cd /workspace/project

# Install dependencies
pip install -r requirements_ssl.txt

# Make scripts executable
chmod +x scripts/vast_ai/*.py
```

### 2. Start Training

```bash
# Option 1: Run the main training script
python self_supervised_model.py

# Option 2: Run with monitoring
python scripts/vast_ai/self_supervised_training.py &
python scripts/vast_ai/remote_monitor.py --interval 300
```

### 3. Monitor Progress

**Weights & Biases (Recommended):**
- Visit your wandb dashboard
- Real-time metrics and visualizations
- System resource monitoring

**TensorBoard:**
```bash
# On the vast.ai instance
tensorboard --logdir=/workspace/outputs/tensorboard_logs --port=6006

# Access via port forwarding (if enabled)
# http://localhost:6006
```

**SSH Monitoring:**
```bash
# Check training progress
tail -f logs/vast_ai/training_monitor_*.log

# Monitor system resources
htop
nvidia-smi
```

## 📊 Monitoring and Logging

### Automatic Monitoring

The monitoring script tracks:
- **System metrics**: CPU, memory, GPU utilization
- **Training progress**: Loss, accuracy, epochs
- **Time estimates**: Remaining training time
- **Health checks**: Instance stability

### Manual Monitoring

```bash
# Check GPU status
nvidia-smi

# Monitor disk usage
df -h

# Check running processes
ps aux | grep python

# View training logs
tail -f outputs/ssl_simclr/ssl_history.csv
```

## 💾 Downloading Results

### Download Trained Models

```bash
# Download all outputs
python scripts/vast_ai/data_transfer.py \
    --hostname YOUR_HOSTNAME \
    --port YOUR_PORT \
    --action download \
    --remote-path /workspace/outputs \
    --local-path ./downloaded_outputs

# Download specific model
python scripts/vast_ai/data_transfer.py \
    --hostname YOUR_HOSTNAME \
    --port YOUR_PORT \
    --action download \
    --remote-path /workspace/outputs/ssl_finetuned \
    --local-path ./ssl_model
```

### Manual Download

```bash
# Using SCP
scp -P YOUR_PORT -r root@YOUR_HOSTNAME:/workspace/outputs ./downloaded_outputs
```

## 🔧 Configuration Optimization

### GPU Memory Optimization

```yaml
# In vast_training_config.yaml
training:
  optimizations:
    mixed_precision: true          # Use FP16 for 2x speedup
    gradient_checkpointing: true   # Reduce memory usage
    batch_size: 64                # Adjust based on GPU memory

memory:
  gpu_memory_growth: true         # Allow memory to grow
  max_gpu_memory: 0.9            # Use max 90% of GPU
```

### Performance Tuning

```yaml
data:
  data_loading:
    num_parallel_calls: 4         # Parallel data loading
    prefetch_factor: 2            # Prefetch batches
    cache_dataset: false          # Don't cache (saves memory)
```

## 💰 Cost Optimization

### Estimated Costs (2024 Pricing)

| GPU | Hourly Rate | Training Time | Total Cost |
|-----|-------------|---------------|------------|
| RTX 3080 | $0.20 | 18 hours | $3.60 |
| RTX 3090 | $0.30 | 15 hours | $4.50 |
| RTX 4090 | $0.50 | 12 hours | $6.00 |
| A100 | $1.20 | 8 hours | $9.60 |

### Cost-Saving Tips

1. **Use Spot Instances**: 50-80% cheaper than on-demand
2. **Monitor Training**: Stop instance when training completes
3. **Optimize Batch Size**: Larger batches = faster training
4. **Use Mixed Precision**: 2x speedup with minimal accuracy loss
5. **Pre-download Data**: Reduce bandwidth costs

## 🚨 Troubleshooting

### Common Issues

**Out of Memory Error:**
```bash
# Solutions:
1. Reduce batch_size in config
2. Enable gradient_checkpointing
3. Use mixed_precision
4. Reduce image_size
```

**Slow Training:**
```bash
# Solutions:
1. Enable mixed_precision
2. Increase num_parallel_calls
3. Use faster GPU instance
4. Enable model compilation
```

**Connection Issues:**
```bash
# Solutions:
1. Check SSH key configuration
2. Verify firewall settings
3. Use VPN if needed
4. Try different SSH port
```

**Training Stuck:**
```bash
# Check:
1. GPU utilization: nvidia-smi
2. Disk space: df -h
3. Memory usage: htop
4. Training logs: tail -f logs/*
```

### Debug Commands

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Test data loading
python -c "import utils; print('Utils imported successfully')"

# Check system resources
python scripts/vast_ai/remote_monitor.py --report
```

## 📈 Performance Expectations

### Training Times (Approximate)

| Dataset Size | GPU | SSL Epochs | Fine-tune Epochs | Total Time |
|--------------|-----|------------|------------------|------------|
| 10K images | RTX 3080 | 15 hours | 8 hours | 23 hours |
| 50K images | RTX 3090 | 12 hours | 6 hours | 18 hours |
| 100K images | RTX 4090 | 8 hours | 4 hours | 12 hours |

### Expected Performance Improvements

- **SSL Pre-training**: +2-5% accuracy on minority classes
- **Mixed Precision**: 2x training speed
- **Optimized Augmentation**: +1-3% overall accuracy
- **Ensemble Methods**: +3-7% accuracy over single models

## 🔐 Security Best Practices

1. **Use SSH Keys**: Don't rely on passwords
2. **Secure Cloud**: Enable for persistent storage
3. **Regular Backups**: Download results promptly
4. **Monitor Access**: Check instance logs
5. **Clean Up**: Terminate instances when done

## 📞 Support and Resources

### Documentation
- [Vast.ai Documentation](https://vast.ai/docs/)
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

### Community
- [Vast.ai Discord](https://discord.gg/vast-ai)
- [TensorFlow Community](https://discuss.tensorflow.org/)

### Emergency Contacts
- Vast.ai Support: support@vast.ai
- Instance Issues: Check vast.ai status page

## 🎯 Next Steps

After successful training:

1. **Evaluate Model**: Run evaluation scripts on test set
2. **Compare Results**: Compare with baseline models
3. **Deploy Model**: Set up inference pipeline
4. **Document Results**: Update thesis with findings
5. **Share Code**: Make repository public (if desired)

---

**Happy Training! 🚀**

For questions or issues, please refer to the troubleshooting section or create an issue in the repository.
