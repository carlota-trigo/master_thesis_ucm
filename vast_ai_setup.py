#!/usr/bin/env python3
"""
Vast.ai Setup and Deployment Script for Self-Supervised Learning
This script helps set up and manage training on vast.ai GPU instances.
"""

import os
import sys
import json
import time
import subprocess
import paramiko
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class VastAIManager:
    def __init__(self, config_path: str = "vast_config.yaml"):
        """Initialize VastAI manager with configuration."""
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.ssh_client = None
        
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            self.create_default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_default_config(self):
        """Create default configuration file."""
        default_config = {
            'vast_ai': {
                'api_key': 'YOUR_VAST_API_KEY_HERE',
                'username': 'root',
                'default_template': 'CUDA:Devel-Ubuntu20.04',
                'storage_size': 100,  # GB
                'max_duration': 24,   # hours
                'secure_cloud': True
            },
            'gpu_requirements': {
                'min_memory': 8,      # GB
                'min_compute': 6.0,   # Compute capability
                'preferred_types': ['RTX 3090', 'RTX 4090', 'A100', 'V100']
            },
            'training': {
                'batch_size': 64,     # Adjust based on GPU memory
                'epochs': 25,
                'learning_rate': 1e-3,
                'mixed_precision': True,
                'gradient_checkpointing': True
            },
            'data': {
                'local_data_path': '../data',
                'remote_data_path': '/workspace/data',
                'model_output_path': '/workspace/outputs'
            },
            'monitoring': {
                'use_wandb': True,
                'wandb_project': 'ssl-dermatology',
                'tensorboard': True,
                'log_interval': 100
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default configuration at {self.config_path}")
        print("Please edit the configuration file with your settings before proceeding.")
    
    def setup_environment(self):
        """Set up the local environment for vast.ai deployment."""
        print("Setting up local environment...")
        
        # Create necessary directories
        dirs_to_create = [
            'scripts/vast_ai',
            'configs/vast_ai',
            'logs/vast_ai',
            'outputs/ssl_simclr',
            'outputs/ssl_finetuned'
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Copy training script to vast_ai directory
        self.copy_training_script()
        
        # Create deployment script
        self.create_deployment_script()
        
        # Create monitoring setup
        self.create_monitoring_setup()
        
        print("✅ Local environment setup complete")
    
    def copy_training_script(self):
        """Copy and modify training script for vast.ai."""
        vast_script_path = Path('scripts/vast_ai/self_supervised_training.py')
        
        # Read the original script
        with open('self_supervised_model.py', 'r') as f:
            script_content = f.read()
        
        # Add vast.ai specific configurations
        vast_config = '''
# Vast.ai specific configurations
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Adjust batch size based on available GPU memory
try:
    import GPUtil
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_memory = gpus[0].memoryTotal
        if gpu_memory >= 24:  # 24GB+ GPU
            BATCH_SIZE = 128
        elif gpu_memory >= 16:  # 16GB+ GPU
            BATCH_SIZE = 96
        elif gpu_memory >= 12:  # 12GB+ GPU
            BATCH_SIZE = 64
        else:  # 8GB GPU
            BATCH_SIZE = 32
        print(f"Detected GPU memory: {gpu_memory}GB, using batch size: {BATCH_SIZE}")
except:
    BATCH_SIZE = 64  # Default fallback
    print("Could not detect GPU memory, using default batch size: 64")

# Enable mixed precision for better performance
if os.environ.get('USE_MIXED_PRECISION', 'true').lower() == 'true':
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")

# Wandb logging setup
if os.environ.get('USE_WANDB', 'true').lower() == 'true':
    import wandb
    wandb.init(
        project=os.environ.get('WANDB_PROJECT', 'ssl-dermatology'),
        config={
            'batch_size': BATCH_SIZE,
            'learning_rate': LR_SSL,
            'epochs': SSL_EPOCHS,
            'temperature': TEMPERATURE,
            'projection_dim': PROJECTION_DIM
        }
    )
'''
        
        # Insert vast.ai config after imports
        insert_point = script_content.find('# -------------------- Config --------------------')
        if insert_point != -1:
            script_content = (script_content[:insert_point] + 
                            vast_config + '\n' + 
                            script_content[insert_point:])
        
        # Write modified script
        with open(vast_script_path, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Training script prepared at {vast_script_path}")
    
    def create_deployment_script(self):
        """Create deployment script for vast.ai."""
        deployment_script = '''#!/bin/bash
# Vast.ai Deployment Script for Self-Supervised Learning

set -e

echo "Starting vast.ai deployment setup..."

# Update system
apt-get update -y
apt-get upgrade -y

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements_ssl.txt

# Install additional system dependencies
apt-get install -y htop tree git-lfs rsync

# Setup directories
mkdir -p /workspace/{data,outputs,logs,scripts}
cd /workspace

# Clone or copy project files (this will be done via SCP)
echo "Project files will be uploaded via SCP..."

# Set proper permissions
chmod -R 755 /workspace
chmod +x /workspace/scripts/vast_ai/*.py

# Test GPU availability
python3 -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

echo "✅ Deployment setup complete"
echo "Ready to start training..."

# Start training
cd /workspace
python3 scripts/vast_ai/self_supervised_training.py
'''
        
        script_path = Path('scripts/vast_ai/deploy.sh')
        with open(script_path, 'w') as f:
            f.write(deployment_script)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"✅ Deployment script created at {script_path}")
    
    def create_monitoring_setup(self):
        """Create monitoring and logging setup."""
        monitoring_script = '''
# Monitoring setup for vast.ai training
import time
import psutil
import GPUtil
import wandb
from datetime import datetime

class VastAIMonitor:
    def __init__(self):
        self.start_time = time.time()
        
    def log_system_stats(self):
        """Log system statistics to wandb."""
        if not wandb.run:
            return
            
        # CPU and memory stats
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU stats
        gpus = GPUtil.getGPUs()
        gpu_stats = {}
        if gpus:
            gpu = gpus[0]
            gpu_stats = {
                'gpu_utilization': gpu.load * 100,
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_temperature': gpu.temperature
            }
        
        # Log to wandb
        wandb.log({
            'system/cpu_percent': cpu_percent,
            'system/memory_percent': memory.percent,
            'system/memory_used_gb': memory.used / (1024**3),
            'system/memory_available_gb': memory.available / (1024**3),
            **gpu_stats
        })
        
    def log_training_progress(self, epoch, logs):
        """Log training progress."""
        if not wandb.run:
            return
            
        # Log training metrics
        wandb.log({
            'epoch': epoch,
            'ssl_loss': logs.get('loss', 0),
            'learning_rate': logs.get('learning_rate', 0),
            **logs
        })
        
        # Log system stats every 10 epochs
        if epoch % 10 == 0:
            self.log_system_stats()

# Usage in training script:
# monitor = VastAIMonitor()
# Add to training callbacks:
# callbacks.append(keras.callbacks.LambdaCallback(
#     on_epoch_end=lambda epoch, logs: monitor.log_training_progress(epoch, logs)
# ))
'''
        
        script_path = Path('scripts/vast_ai/monitoring.py')
        with open(script_path, 'w') as f:
            f.write(monitoring_script)
        
        print(f"✅ Monitoring setup created at {script_path}")
    
    def create_data_transfer_script(self):
        """Create script for transferring data to/from vast.ai instance."""
        transfer_script = '''#!/usr/bin/env python3
"""
Data transfer utilities for vast.ai instances
"""

import os
import sys
import paramiko
import scp
from pathlib import Path
import argparse
from typing import List

class DataTransfer:
    def __init__(self, hostname: str, port: int, username: str = 'root'):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.ssh_client = None
        self.scp_client = None
        
    def connect(self, ssh_key_path: str = None):
        """Connect to the vast.ai instance."""
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        if ssh_key_path:
            self.ssh_client.connect(
                self.hostname, 
                port=self.port, 
                username=self.username,
                key_filename=ssh_key_path
            )
        else:
            self.ssh_client.connect(
                self.hostname, 
                port=self.port, 
                username=self.username
            )
        
        self.scp_client = scp.SCPClient(self.ssh_client.get_transport())
        print(f"✅ Connected to {self.hostname}:{self.port}")
    
    def upload_data(self, local_path: str, remote_path: str):
        """Upload data to vast.ai instance."""
        print(f"Uploading {local_path} to {remote_path}...")
        self.scp_client.put(local_path, remote_path, recursive=True)
        print(f"✅ Upload complete")
    
    def download_results(self, remote_path: str, local_path: str):
        """Download results from vast.ai instance."""
        print(f"Downloading {remote_path} to {local_path}...")
        Path(local_path).mkdir(parents=True, exist_ok=True)
        self.scp_client.get(remote_path, local_path, recursive=True)
        print(f"✅ Download complete")
    
    def close(self):
        """Close connections."""
        if self.scp_client:
            self.scp_client.close()
        if self.ssh_client:
            self.ssh_client.close()

def main():
    parser = argparse.ArgumentParser(description='Transfer data to/from vast.ai instance')
    parser.add_argument('--hostname', required=True, help='Vast.ai instance hostname')
    parser.add_argument('--port', type=int, required=True, help='SSH port')
    parser.add_argument('--action', choices=['upload', 'download'], required=True)
    parser.add_argument('--local-path', required=True, help='Local file/directory path')
    parser.add_argument('--remote-path', required=True, help='Remote file/directory path')
    parser.add_argument('--ssh-key', help='Path to SSH key file')
    
    args = parser.parse_args()
    
    transfer = DataTransfer(args.hostname, args.port)
    transfer.connect(args.ssh_key)
    
    try:
        if args.action == 'upload':
            transfer.upload_data(args.local_path, args.remote_path)
        elif args.action == 'download':
            transfer.download_results(args.remote_path, args.local_path)
    finally:
        transfer.close()

if __name__ == '__main__':
    main()
'''
        
        script_path = Path('scripts/vast_ai/data_transfer.py')
        with open(script_path, 'w') as f:
            f.write(transfer_script)
        
        os.chmod(script_path, 0o755)
        print(f"✅ Data transfer script created at {script_path}")
    
    def generate_instructions(self):
        """Generate setup instructions."""
        instructions = '''
# Vast.ai Setup Instructions for Self-Supervised Learning

## Prerequisites
1. Vast.ai account with payment method added
2. SSH key pair (optional but recommended)
3. Your dataset prepared and accessible locally

## Step 1: Configure the Setup
1. Edit `vast_config.yaml` with your settings:
   - Add your vast.ai API key
   - Adjust GPU requirements based on your needs
   - Set data paths and training parameters

## Step 2: Launch Vast.ai Instance
1. Go to vast.ai and select a GPU instance:
   - Recommended: RTX 3090/4090 or A100
   - Minimum: 8GB VRAM, 16GB system RAM
   - Storage: 100GB+ (depending on dataset size)

2. Use these instance settings:
   - Template: CUDA:Devel-Ubuntu20.04
   - Enable "Secure Cloud" if using persistent storage
   - Set appropriate duration (24+ hours recommended)

## Step 3: Upload Your Project
After launching the instance, use the data transfer script:

```bash
# Upload your project files
python scripts/vast_ai/data_transfer.py \\
    --hostname YOUR_INSTANCE_HOSTNAME \\
    --port YOUR_SSH_PORT \\
    --action upload \\
    --local-path . \\
    --remote-path /workspace/project

# Upload your dataset
python scripts/vast_ai/data_transfer.py \\
    --hostname YOUR_INSTANCE_HOSTNAME \\
    --port YOUR_SSH_PORT \\
    --action upload \\
    --local-path ../data \\
    --remote-path /workspace/data
```

## Step 4: Connect and Start Training
```bash
# SSH into your instance
ssh -p YOUR_SSH_PORT root@YOUR_INSTANCE_HOSTNAME

# Navigate to project directory
cd /workspace/project

# Run deployment script
chmod +x scripts/vast_ai/deploy.sh
./scripts/vast_ai/deploy.sh
```

## Step 5: Monitor Training
1. **TensorBoard**: Access via port forwarding or web interface
2. **Wandb**: Check your wandb dashboard for real-time metrics
3. **SSH**: Connect periodically to check logs

## Step 6: Download Results
After training completes:
```bash
# Download trained models
python scripts/vast_ai/data_transfer.py \\
    --hostname YOUR_INSTANCE_HOSTNAME \\
    --port YOUR_SSH_PORT \\
    --action download \\
    --remote-path /workspace/outputs \\
    --local-path ./downloaded_outputs
```

## Troubleshooting
- **Out of memory**: Reduce batch size in config
- **Connection issues**: Check SSH key and firewall settings
- **Slow training**: Enable mixed precision and gradient checkpointing
- **Data transfer issues**: Use rsync for large datasets

## Cost Optimization
- Use spot instances for cost savings
- Monitor training progress to avoid over-billing
- Download results promptly after training
- Terminate instance when not in use

## Security Notes
- Use SSH keys instead of passwords
- Enable secure cloud for persistent storage
- Regularly backup important results
- Don't store sensitive data in public repositories
'''
        
        with open('VAST_AI_SETUP_INSTRUCTIONS.md', 'w') as f:
            f.write(instructions)
        
        print("✅ Setup instructions created at VAST_AI_SETUP_INSTRUCTIONS.md")

def main():
    parser = argparse.ArgumentParser(description='Vast.ai setup manager')
    parser.add_argument('--action', choices=['setup', 'config', 'instructions'], 
                       default='setup', help='Action to perform')
    parser.add_argument('--config', default='vast_config.yaml', 
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    manager = VastAIManager(args.config)
    
    if args.action == 'setup':
        manager.setup_environment()
        manager.create_data_transfer_script()
        manager.generate_instructions()
        print("\n🎉 Vast.ai setup complete!")
        print("Next steps:")
        print("1. Edit vast_config.yaml with your settings")
        print("2. Launch a vast.ai GPU instance")
        print("3. Follow the instructions in VAST_AI_SETUP_INSTRUCTIONS.md")
        
    elif args.action == 'config':
        manager.create_default_config()
        print("Configuration file created. Please edit it with your settings.")
        
    elif args.action == 'instructions':
        manager.generate_instructions()
        print("Instructions generated.")

if __name__ == '__main__':
    main()
