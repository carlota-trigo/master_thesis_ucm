#!/usr/bin/env python3
"""
Quick Setup Script for Vast.ai Self-Supervised Learning
Run this script to automatically set up everything needed for vast.ai training.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Vast.ai Self-Supervised Learning Environment")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("self_supervised_model.py").exists():
        print("❌ Error: self_supervised_model.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Create necessary directories
    print("📁 Creating directory structure...")
    directories = [
        "scripts/vast_ai",
        "configs/vast_ai", 
        "logs/vast_ai",
        "outputs/ssl_simclr",
        "outputs/ssl_finetuned"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure created")
    
    # Install required packages
    print("\n📦 Installing required packages...")
    if not run_command("pip install -r requirements_ssl.txt", "Installing Python packages"):
        print("⚠️  Package installation failed. Please install manually:")
        print("   pip install -r requirements_ssl.txt")
    
    # Run the main setup script
    print("\n🔧 Running main setup script...")
    if not run_command("python vast_ai_setup.py --action setup", "Running vast.ai setup"):
        print("⚠️  Setup script failed. Please run manually:")
        print("   python vast_ai_setup.py --action setup")
    
    # Make scripts executable
    print("\n🔐 Making scripts executable...")
    scripts_to_make_executable = [
        "scripts/vast_ai/data_transfer.py",
        "scripts/vast_ai/remote_monitor.py",
        "scripts/vast_ai/deploy.sh"
    ]
    
    for script_path in scripts_to_make_executable:
        if Path(script_path).exists():
            os.chmod(script_path, 0o755)
            print(f"✅ Made {script_path} executable")
    
    # Check if configuration file exists
    if Path("vast_training_config.yaml").exists():
        print("\n⚙️  Configuration file found: vast_training_config.yaml")
        print("   Please edit this file with your specific settings:")
        print("   - Vast.ai API key")
        print("   - Wandb credentials")
        print("   - Data paths")
        print("   - Training parameters")
    else:
        print("\n⚠️  Configuration file not found. Creating default...")
        run_command("python vast_ai_setup.py --action config", "Creating default configuration")
    
    # Final instructions
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit vast_training_config.yaml with your settings")
    print("2. Read VAST_AI_SETUP_GUIDE.md for detailed instructions")
    print("3. Launch a vast.ai GPU instance")
    print("4. Upload your project and start training")
    print("\nQuick commands:")
    print("   # View setup guide")
    print("   cat VAST_AI_SETUP_GUIDE.md")
    print("\n   # Edit configuration")
    print("   nano vast_training_config.yaml")
    print("\n   # Generate instructions")
    print("   python vast_ai_setup.py --action instructions")
    print("\nHappy training! 🚀")

if __name__ == "__main__":
    main()
