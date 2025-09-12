#!/usr/bin/env python3
"""
Remote Monitoring Script for Vast.ai Self-Supervised Learning Training
This script provides comprehensive monitoring of training progress on vast.ai instances.
"""

import os
import sys
import time
import json
import psutil
import GPUtil
import requests
import wandb
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import threading
import queue

class VastAITrainingMonitor:
    """Comprehensive monitoring system for vast.ai training instances."""
    
    def __init__(self, config_path: str = "vast_training_config.yaml"):
        """Initialize the monitoring system."""
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.monitoring = True
        self.metrics_queue = queue.Queue()
        self.start_time = datetime.now()
        
        # Initialize monitoring components
        self.setup_wandb()
        self.setup_logging()
        
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config {self.config_path}: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'monitoring': {
                'wandb': {'enabled': True, 'project': 'ssl-dermatology-vast'},
                'system_monitoring': {'enabled': True, 'log_interval': 100},
                'checkpointing': {'save_best_only': True}
            }
        }
    
    def setup_wandb(self):
        """Setup Weights & Biases logging."""
        wandb_config = self.config.get('monitoring', {}).get('wandb', {})
        
        if wandb_config.get('enabled', False):
            try:
                wandb.init(
                    project=wandb_config.get('project', 'ssl-dermatology-vast'),
                    entity=wandb_config.get('entity'),
                    tags=wandb_config.get('tags', ['vast-ai', 'ssl', 'monitoring']),
                    config={
                        'monitoring_start_time': self.start_time.isoformat(),
                        'instance_type': self.detect_instance_type(),
                        'config_file': str(self.config_path)
                    }
                )
                print("✅ Wandb monitoring initialized")
            except Exception as e:
                print(f"⚠️  Wandb initialization failed: {e}")
                wandb = None
        else:
            wandb = None
            
        self.wandb = wandb
    
    def setup_logging(self):
        """Setup local logging."""
        log_dir = Path("logs/vast_ai")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = log_dir / f"training_monitor_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        print(f"📝 Logging to: {self.log_file}")
    
    def detect_instance_type(self) -> str:
        """Detect the type of GPU instance."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return f"{gpu.name} ({gpu.memoryTotal}GB)"
        except:
            pass
        return "Unknown"
    
    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics."""
        metrics = {}
        
        # CPU metrics
        metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        metrics['cpu_count'] = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_used_gb'] = memory.used / (1024**3)
        metrics['memory_available_gb'] = memory.available / (1024**3)
        metrics['memory_total_gb'] = memory.total / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics['disk_percent'] = (disk.used / disk.total) * 100
        metrics['disk_used_gb'] = disk.used / (1024**3)
        metrics['disk_free_gb'] = disk.free / (1024**3)
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['gpu_memory_used'] = gpu.memoryUsed
                metrics['gpu_memory_total'] = gpu.memoryTotal
                metrics['gpu_memory_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                metrics['gpu_temperature'] = gpu.temperature
                metrics['gpu_name'] = gpu.name
        except Exception as e:
            metrics['gpu_error'] = str(e)
        
        # Network metrics (if available)
        try:
            net_io = psutil.net_io_counters()
            metrics['network_bytes_sent'] = net_io.bytes_sent
            metrics['network_bytes_recv'] = net_io.bytes_recv
        except:
            pass
        
        return metrics
    
    def get_training_progress(self) -> Dict:
        """Get training progress from log files."""
        progress = {}
        
        # Check for training history files
        output_dirs = [
            Path("outputs/ssl_simclr"),
            Path("outputs/ssl_finetuned")
        ]
        
        for output_dir in output_dirs:
            if output_dir.exists():
                # Look for CSV history files
                csv_files = list(output_dir.glob("*history.csv"))
                for csv_file in csv_files:
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_file)
                        if not df.empty:
                            latest_epoch = df.iloc[-1]
                            progress[f"{output_dir.name}_latest_epoch"] = latest_epoch.to_dict()
                    except Exception as e:
                        progress[f"{output_dir.name}_error"] = str(e)
        
        return progress
    
    def estimate_remaining_time(self, progress: Dict) -> Optional[float]:
        """Estimate remaining training time."""
        try:
            # Look for SSL training progress
            ssl_progress = progress.get('ssl_simclr_latest_epoch', {})
            if ssl_progress:
                current_epoch = ssl_progress.get('epoch', 0)
                total_epochs = 25  # From config
                
                if current_epoch > 0:
                    elapsed_time = (datetime.now() - self.start_time).total_seconds() / 3600
                    epochs_per_hour = current_epoch / elapsed_time
                    remaining_epochs = total_epochs - current_epoch
                    remaining_hours = remaining_epochs / epochs_per_hour
                    return remaining_hours
        except:
            pass
        return None
    
    def log_metrics(self, metrics: Dict, progress: Dict):
        """Log metrics to wandb and local file."""
        timestamp = datetime.now()
        
        # Prepare log entry
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'system': metrics,
            'training': progress
        }
        
        # Log to wandb
        if self.wandb:
            wandb_metrics = {}
            
            # System metrics
            for key, value in metrics.items():
                wandb_metrics[f'system/{key}'] = value
            
            # Training metrics
            for key, value in progress.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        wandb_metrics[f'training/{key}_{subkey}'] = subvalue
                else:
                    wandb_metrics[f'training/{key}'] = value
            
            # Add time estimates
            remaining_time = self.estimate_remaining_time(progress)
            if remaining_time:
                wandb_metrics['training/estimated_remaining_hours'] = remaining_time
            
            wandb.log(wandb_metrics)
        
        # Log to local file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Print summary
        self.print_status_summary(metrics, progress, remaining_time)
    
    def print_status_summary(self, metrics: Dict, progress: Dict, remaining_time: Optional[float]):
        """Print a formatted status summary."""
        print(f"\n{'='*60}")
        print(f"📊 TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # System status
        print(f"🖥️  System Status:")
        print(f"   CPU: {metrics.get('cpu_percent', 0):.1f}% | "
              f"Memory: {metrics.get('memory_percent', 0):.1f}% "
              f"({metrics.get('memory_used_gb', 0):.1f}/{metrics.get('memory_total_gb', 0):.1f} GB)")
        
        if 'gpu_utilization' in metrics:
            print(f"   GPU: {metrics['gpu_utilization']:.1f}% | "
                  f"Memory: {metrics.get('gpu_memory_percent', 0):.1f}% "
                  f"({metrics.get('gpu_memory_used', 0)}/{metrics.get('gpu_memory_total', 0)} MB)")
            print(f"   GPU Temp: {metrics.get('gpu_temperature', 'N/A')}°C")
        
        # Training status
        print(f"\n🎯 Training Status:")
        for key, value in progress.items():
            if isinstance(value, dict) and 'epoch' in value:
                epoch = value.get('epoch', 0)
                loss = value.get('loss', 0)
                print(f"   {key}: Epoch {epoch}, Loss: {loss:.4f}")
        
        # Time estimates
        if remaining_time:
            print(f"\n⏱️  Estimated remaining time: {remaining_time:.1f} hours")
        
        elapsed_time = (datetime.now() - self.start_time).total_seconds() / 3600
        print(f"   Total elapsed time: {elapsed_time:.1f} hours")
        
        print(f"{'='*60}")
    
    def monitor_training_files(self) -> Dict:
        """Monitor training output files for changes."""
        file_stats = {}
        
        # Monitor key output directories
        output_dirs = [
            "outputs/ssl_simclr",
            "outputs/ssl_finetuned",
            "logs"
        ]
        
        for dir_path in output_dirs:
            if Path(dir_path).exists():
                try:
                    files = list(Path(dir_path).rglob("*"))
                    for file_path in files:
                        if file_path.is_file():
                            stat = file_path.stat()
                            file_stats[str(file_path)] = {
                                'size': stat.st_size,
                                'modified': stat.st_mtime
                            }
                except Exception as e:
                    file_stats[f"{dir_path}_error"] = str(e)
        
        return file_stats
    
    def check_instance_health(self) -> Dict:
        """Check the health of the vast.ai instance."""
        health = {'status': 'healthy', 'issues': []}
        
        # Check system resources
        metrics = self.get_system_metrics()
        
        # Memory check
        if metrics.get('memory_percent', 0) > 90:
            health['issues'].append(f"High memory usage: {metrics['memory_percent']:.1f}%")
            health['status'] = 'warning'
        
        # Disk check
        if metrics.get('disk_percent', 0) > 85:
            health['issues'].append(f"High disk usage: {metrics['disk_percent']:.1f}%")
            health['status'] = 'warning'
        
        # GPU check
        if metrics.get('gpu_temperature', 0) > 80:
            health['issues'].append(f"High GPU temperature: {metrics['gpu_temperature']}°C")
            health['status'] = 'warning'
        
        # Check for training progress
        progress = self.get_training_progress()
        if not progress:
            health['issues'].append("No training progress detected")
            health['status'] = 'warning'
        
        return health
    
    def run_monitoring_loop(self, interval: int = 300):
        """Run the main monitoring loop."""
        print(f"🚀 Starting monitoring loop (interval: {interval}s)")
        print(f"📊 Wandb project: {self.config.get('monitoring', {}).get('wandb', {}).get('project', 'ssl-dermatology-vast')}")
        
        try:
            while self.monitoring:
                # Collect metrics
                system_metrics = self.get_system_metrics()
                training_progress = self.get_training_progress()
                instance_health = self.check_instance_health()
                
                # Log everything
                self.log_metrics(system_metrics, training_progress)
                
                # Log health status
                if instance_health['status'] != 'healthy':
                    print(f"⚠️  Health check: {instance_health['status']}")
                    for issue in instance_health['issues']:
                        print(f"   - {issue}")
                
                # Check if training is complete
                if self.is_training_complete(training_progress):
                    print("🎉 Training appears to be complete!")
                    break
                
                # Wait for next interval
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        finally:
            if self.wandb:
                wandb.finish()
            print("✅ Monitoring session ended")
    
    def is_training_complete(self, progress: Dict) -> bool:
        """Check if training is complete."""
        # Check for completion indicators
        ssl_progress = progress.get('ssl_simclr_latest_epoch', {})
        if ssl_progress:
            epoch = ssl_progress.get('epoch', 0)
            if epoch >= 25:  # SSL epochs complete
                finetune_progress = progress.get('ssl_finetuned_latest_epoch', {})
                if finetune_progress:
                    finetune_epoch = finetune_progress.get('epoch', 0)
                    if finetune_epoch >= 25:  # Fine-tuning epochs complete
                        return True
        return False
    
    def generate_report(self) -> str:
        """Generate a comprehensive training report."""
        report = []
        report.append("# Vast.ai Training Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # System information
        metrics = self.get_system_metrics()
        report.append("## System Information")
        report.append(f"- GPU: {metrics.get('gpu_name', 'Unknown')}")
        report.append(f"- GPU Memory: {metrics.get('gpu_memory_total', 0)} MB")
        report.append(f"- System Memory: {metrics.get('memory_total_gb', 0):.1f} GB")
        report.append(f"- CPU Cores: {metrics.get('cpu_count', 0)}")
        report.append("")
        
        # Training progress
        progress = self.get_training_progress()
        report.append("## Training Progress")
        for key, value in progress.items():
            if isinstance(value, dict) and 'epoch' in value:
                epoch = value.get('epoch', 0)
                loss = value.get('loss', 0)
                report.append(f"- {key}: Epoch {epoch}, Loss: {loss:.4f}")
        report.append("")
        
        # Time information
        elapsed_time = (datetime.now() - self.start_time).total_seconds() / 3600
        remaining_time = self.estimate_remaining_time(progress)
        report.append("## Time Information")
        report.append(f"- Elapsed time: {elapsed_time:.1f} hours")
        if remaining_time:
            report.append(f"- Estimated remaining: {remaining_time:.1f} hours")
        report.append("")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Monitor vast.ai training instance')
    parser.add_argument('--config', default='vast_training_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--interval', type=int, default=300,
                       help='Monitoring interval in seconds')
    parser.add_argument('--report', action='store_true',
                       help='Generate and print training report')
    
    args = parser.parse_args()
    
    monitor = VastAITrainingMonitor(args.config)
    
    if args.report:
        report = monitor.generate_report()
        print(report)
        
        # Save report to file
        report_file = Path(f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    else:
        monitor.run_monitoring_loop(args.interval)

if __name__ == '__main__':
    main()
