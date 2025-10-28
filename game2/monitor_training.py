#!/usr/bin/env python3
"""
Monitor the progress of the fixed training system
"""

import os
import time
import torch

def monitor_progress():
    """Monitor training progress."""
    print("🔍 MONITORING FIXED TRAINING PROGRESS")
    print("=" * 45)
    
    # Check if training is running
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    training_running = 'fixed_improved_training.py' in result.stdout
    
    print(f"Training process running: {'✅ Yes' if training_running else '❌ No'}")
    
    # Check for model files
    fixed_models = [f for f in os.listdir('.') if f.startswith('orc_ai_fixed_') and f.endswith('.pth')]
    fixed_models.sort(key=lambda x: int(x.split('_')[3].split('.')[0]) if x.split('_')[3].split('.')[0].isdigit() else 0)
    
    print(f"\n📁 Model Files Found: {len(fixed_models)}")
    for model in fixed_models[-5:]:  # Show last 5
        try:
            stat = os.stat(model)
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = time.ctime(stat.st_mtime)
            print(f"   {model} ({size_mb:.1f} MB) - {mod_time}")
        except:
            print(f"   {model}")
    
    # Try to load latest model and check progress
    if fixed_models:
        latest_model = fixed_models[-1]
        try:
            print(f"\n🔬 ANALYZING LATEST MODEL: {latest_model}")
            checkpoint = torch.load(latest_model, map_location='cpu', weights_only=False)
            
            games_played = latest_model.split('_')[3].split('.')[0]
            epsilon = checkpoint.get('epsilon', 'Unknown')
            memory_size = len(checkpoint.get('memory', []))
            
            print(f"   Games completed: {games_played}")
            print(f"   Current epsilon: {epsilon}")
            print(f"   Memory experiences: {memory_size:,}")
            
            if 'successful_strategies' in checkpoint:
                strategies = checkpoint['successful_strategies']
                print(f"   Successful strategies: {len(strategies)}")
            
            if 'victories' in checkpoint:
                victories = checkpoint.get('victories', 0)
                defeats = checkpoint.get('defeats', 0)
                total_battles = victories + defeats
                if total_battles > 0:
                    win_rate = (victories / total_battles) * 100
                    print(f"   Win rate: {win_rate:.1f}% ({victories}/{total_battles})")
            
        except Exception as e:
            print(f"   Error analyzing model: {e}")
    
    # Check for the main model file
    if os.path.exists('orc_ai_fixed_improved.pth'):
        stat = os.stat('orc_ai_fixed_improved.pth')
        size_mb = stat.st_size / (1024 * 1024)
        print(f"\n💾 Main model: orc_ai_fixed_improved.pth ({size_mb:.1f} MB)")
    
    print(f"\n⏰ Monitoring timestamp: {time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    monitor_progress() 