#!/usr/bin/env python3
"""
Monitor the massive dual training progress
"""

import os
import time
import subprocess
import torch
from datetime import datetime

def monitor_massive_training():
    """Monitor the massive training progress"""
    print("📊 MASSIVE DUAL TRAINING MONITOR")
    print("=" * 45)
    print(f"⏰ Monitor started: {datetime.now().strftime('%H:%M:%S')}")
    
    # Check if training process is running
    try:
        result = subprocess.run(['pgrep', '-f', 'massive_dual_training.py'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Training process is RUNNING")
            print(f"   Process ID: {result.stdout.strip()}")
        else:
            print("❌ Training process NOT FOUND")
            return
    except:
        print("⚠️ Cannot check process status")
    
    # Check log file
    print(f"\n📋 RECENT LOG OUTPUT:")
    print("-" * 30)
    try:
        with open('massive_training_log.txt', 'r') as f:
            lines = f.readlines()
            # Show last 15 lines
            for line in lines[-15:]:
                print(f"   {line.rstrip()}")
    except FileNotFoundError:
        print("   Log file not found yet...")
    except:
        print("   Error reading log file")
    
    # Check for model files
    print(f"\n💾 MODEL FILES:")
    print("-" * 20)
    
    empire_models = [f for f in os.listdir('.') if f.startswith('empire_massive_') and f.endswith('.pth')]
    orc_models = [f for f in os.listdir('.') if f.startswith('orc_massive_') and f.endswith('.pth')]
    
    print(f"🏛️ Empire models: {len(empire_models)}")
    for model in sorted(empire_models)[-5:]:  # Show last 5
        try:
            stat = os.stat(model)
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = time.ctime(stat.st_mtime)
            print(f"   {model} ({size_mb:.1f} MB) - {mod_time}")
        except:
            print(f"   {model}")
    
    print(f"🟢 Orc models: {len(orc_models)}")
    for model in sorted(orc_models)[-5:]:  # Show last 5
        try:
            stat = os.stat(model)
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = time.ctime(stat.st_mtime)
            print(f"   {model} ({size_mb:.1f} MB) - {mod_time}")
        except:
            print(f"   {model}")
    
    # Analyze latest models if available
    latest_empire = None
    latest_orc = None
    
    if empire_models:
        # Find latest by extracting number from filename
        def extract_number(filename):
            try:
                # Extract number from empire_massive_5000.pth -> 5000
                parts = filename.replace('empire_massive_', '').replace('.pth', '')
                if parts == '300k_final':
                    return 300000
                return int(parts)
            except:
                return 0
        
        latest_empire = max(empire_models, key=extract_number)
    
    if orc_models:
        def extract_number_orc(filename):
            try:
                parts = filename.replace('orc_massive_', '').replace('.pth', '')
                if parts == '300k_final':
                    return 300000
                return int(parts)
            except:
                return 0
        
        latest_orc = max(orc_models, key=extract_number_orc)
    
    print(f"\n🔬 LATEST MODEL ANALYSIS:")
    print("-" * 30)
    
    if latest_empire:
        try:
            print(f"🏛️ Latest Empire: {latest_empire}")
            checkpoint = torch.load(latest_empire, map_location='cpu', weights_only=False)
            games_str = latest_empire.replace('empire_massive_', '').replace('.pth', '')
            if games_str == '300k_final':
                games = 300000
            else:
                games = int(games_str)
            
            epsilon = checkpoint.get('epsilon', 'Unknown')
            memory_size = len(checkpoint.get('memory', []))
            victories = checkpoint.get('victories', 0)
            defeats = checkpoint.get('defeats', 0)
            
            win_rate = (victories / (victories + defeats)) * 100 if (victories + defeats) > 0 else 0
            
            print(f"   Games: {games:,}")
            print(f"   Win rate: {win_rate:.1f}% ({victories}W-{defeats}L)")
            print(f"   Epsilon: {epsilon}")
            print(f"   Memory: {memory_size:,} experiences")
            
        except Exception as e:
            print(f"   Error analyzing Empire model: {e}")
    
    if latest_orc:
        try:
            print(f"🟢 Latest Orc: {latest_orc}")
            checkpoint = torch.load(latest_orc, map_location='cpu', weights_only=False)
            games_str = latest_orc.replace('orc_massive_', '').replace('.pth', '')
            if games_str == '300k_final':
                games = 300000
            else:
                games = int(games_str)
            
            epsilon = checkpoint.get('epsilon', 'Unknown')
            memory_size = len(checkpoint.get('memory', []))
            victories = checkpoint.get('victories', 0)
            defeats = checkpoint.get('defeats', 0)
            
            win_rate = (victories / (victories + defeats)) * 100 if (victories + defeats) > 0 else 0
            
            print(f"   Games: {games:,}")
            print(f"   Win rate: {win_rate:.1f}% ({victories}W-{defeats}L)")
            print(f"   Epsilon: {epsilon}")
            print(f"   Memory: {memory_size:,} experiences")
            
        except Exception as e:
            print(f"   Error analyzing Orc model: {e}")
    
    # Estimate progress
    print(f"\n📈 PROGRESS ESTIMATION:")
    print("-" * 25)
    
    total_target = 600000  # 300k empire + 300k orc
    empire_progress = 0
    orc_progress = 0
    
    if latest_empire:
        try:
            games_str = latest_empire.replace('empire_massive_', '').replace('.pth', '')
            if games_str == '300k_final':
                empire_progress = 300000
            else:
                empire_progress = int(games_str)
        except:
            pass
    
    if latest_orc:
        try:
            games_str = latest_orc.replace('orc_massive_', '').replace('.pth', '')
            if games_str == '300k_final':
                orc_progress = 300000
            else:
                orc_progress = int(games_str)
        except:
            pass
    
    total_progress = empire_progress + orc_progress
    progress_pct = (total_progress / total_target) * 100
    
    print(f"🏛️ Empire: {empire_progress:,}/300,000 ({empire_progress/3000:.1f}%)")
    print(f"🟢 Orc: {orc_progress:,}/300,000 ({orc_progress/3000:.1f}%)")
    print(f"📊 Total: {total_progress:,}/600,000 ({progress_pct:.1f}%)")
    
    if progress_pct >= 100:
        print("🎉 TRAINING COMPLETE!")
    elif empire_progress >= 300000:
        print("🏛️ Empire training complete, Orc training in progress")
    else:
        print("🏛️ Empire training in progress")
    
    print(f"\n⏰ Monitoring completed: {datetime.now().strftime('%H:%M:%S')}")

def monitor_loop():
    """Monitor in a loop with updates"""
    try:
        while True:
            os.system('clear')  # Clear screen
            monitor_massive_training()
            print(f"\n⏳ Next update in 30 seconds... (Ctrl+C to stop)")
            time.sleep(30)
    except KeyboardInterrupt:
        print(f"\n👋 Monitoring stopped by user")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'loop':
        monitor_loop()
    else:
        monitor_massive_training() 