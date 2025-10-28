#!/usr/bin/env python3
"""
📊 ORC TRAINING PROGRESS MONITOR
===============================
Real-time monitoring of the massive Orc AI training session
"""

import json
import glob
import time
import os
from datetime import datetime

def get_latest_training_stats():
    """Get the most recent training statistics"""
    stats_files = glob.glob("orc_training_stats_*.json")
    if not stats_files:
        return None
        
    # Get the most recent file
    latest_file = max(stats_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            return json.load(f)
    except:
        return None

def monitor_training():
    """Monitor training progress with live updates"""
    print("📊 ORC TRAINING PROGRESS MONITOR")
    print("=" * 45)
    print("Monitoring Orc AI training progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    previous_games = 0
    start_time = time.time()
    
    try:
        while True:
            stats = get_latest_training_stats()
            
            if stats:
                games = stats['games_completed']
                win_rate = stats['win_rate'] * 100
                avg_score = stats['average_score']
                best_score = stats['best_score']
                epsilon = stats['epsilon']
                
                # Calculate progress
                progress_percent = (games / 100000) * 100
                games_remaining = 100000 - games
                
                # Calculate speed
                elapsed = time.time() - start_time
                if elapsed > 0 and games > previous_games:
                    games_per_second = (games - previous_games) / elapsed if elapsed > 60 else 0
                    eta_seconds = games_remaining / games_per_second if games_per_second > 0 else 0
                    eta_hours = eta_seconds / 3600
                else:
                    games_per_second = 0
                    eta_hours = 0
                
                # Clear screen and display stats
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("🟢 ORC AI MASSIVE TRAINING - LIVE MONITOR")
                print("=" * 50)
                print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()
                
                print("📊 TRAINING PROGRESS:")
                print(f"   Games Completed: {games:,} / 100,000")
                print(f"   Progress: {progress_percent:.1f}%")
                print(f"   Games Remaining: {games_remaining:,}")
                print()
                
                print("🏆 PERFORMANCE METRICS:")
                print(f"   Win Rate: {win_rate:.2f}%")
                print(f"   Average Score: {avg_score:.1f}")
                print(f"   Best Score: {best_score}")
                print(f"   Learning Rate (ε): {epsilon:.6f}")
                print()
                
                if games_per_second > 0:
                    print("⚡ TRAINING SPEED:")
                    print(f"   Games/Second: {games_per_second:.1f}")
                    print(f"   Games/Hour: {games_per_second * 3600:.0f}")
                    print(f"   ETA: {eta_hours:.1f} hours")
                    print()
                
                # Show top strategies
                if 'top_strategies' in stats and stats['top_strategies']:
                    print("🧠 TOP STRATEGIES:")
                    strategies = stats['top_strategies']
                    for i, (action, count) in enumerate(list(strategies.items())[:3]):
                        print(f"   {i+1}. Action {action}: {count} successes")
                    print()
                
                # Progress bar
                bar_width = 40
                filled = int(bar_width * progress_percent / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"Progress: [{bar}] {progress_percent:.1f}%")
                
                previous_games = games
                start_time = time.time()
                
            else:
                print("⏳ Waiting for training data...")
            
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        print("Training continues in background...")

if __name__ == "__main__":
    monitor_training() 