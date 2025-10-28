#!/usr/bin/env python3
"""
📊 ADVANCED ORC TRAINING MONITOR
===============================
Real-time monitoring for the 200K continuation training session
Tracks learning curves, strategy evolution, and performance metrics
"""

import json
import glob
import time
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

class AdvancedTrainingMonitor:
    def __init__(self):
        self.training_history = []
        self.start_games = 100000  # Starting from 100K
        self.target_games = 300000  # Target 300K total
        self.start_time = time.time()
        
    def get_latest_training_stats(self):
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
    
    def plot_learning_curves(self, history):
        """Generate learning curve plots"""
        if len(history) < 2:
            return
            
        try:
            games = [h['games_completed'] for h in history]
            win_rates = [h['win_rate'] * 100 for h in history]
            avg_scores = [h['average_score'] for h in history]
            epsilons = [h['epsilon'] for h in history]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Win Rate Progress
            ax1.plot(games, win_rates, 'g-', linewidth=2, marker='o', markersize=4)
            ax1.set_title('🏆 Win Rate Evolution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Games Completed')
            ax1.set_ylabel('Win Rate (%)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(max(win_rates) * 1.1, 20))
            
            # Average Score Progress  
            ax2.plot(games, avg_scores, 'b-', linewidth=2, marker='s', markersize=4)
            ax2.set_title('📊 Average Score Trend', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Games Completed')
            ax2.set_ylabel('Average Score')
            ax2.grid(True, alpha=0.3)
            
            # Learning Rate (Epsilon) Decay
            ax3.plot(games, epsilons, 'r-', linewidth=2, marker='^', markersize=4)
            ax3.set_title('🧠 Learning Rate (Epsilon) Decay', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Games Completed')
            ax3.set_ylabel('Epsilon (Exploration Rate)')
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            
            # Games per Hour
            if len(history) > 1:
                times = [(h.get('timestamp_epoch', time.time()) - self.start_time) / 3600 for h in history]
                rates = []
                for i in range(1, len(games)):
                    if times[i] > times[i-1]:
                        rate = (games[i] - games[i-1]) / (times[i] - times[i-1])
                        rates.append(rate)
                    else:
                        rates.append(0)
                
                if rates:
                    ax4.plot(games[1:], rates, 'm-', linewidth=2, marker='d', markersize=4)
                    ax4.set_title('⚡ Training Speed', fontsize=14, fontweight='bold')
                    ax4.set_xlabel('Games Completed')
                    ax4.set_ylabel('Games per Hour')
                    ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('orc_training_progress_advanced.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"📊 Plot generation error: {e}")
    
    def analyze_strategy_evolution(self, stats):
        """Analyze how strategies are evolving"""
        if 'top_strategies' not in stats:
            return {}
            
        strategies = stats['top_strategies']
        total_successes = sum(strategies.values())
        
        strategy_analysis = {}
        for action, count in strategies.items():
            percentage = (count / total_successes) * 100 if total_successes > 0 else 0
            strategy_analysis[action] = {
                'count': count,
                'percentage': percentage
            }
        
        return strategy_analysis
    
    def monitor_training(self):
        """Monitor training with advanced analytics"""
        print("📊 ADVANCED ORC TRAINING MONITOR")
        print("=" * 50)
        print("🎯 Monitoring 200K continuation training...")
        print(f"📈 Target: {self.start_games:,} → {self.target_games:,} games")
        print("Press Ctrl+C to stop monitoring\n")
        
        iteration = 0
        
        try:
            while True:
                stats = self.get_latest_training_stats()
                
                if stats:
                    # Add timestamp for rate calculations
                    stats['timestamp_epoch'] = time.time()
                    self.training_history.append(stats)
                    
                    games = stats['games_completed']
                    win_rate = stats['win_rate'] * 100
                    avg_score = stats['average_score']
                    best_score = stats['best_score']
                    epsilon = stats['epsilon']
                    
                    # Calculate progress
                    total_progress = (games / self.target_games) * 100
                    session_progress = ((games - self.start_games) / (self.target_games - self.start_games)) * 100
                    games_remaining = self.target_games - games
                    
                    # Calculate speeds
                    elapsed_hours = (time.time() - self.start_time) / 3600
                    if elapsed_hours > 0:
                        session_rate = (games - self.start_games) / elapsed_hours
                        eta_hours = games_remaining / session_rate if session_rate > 0 else 0
                    else:
                        session_rate = 0
                        eta_hours = 0
                    
                    # Clear screen and display
                    os.system('clear' if os.name == 'posix' else 'cls')
                    
                    print("🟢 ORC AI ADVANCED TRAINING MONITOR")
                    print("=" * 55)
                    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"⏱️  Session Time: {elapsed_hours:.1f}h")
                    print()
                    
                    print("📊 TRAINING PROGRESS:")
                    print(f"   Games: {games:,} / {self.target_games:,}")
                    print(f"   Total Progress: {total_progress:.1f}%")
                    print(f"   Session Progress: {session_progress:.1f}%")
                    print(f"   Remaining: {games_remaining:,} games")
                    print()
                    
                    print("🏆 PERFORMANCE METRICS:")
                    print(f"   Win Rate: {win_rate:.2f}%")
                    current_improvement = ""
                    if len(self.training_history) > 1:
                        prev_wr = self.training_history[-2]['win_rate'] * 100
                        improvement = win_rate - prev_wr
                        if improvement > 0:
                            current_improvement = f" (+{improvement:.2f}%)"
                        elif improvement < 0:
                            current_improvement = f" ({improvement:.2f}%)"
                    print(f"   Current Trend: {current_improvement}")
                    print(f"   Average Score: {avg_score:.1f}")
                    print(f"   Best Score: {best_score}")
                    print(f"   Epsilon: {epsilon:.6f}")
                    print()
                    
                    if session_rate > 0:
                        print("⚡ TRAINING SPEED:")
                        print(f"   Current Rate: {session_rate:.0f} games/hour")
                        print(f"   ETA: {eta_hours:.1f} hours")
                        days_remaining = eta_hours / 24
                        if days_remaining >= 1:
                            print(f"   ETA: {days_remaining:.1f} days")
                        print()
                    
                    # Strategy analysis
                    strategy_analysis = self.analyze_strategy_evolution(stats)
                    if strategy_analysis:
                        print("🧠 STRATEGY EVOLUTION:")
                        sorted_strategies = sorted(strategy_analysis.items(), 
                                                 key=lambda x: x[1]['count'], reverse=True)
                        for i, (action, data) in enumerate(sorted_strategies[:5]):
                            print(f"   {i+1}. Action {action}: {data['count']} uses ({data['percentage']:.1f}%)")
                        print()
                    
                    # Progress bars
                    bar_width = 50
                    
                    # Total progress bar
                    total_filled = int(bar_width * total_progress / 100)
                    total_bar = "█" * total_filled + "░" * (bar_width - total_filled)
                    print(f"Total:   [{total_bar}] {total_progress:.1f}%")
                    
                    # Session progress bar
                    session_filled = int(bar_width * session_progress / 100)
                    session_bar = "█" * session_filled + "░" * (bar_width - session_filled)
                    print(f"Session: [{session_bar}] {session_progress:.1f}%")
                    
                    # Generate plots every 10 iterations
                    if iteration % 10 == 0 and len(self.training_history) > 1:
                        self.plot_learning_curves(self.training_history)
                        print(f"\n📊 Learning curves updated: orc_training_progress_advanced.png")
                    
                else:
                    print("⏳ Waiting for training data...")
                
                iteration += 1
                time.sleep(15)  # Update every 15 seconds
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
            if len(self.training_history) > 1:
                self.plot_learning_curves(self.training_history)
                print("📊 Final learning curves saved!")
            print("🔄 Training continues in background...")

if __name__ == "__main__":
    monitor = AdvancedTrainingMonitor()
    monitor.monitor_training() 