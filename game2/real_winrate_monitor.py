#!/usr/bin/env python3
"""
🎯 REAL WIN RATE MONITOR
=======================
Shows the TRUE Orc AI performance by correctly calculating win rates
from the continuation point (100,000 games baseline)
"""

import json
import glob
import time
import os
from datetime import datetime

class RealWinRateMonitor:
    def __init__(self):
        self.baseline_games = 100000  # Starting point for continuation
        self.target_games = 400000    # Target total games (100K base + 300K new)
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
    
    def calculate_real_metrics(self, stats):
        """Calculate the TRUE performance metrics"""
        if not stats:
            return None
            
        total_games = stats['games_completed']
        total_wins = stats['wins']
        total_losses = stats['losses'] 
        total_draws = stats['draws']
        
        # Calculate metrics for NEW games only (since continuation)
        new_games = total_games - self.baseline_games
        new_wins = total_wins  # All recorded wins are from new games
        new_losses = total_losses  # All recorded losses are from new games
        new_draws = total_draws
        
        if new_games <= 0:
            return None
            
        # Real performance metrics
        real_win_rate = (new_wins / new_games) * 100
        real_loss_rate = (new_losses / new_games) * 100
        real_draw_rate = (new_draws / new_games) * 100
        
        return {
            'total_games': total_games,
            'new_games': new_games,
            'new_wins': new_wins,
            'new_losses': new_losses,
            'new_draws': new_draws,
            'real_win_rate': real_win_rate,
            'real_loss_rate': real_loss_rate,
            'real_draw_rate': real_draw_rate,
            'avg_score': stats.get('average_score', 0),
            'best_score': stats.get('best_score', 0),
            'epsilon': stats.get('epsilon', 1.0),
            'top_strategies': stats.get('top_strategies', {})
        }
    
    def monitor_real_performance(self):
        """Monitor with REAL win rate calculations"""
        print("🎯 REAL WIN RATE MONITOR")
        print("=" * 50)
        print("🔧 Showing TRUE Orc AI performance")
        print(f"📊 Baseline: {self.baseline_games:,} games")
        print(f"🎯 Target: {self.target_games:,} games")
        print("Press Ctrl+C to stop monitoring\n")
        
        previous_real_wr = None
        
        try:
            while True:
                stats = self.get_latest_training_stats()
                
                if stats:
                    real_metrics = self.calculate_real_metrics(stats)
                    
                    if real_metrics:
                        # Calculate progress
                        total_progress = (real_metrics['total_games'] / self.target_games) * 100
                        session_progress = (real_metrics['new_games'] / (self.target_games - self.baseline_games)) * 100
                        games_remaining = self.target_games - real_metrics['total_games']
                        
                        # Calculate speed  
                        elapsed_hours = (time.time() - self.start_time) / 3600
                        if elapsed_hours > 0:
                            session_rate = real_metrics['new_games'] / elapsed_hours
                            eta_hours = games_remaining / session_rate if session_rate > 0 else 0
                        else:
                            session_rate = 0
                            eta_hours = 0
                        
                        # Clear screen and display
                        os.system('clear' if os.name == 'posix' else 'cls')
                        
                        print("🎯 REAL ORC AI PERFORMANCE MONITOR")
                        print("=" * 55)
                        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"⏱️  Monitoring Time: {elapsed_hours:.1f}h")
                        print()
                        
                        print("📊 TRAINING PROGRESS:")
                        print(f"   Total Games: {real_metrics['total_games']:,} / {self.target_games:,}")
                        print(f"   New Games: {real_metrics['new_games']:,}")
                        print(f"   Session Progress: {session_progress:.1f}%")
                        print(f"   Remaining: {games_remaining:,} games")
                        print()
                        
                        print("🏆 REAL PERFORMANCE METRICS:")
                        print(f"   🎯 TRUE Win Rate: {real_metrics['real_win_rate']:.2f}%")
                        
                        # Show trend
                        trend_indicator = ""
                        if previous_real_wr is not None:
                            change = real_metrics['real_win_rate'] - previous_real_wr
                            if change > 0.1:
                                trend_indicator = f" 📈 (+{change:.2f}%)"
                            elif change < -0.1:
                                trend_indicator = f" 📉 ({change:.2f}%)"
                            else:
                                trend_indicator = " ➡️ (stable)"
                        
                        print(f"   📈 Performance Trend: {trend_indicator}")
                        print(f"   🔴 Loss Rate: {real_metrics['real_loss_rate']:.2f}%")
                        print(f"   ⚪ Draw Rate: {real_metrics['real_draw_rate']:.2f}%")
                        print()
                        
                        print("📈 DETAILED STATS:")
                        print(f"   Wins: {real_metrics['new_wins']:,}")
                        print(f"   Losses: {real_metrics['new_losses']:,}")
                        print(f"   Draws: {real_metrics['new_draws']:,}")
                        print(f"   Average Score: {real_metrics['avg_score']:.1f}")
                        print(f"   Best Score: {real_metrics['best_score']}")
                        print(f"   Epsilon: {real_metrics['epsilon']:.6f}")
                        print()
                        
                        if session_rate > 0:
                            print("⚡ TRAINING SPEED:")
                            print(f"   Rate: {session_rate:.0f} games/hour")
                            print(f"   ETA: {eta_hours:.1f} hours")
                            if eta_hours > 24:
                                print(f"   ETA: {eta_hours/24:.1f} days")
                            print()
                        
                        # Show top strategies
                        if real_metrics['top_strategies']:
                            print("🧠 TOP STRATEGIES:")
                            strategies = real_metrics['top_strategies']
                            total_strategy_uses = sum(strategies.values())
                            sorted_strategies = sorted(strategies.items(), key=lambda x: x[1], reverse=True)
                            
                            for i, (action, count) in enumerate(sorted_strategies[:5]):
                                percentage = (count / total_strategy_uses) * 100 if total_strategy_uses > 0 else 0
                                print(f"   {i+1}. Action {action}: {count} uses ({percentage:.1f}%)")
                            print()
                        
                        # Progress bars
                        bar_width = 50
                        
                        # Total progress
                        total_filled = int(bar_width * total_progress / 100)
                        total_bar = "█" * total_filled + "░" * (bar_width - total_filled)
                        print(f"Total:   [{total_bar}] {total_progress:.1f}%")
                        
                        # Session progress
                        session_filled = int(bar_width * session_progress / 100)
                        session_bar = "█" * session_filled + "░" * (bar_width - session_filled)
                        print(f"Session: [{session_bar}] {session_progress:.1f}%")
                        
                        # Performance indicator
                        performance_level = "🔥 EXCELLENT" if real_metrics['real_win_rate'] > 60 else \
                                          "🎯 GOOD" if real_metrics['real_win_rate'] > 40 else \
                                          "📈 IMPROVING" if real_metrics['real_win_rate'] > 20 else \
                                          "🎲 LEARNING"
                        
                        print(f"\n🏆 Performance Level: {performance_level}")
                        
                        previous_real_wr = real_metrics['real_win_rate']
                        
                    else:
                        print("⏳ Waiting for sufficient training data...")
                        
                else:
                    print("⏳ Waiting for training data...")
                
                time.sleep(10)  # Update every 10 seconds
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
            print("🔄 Training continues in background...")
            if real_metrics:
                print(f"🏆 Final TRUE win rate: {real_metrics['real_win_rate']:.2f}%")

if __name__ == "__main__":
    monitor = RealWinRateMonitor()
    monitor.monitor_real_performance() 