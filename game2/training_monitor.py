#!/usr/bin/env python3
"""
WARHAMMER: THE OLD WORLD - TRAINING PROGRESS MONITOR
==================================================
Real-time monitoring of AI training progress
"""

import time
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
from collections import deque

class TrainingMonitor:
    """Monitor AI training progress in real-time"""
    
    def __init__(self, model_path='warhammer_ai_model.pth', check_interval=30):
        self.model_path = model_path
        self.check_interval = check_interval
        self.progress_history = []
        self.last_check_time = None
        self.start_time = datetime.now()
        
    def check_progress(self):
        """Check current training progress"""
        if not os.path.exists(self.model_path):
            return None
            
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            progress = {
                'timestamp': datetime.now().isoformat(),
                'episodes': len(checkpoint.get('scores', [])),
                'victories': checkpoint.get('victories', 0),
                'defeats': checkpoint.get('defeats', 0),
                'draws': checkpoint.get('draws', 0),
                'epsilon': checkpoint.get('epsilon', 1.0),
                'avg_score_last_100': np.mean(checkpoint.get('scores', [])[-100:]) if checkpoint.get('scores') else 0,
                'avg_score_overall': np.mean(checkpoint.get('scores', [])) if checkpoint.get('scores') else 0,
                'successful_strategies': len(checkpoint.get('successful_strategies', [])),
            }
            
            # Calculate win rate
            total_games = progress['victories'] + progress['defeats'] + progress['draws']
            progress['win_rate'] = progress['victories'] / max(1, total_games)
            
            # Calculate improvement metrics
            if len(self.progress_history) > 0:
                last_progress = self.progress_history[-1]
                progress['episodes_since_last'] = progress['episodes'] - last_progress['episodes']
                progress['score_improvement'] = progress['avg_score_last_100'] - last_progress['avg_score_last_100']
                progress['win_rate_improvement'] = progress['win_rate'] - last_progress['win_rate']
            else:
                progress['episodes_since_last'] = progress['episodes']
                progress['score_improvement'] = 0
                progress['win_rate_improvement'] = 0
                
            return progress
            
        except Exception as e:
            print(f"Error checking progress: {e}")
            return None
    
    def display_progress(self, progress):
        """Display current progress"""
        if not progress:
            print("❌ No training progress detected yet...")
            return
            
        elapsed_time = datetime.now() - self.start_time
        
        print("\n" + "="*60)
        print(f"🤖 WARHAMMER AI TRAINING PROGRESS - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        print(f"⏱️  Training Time: {str(elapsed_time).split('.')[0]}")
        print(f"🎮 Episodes Completed: {progress['episodes']:,}/10,000")
        print(f"📊 Progress: {progress['episodes']/10000*100:.1f}%")
        print()
        
        # Performance metrics
        print(f"🏆 Battle Results:")
        print(f"   Victories: {progress['victories']:,}")
        print(f"   Defeats: {progress['defeats']:,}")
        print(f"   Draws: {progress['draws']:,}")
        print(f"   Win Rate: {progress['win_rate']:.2%}")
        print()
        
        # Learning metrics
        print(f"🧠 Learning Metrics:")
        print(f"   Current Exploration (ε): {progress['epsilon']:.3f}")
        print(f"   Avg Score (Last 100): {progress['avg_score_last_100']:.1f}")
        print(f"   Avg Score (Overall): {progress['avg_score_overall']:.1f}")
        print(f"   Strategies Discovered: {progress['successful_strategies']}")
        print()
        
        # Improvement indicators
        if len(self.progress_history) > 0:
            print(f"📈 Recent Improvement:")
            print(f"   Episodes since last check: {progress['episodes_since_last']}")
            improvement_indicator = "📈" if progress['score_improvement'] > 0 else "📉" if progress['score_improvement'] < 0 else "➡️"
            print(f"   Score change: {improvement_indicator} {progress['score_improvement']:+.1f}")
            
            wr_indicator = "📈" if progress['win_rate_improvement'] > 0 else "📉" if progress['win_rate_improvement'] < 0 else "➡️"
            print(f"   Win rate change: {wr_indicator} {progress['win_rate_improvement']:+.2%}")
        
        # Estimated completion
        if progress['episodes'] > 0:
            episodes_per_hour = progress['episodes'] / max(elapsed_time.total_seconds() / 3600, 0.1)
            remaining_episodes = 10000 - progress['episodes']
            hours_remaining = remaining_episodes / max(episodes_per_hour, 0.1)
            completion_time = datetime.now() + pd.Timedelta(hours=hours_remaining)
            print(f"⏰ Estimated Completion: {completion_time.strftime('%H:%M:%S')} ({hours_remaining:.1f}h remaining)")
    
    def generate_progress_chart(self, progress):
        """Generate real-time progress visualization"""
        if not progress or len(self.progress_history) < 2:
            return
            
        try:
            plt.style.use('dark_background')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'🤖 AI Training Progress - Episode {progress["episodes"]:,}', 
                        fontsize=16, color='gold', weight='bold')
            
            # Extract time series data
            episodes = [p['episodes'] for p in self.progress_history]
            win_rates = [p['win_rate'] for p in self.progress_history]
            avg_scores = [p['avg_score_last_100'] for p in self.progress_history]
            epsilons = [p['epsilon'] for p in self.progress_history]
            
            # 1. Win Rate Evolution
            ax1.plot(episodes, win_rates, color='green', linewidth=2, marker='o', markersize=3)
            ax1.set_title('Win Rate Evolution', color='white')
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Win Rate')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # 2. Average Score Trend
            ax2.plot(episodes, avg_scores, color='orange', linewidth=2, marker='s', markersize=3)
            ax2.set_title('Average Score (Last 100 Episodes)', color='white')
            ax2.set_xlabel('Episodes')
            ax2.set_ylabel('Score')
            ax2.grid(True, alpha=0.3)
            
            # 3. Exploration Rate
            ax3.plot(episodes, epsilons, color='cyan', linewidth=2)
            ax3.set_title('Exploration Rate (ε)', color='white')
            ax3.set_xlabel('Episodes')
            ax3.set_ylabel('Epsilon')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            # 4. Training Speed
            if len(episodes) >= 2:
                time_diffs = [(datetime.fromisoformat(self.progress_history[i]['timestamp']) - 
                              datetime.fromisoformat(self.progress_history[i-1]['timestamp'])).total_seconds()
                             for i in range(1, len(self.progress_history))]
                episode_diffs = [episodes[i] - episodes[i-1] for i in range(1, len(episodes))]
                speeds = [ep_diff / max(time_diff, 1) * 3600 for ep_diff, time_diff in zip(episode_diffs, time_diffs)]
                
                ax4.plot(episodes[1:], speeds, color='red', linewidth=2)
                ax4.set_title('Training Speed (Episodes/Hour)', color='white')
                ax4.set_xlabel('Episodes')
                ax4.set_ylabel('Episodes per Hour')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('training_progress.png', dpi=150, bbox_inches='tight', 
                       facecolor='#1a1a1a', edgecolor='none')
            plt.close()
            
        except Exception as e:
            print(f"Error generating chart: {e}")
    
    def save_progress_log(self, progress):
        """Save progress to log file"""
        if not progress:
            return
            
        log_entry = {
            'timestamp': progress['timestamp'],
            'episode': progress['episodes'],
            'win_rate': progress['win_rate'],
            'avg_score': progress['avg_score_last_100'],
            'epsilon': progress['epsilon'],
            'victories': progress['victories'],
            'defeats': progress['defeats'],
            'draws': progress['draws']
        }
        
        # Append to log file
        with open('training_log.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def monitor_training(self, duration_hours=None):
        """Monitor training progress continuously"""
        print("🔍 STARTING TRAINING MONITOR")
        print("=" * 40)
        print(f"📊 Checking progress every {self.check_interval} seconds")
        print(f"💾 Model path: {self.model_path}")
        if duration_hours:
            print(f"⏰ Monitoring for {duration_hours} hours")
        print()
        
        start_time = time.time()
        
        try:
            while True:
                progress = self.check_progress()
                
                if progress:
                    self.progress_history.append(progress)
                    self.display_progress(progress)
                    self.save_progress_log(progress)
                    
                    # Generate chart every 10 checks
                    if len(self.progress_history) % 10 == 0:
                        self.generate_progress_chart(progress)
                        print(f"📊 Progress chart updated: training_progress.png")
                    
                    # Check if training is complete
                    if progress['episodes'] >= 10000:
                        print("\n🎉 TRAINING COMPLETED!")
                        break
                else:
                    print(f"⏳ Waiting for training to start... ({datetime.now().strftime('%H:%M:%S')})")
                
                # Check duration limit
                if duration_hours and (time.time() - start_time) > duration_hours * 3600:
                    print(f"\n⏰ Monitoring time limit ({duration_hours}h) reached")
                    break
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        
        # Final summary
        if self.progress_history:
            final_progress = self.progress_history[-1]
            print(f"\n📋 FINAL TRAINING SUMMARY:")
            print(f"   Episodes: {final_progress['episodes']:,}")
            print(f"   Final Win Rate: {final_progress['win_rate']:.2%}")
            print(f"   Final Avg Score: {final_progress['avg_score_last_100']:.1f}")
            print(f"   Strategies Learned: {final_progress['successful_strategies']}")


def main():
    """Main monitoring interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Warhammer AI Training')
    parser.add_argument('--interval', type=int, default=30, help='Check interval in seconds')
    parser.add_argument('--duration', type=float, help='Monitoring duration in hours')
    parser.add_argument('--model-path', default='warhammer_ai_model.pth', help='Path to model file')
    
    args = parser.parse_args()
    
    # Fix pandas import issue
    try:
        import pandas as pd
        globals()['pd'] = pd
    except ImportError:
        # Create a simple timedelta replacement
        class TimeDelta:
            def __init__(self, hours=0):
                self.hours = hours
        
        class PD:
            Timedelta = TimeDelta
        
        globals()['pd'] = PD()
    
    monitor = TrainingMonitor(
        model_path=args.model_path,
        check_interval=args.interval
    )
    
    monitor.monitor_training(duration_hours=args.duration)


if __name__ == "__main__":
    main() 