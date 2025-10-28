#!/usr/bin/env python3
"""
🔧 FIXED CONTINUE ORC TRAINING  
=============================
Properly continue Orc AI training preserving ALL statistics
"""

import glob
import os
import json
from mass_training_system import MassTrainingSystem
import time

def find_latest_orc_model_and_stats():
    """Find the most recent Orc AI model and its corresponding stats"""
    model_files = glob.glob("orc_ai_mass_trained_*.pth")
    if not model_files:
        return None, 0, None
    
    # Extract game numbers and find the highest
    game_counts = []
    for file in model_files:
        try:
            game_count = int(file.split('_')[-1].split('.')[0])
            game_counts.append((game_count, file))
        except:
            continue
    
    if not game_counts:
        return None, 0, None
    
    # Get the model with highest game count
    latest_count, latest_file = max(game_counts)
    
    # Find corresponding stats file
    stats_files = glob.glob("orc_training_stats_*.json")
    latest_stats = None
    
    if stats_files:
        # Find the stats file with the closest game count
        best_match = None
        best_diff = float('inf')
        
        for stats_file in stats_files:
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    games = stats.get('games_completed', 0)
                    diff = abs(games - latest_count)
                    if diff < best_diff:
                        best_diff = diff
                        best_match = stats
            except:
                continue
        
        latest_stats = best_match
    
    return latest_file, latest_count, latest_stats

def continue_orc_training_fixed():
    """Continue Orc training with proper statistics preservation"""
    print("🔧 FIXED CONTINUE ORC AI TRAINING")
    print("=" * 45)
    
    # Find the latest model and stats
    latest_model, games_completed, previous_stats = find_latest_orc_model_and_stats()
    
    if latest_model:
        print(f"✅ Found latest Orc model: {latest_model}")
        print(f"📊 Games already completed: {games_completed:,}")
        
        if previous_stats:
            prev_win_rate = previous_stats.get('win_rate', 0) * 100
            prev_wins = previous_stats.get('wins', 0)
            prev_losses = previous_stats.get('losses', 0)
            prev_best_score = previous_stats.get('best_score', 0)
            print(f"🏆 Previous win rate: {prev_win_rate:.2f}%")
            print(f"📈 Previous record: {prev_wins}W / {prev_losses}L")
            print(f"⭐ Previous best score: {prev_best_score}")
        else:
            print("⚠️  No previous statistics found")
        print()
        
        # Copy to expected filename
        expected_filename = "orc_ai_mass_trained.pth"
        if os.path.exists(expected_filename):
            os.remove(expected_filename)
        os.system(f"cp '{latest_model}' '{expected_filename}'")
        print(f"📋 Copied {latest_model} to {expected_filename}")
        
    else:
        print("❌ No previous Orc models found")
        print("🆕 Will start training from scratch")
        games_completed = 0
        previous_stats = None
    
    # Ask for additional games
    print(f"\n🎯 TRAINING CONTINUATION:")
    additional_games = input(f"How many additional games to train? (e.g., 100000): ").strip()
    
    try:
        additional_games = int(additional_games)
    except ValueError:
        print("❌ Invalid number. Using default 100,000")
        additional_games = 100000
    
    total_games_target = games_completed + additional_games
    print(f"🎮 Will train from {games_completed:,} to {total_games_target:,} games")
    print(f"📈 Additional games: {additional_games:,}")
    print()
    
    confirm = input("🚀 Continue training? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    print(f"\n🟢 CONTINUING ORC TRAINING (FIXED STATS)...")
    print("=" * 50)
    
    # Create trainer for continuation
    trainer = MassTrainingSystem(
        empire_games=0,
        orc_games=additional_games,
        batch_size=1000,
        save_interval=5000
    )
    
    # PROPERLY preserve ALL previous statistics
    if previous_stats and games_completed > 0:
        print("📊 RESTORING PREVIOUS STATISTICS:")
        trainer.orc_stats['games_completed'] = games_completed
        trainer.orc_stats['wins'] = previous_stats.get('wins', 0)
        trainer.orc_stats['losses'] = previous_stats.get('losses', 0) 
        trainer.orc_stats['draws'] = previous_stats.get('draws', 0)
        trainer.orc_stats['total_score'] = previous_stats.get('average_score', 0) * games_completed
        trainer.orc_stats['best_score'] = previous_stats.get('best_score', -float('inf'))
        
        # Restore strategy success tracking
        if 'top_strategies' in previous_stats:
            for strategy, count in previous_stats['top_strategies'].items():
                trainer.orc_stats['strategy_success'][int(strategy)] = count
        
        print(f"   ✅ Games: {trainer.orc_stats['games_completed']:,}")
        print(f"   ✅ Wins: {trainer.orc_stats['wins']:,}")
        print(f"   ✅ Losses: {trainer.orc_stats['losses']:,}")
        print(f"   ✅ Best Score: {trainer.orc_stats['best_score']}")
        print(f"   ✅ Strategies: {len(trainer.orc_stats['strategy_success'])} tracked")
        print()
    
    start_time = time.time()
    
    # Initialize (will load the copied model)
    trainer.initialize_ai_agents()
    
    print(f"🧠 Current Orc AI epsilon: {trainer.orc_ai.epsilon:.6f}")
    
    # Calculate and display TRUE current win rate
    if trainer.orc_stats['games_completed'] > 0:
        true_win_rate = (trainer.orc_stats['wins'] / trainer.orc_stats['games_completed']) * 100
        print(f"🏆 TRUE current win rate: {true_win_rate:.2f}%")
    print()
    
    # Run training
    trainer.run_orc_training()
    
    # Calculate time
    total_time = (time.time() - start_time) / 3600
    
    print(f"\n🎉 CONTINUED TRAINING COMPLETE!")
    print("=" * 45)
    print(f"⏱️  Training Time: {total_time:.2f} hours")
    print(f"🎮 Total Games Now: {trainer.orc_stats['games_completed']:,}")
    print(f"📈 Games Added: {additional_games:,}")
    
    # Final results with CORRECT calculations
    if trainer.orc_stats['games_completed'] > 0:
        final_win_rate = (trainer.orc_stats['wins'] / trainer.orc_stats['games_completed']) * 100
        print(f"🏆 FINAL Win Rate: {final_win_rate:.2f}%")
        print(f"⭐ Best Score: {trainer.orc_stats['best_score']:.1f}")
        print(f"🧠 Final Epsilon: {trainer.orc_ai.epsilon:.6f}")
    
    print(f"\n💾 Final model: orc_ai_mass_trained_{trainer.orc_stats['games_completed']}.pth")
    print("✅ Statistics properly preserved!")

if __name__ == "__main__":
    continue_orc_training_fixed() 