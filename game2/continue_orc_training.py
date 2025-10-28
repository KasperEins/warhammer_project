#!/usr/bin/env python3
"""
🔄 CONTINUE ORC TRAINING
=======================
Continue Orc AI training from the most recent saved checkpoint
"""

import glob
import os
from mass_training_system import MassTrainingSystem
import time

def find_latest_orc_model():
    """Find the most recent Orc AI model"""
    model_files = glob.glob("orc_ai_mass_trained_*.pth")
    if not model_files:
        return None, 0
    
    # Extract game numbers and find the highest
    game_counts = []
    for file in model_files:
        try:
            # Extract number from filename like "orc_ai_mass_trained_100000.pth"
            game_count = int(file.split('_')[-1].split('.')[0])
            game_counts.append((game_count, file))
        except:
            continue
    
    if not game_counts:
        return None, 0
    
    # Get the model with highest game count
    latest_count, latest_file = max(game_counts)
    return latest_file, latest_count

def continue_orc_training():
    """Continue Orc training from the latest checkpoint"""
    print("🔄 CONTINUE ORC AI TRAINING")
    print("=" * 40)
    
    # Find the latest model
    latest_model, games_completed = find_latest_orc_model()
    
    if latest_model:
        print(f"✅ Found latest Orc model: {latest_model}")
        print(f"📊 Games already completed: {games_completed:,}")
        print()
        
        # Copy to the expected filename for loading
        expected_filename = "orc_ai_mass_trained.pth"
        if os.path.exists(expected_filename):
            os.remove(expected_filename)
        os.system(f"cp '{latest_model}' '{expected_filename}'")
        print(f"📋 Copied {latest_model} to {expected_filename}")
        
    else:
        print("❌ No previous Orc models found")
        print("🆕 Will start training from scratch")
        games_completed = 0
    
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
    
    print(f"\n🟢 CONTINUING ORC TRAINING...")
    print("=" * 40)
    
    # Create trainer for continuation
    trainer = MassTrainingSystem(
        empire_games=0,                    # No Empire training
        orc_games=additional_games,        # Additional games to train
        batch_size=1000,                   # Large batches for efficiency  
        save_interval=5000                 # Save every 5K games
    )
    
    # Override the initial games_completed count if we're continuing
    if games_completed > 0:
        trainer.orc_stats['games_completed'] = games_completed
        print(f"📊 Starting from {games_completed:,} completed games")
    
    start_time = time.time()
    
    # Initialize (will load the copied model)
    trainer.initialize_ai_agents()
    
    print(f"🧠 Current Orc AI epsilon: {trainer.orc_ai.epsilon:.6f}")
    print()
    
    # Run training
    trainer.run_orc_training()
    
    # Calculate time
    total_time = (time.time() - start_time) / 3600
    
    print(f"\n🎉 CONTINUED TRAINING COMPLETE!")
    print("=" * 40)
    print(f"⏱️  Training Time: {total_time:.2f} hours")
    print(f"🎮 Total Games Now: {trainer.orc_stats['games_completed']:,}")
    print(f"📈 Games Added: {additional_games:,}")
    
    # Final results
    if trainer.orc_stats['games_completed'] > 0:
        win_rate = (trainer.orc_stats['wins'] / trainer.orc_stats['games_completed']) * 100
        print(f"🏆 Final Win Rate: {win_rate:.2f}%")
        print(f"⭐ Best Score: {trainer.orc_stats['best_score']:.1f}")
        print(f"🧠 Final Epsilon: {trainer.orc_ai.epsilon:.6f}")
    
    print(f"\n💾 Final model: orc_ai_mass_trained_{trainer.orc_stats['games_completed']}.pth")
    print("🔄 Ready for another training session!")

if __name__ == "__main__":
    continue_orc_training() 