#!/usr/bin/env python3
"""
🟢 ORC & GOBLIN MASSIVE TRAINING
===============================
Dedicated training system for Orc AI with 100,000 games
"""

from mass_training_system import MassTrainingSystem
import time

def run_orc_massive_training():
    """Run massive training for Orc AI only"""
    print("🟢 ORC & GOBLIN MASSIVE TRAINING SYSTEM")
    print("=" * 50)
    print("🎯 Training Orc AI with 100,000 games")
    print("🧠 Anti-Artillery Specialist Training")
    print("⚡ Against Empire Artillery Formations")
    print()
    
    # Show training scope
    print("📊 TRAINING PARAMETERS:")
    print("   • Orc Games: 100,000")
    print("   • Empire Games: 0 (Orc-only training)")
    print("   • Batch Size: 1,000")
    print("   • Save Interval: 5,000")
    print("   • Expected Time: ~2-3 hours")
    print()
    
    # Confirm start
    confirm = input("🚀 Start massive Orc training? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    print("\n🟢 INITIATING ORC WAAAGH! TRAINING PROTOCOL")
    print("=" * 50)
    
    # Create trainer focused on Orcs
    trainer = MassTrainingSystem(
        empire_games=0,           # No Empire training
        orc_games=100000,         # 100K Orc games
        batch_size=1000,          # Larger batches for efficiency
        save_interval=5000        # Save every 5K games
    )
    
    # Start the timer
    start_time = time.time()
    
    # Initialize AI agents
    trainer.initialize_ai_agents()
    
    # Run only Orc training
    print("\n🟢 STARTING ORC AI MASSIVE TRAINING")
    trainer.run_orc_training()
    
    # Calculate total time
    total_time = (time.time() - start_time) / 3600
    
    # Final summary
    print("\n🎉 MASSIVE ORC TRAINING COMPLETE!")
    print("=" * 50)
    print(f"⏱️  Total Time: {total_time:.2f} hours")
    print(f"🎮 Total Games: {trainer.orc_stats['games_completed']:,}")
    print(f"⚡ Games/Hour: {trainer.orc_stats['games_completed'] / total_time:.0f}")
    
    # Orc results
    orc_wr = (trainer.orc_stats['wins'] / max(trainer.orc_stats['games_completed'], 1)) * 100
    print(f"\n🟢 FINAL ORC AI RESULTS:")
    print(f"   • Win Rate: {orc_wr:.2f}%")
    print(f"   • Best Score: {trainer.orc_stats['best_score']:.1f}")
    print(f"   • Games Completed: {trainer.orc_stats['games_completed']:,}")
    print(f"   • Final Epsilon: {trainer.orc_ai.epsilon:.6f}")
    
    # Show strategy evolution
    if trainer.orc_stats['strategy_success']:
        print(f"\n🧠 TOP STRATEGIES LEARNED:")
        sorted_strategies = sorted(trainer.orc_stats['strategy_success'].items(), 
                                 key=lambda x: x[1], reverse=True)
        for i, (strategy, wins) in enumerate(sorted_strategies[:5]):
            print(f"   {i+1}. Action {strategy}: {wins} successful uses")
    
    print(f"\n💾 Final model saved as: orc_ai_mass_trained_{trainer.orc_stats['games_completed']}.pth")
    print("🏆 Your Orc AI is now a battle-hardened veteran!")

if __name__ == "__main__":
    run_orc_massive_training() 