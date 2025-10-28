#!/usr/bin/env python3
"""
🚀 DEMO: AI SELF-PLAY TRAINING (100 battles)
==============================================
Demonstration of the 10,000 battle training system
"""

from ai_self_play_trainer import SelfPlayTrainer

def run_demo():
    print("🎮 DEMO: AI SELF-PLAY TRAINING")
    print("=" * 50)
    print("This demo will run 100 battles instead of 10,000")
    print("to show you how the self-play system works!")
    print()
    
    # Create demo trainer with 100 battles
    trainer = SelfPlayTrainer(num_simulations=100)
    trainer.batch_size = 25  # Smaller batches for demo
    
    # Run the demo training
    trainer.run_training_simulation()
    
    print("\n🎯 DEMO COMPLETE!")
    print("The full 10,000 battle training would:")
    print("• Take approximately 2-4 hours")
    print("• Generate significantly more sophisticated strategies")
    print("• Create perfectly balanced AI opponents")
    print("• Discover emergent tactical combinations")

if __name__ == "__main__":
    run_demo() 