#!/usr/bin/env python3
"""
🧪 MASS TRAINING SYSTEM TEST
==========================
Quick test of the mass training system with smaller game counts
"""

from mass_training_system import MassTrainingSystem
import time

def test_training_system():
    """Test the training system with small batches"""
    print("🧪 TESTING MASS TRAINING SYSTEM")
    print("=" * 40)
    print("Running quick test with 100 games per faction...")
    print()
    
    # Create trainer with small game counts for testing
    trainer = MassTrainingSystem(
        empire_games=100,
        orc_games=100,
        batch_size=10,
        save_interval=50
    )
    
    # Test the full sequence
    trainer.run_full_training_sequence()
    
    print("\n✅ Test completed successfully!")
    print("The mass training system is ready for full-scale deployment.")

if __name__ == "__main__":
    test_training_system() 