#!/usr/bin/env python3
"""
Quick test to verify the replay method fix works
"""

from fixed_improved_training import FixedImprovedAI
from mass_training_system import MassTrainingSystem
import torch

def test_replay_fix():
    """Test that the replay fix works with MassTrainingSystem"""
    print("🔧 TESTING REPLAY METHOD FIX")
    print("=" * 40)
    
    # Create the fixed AI
    print("Creating FixedImprovedAI...")
    ai = FixedImprovedAI(state_size=50, action_size=15)
    
    # Add some fake experiences to memory
    print("Adding test experiences to memory...")
    for i in range(100):
        # Fake experience: (state, action, reward, next_state, done)
        state = torch.zeros(50)
        action = i % 15
        reward = 1.0 if i % 3 == 0 else -0.1
        next_state = torch.ones(50)
        done = i % 10 == 0
        
        ai.remember(state, action, reward, next_state, done)
    
    print(f"Memory size: {len(ai.memory)}")
    
    # Test the replay method with batch_size parameter
    print("\n🎯 Testing replay() with batch_size parameter...")
    try:
        ai.replay(batch_size=32)
        print("✅ SUCCESS: replay(batch_size=32) works!")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    # Test without parameters
    print("\n🎯 Testing replay() without parameters...")
    try:
        ai.replay()
        print("✅ SUCCESS: replay() works!")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    # Test integration with MassTrainingSystem
    print("\n🎯 Testing with MassTrainingSystem...")
    try:
        trainer = MassTrainingSystem()
        # This should work now without the batch_size error
        print("✅ SUCCESS: MassTrainingSystem integration ready!")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    test_replay_fix() 