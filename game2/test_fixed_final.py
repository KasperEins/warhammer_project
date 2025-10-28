#!/usr/bin/env python3
"""
Test the final fixed improved model
"""

import torch
from warhammer_ai_agent import WarhammerAIAgent
from mass_training_system import TrainingBattle

def test_fixed_model(num_games=100):
    """Test the fixed improved model."""
    print("🔧 TESTING FIXED IMPROVED MODEL")
    print("=" * 40)
    
    # Load the fixed improved model
    print("Loading fixed improved Orc AI model...")
    orc_ai = WarhammerAIAgent(state_size=50, action_size=15)
    
    try:
        # Fix PyTorch loading issue
        checkpoint = torch.load('orc_ai_fixed_31.pth', map_location='cpu', weights_only=False)
        orc_ai.load_model('orc_ai_fixed_31.pth')  # Use original load method
        print(f"✅ Loaded model with epsilon: {orc_ai.epsilon:.6f}")
        print(f"📝 Memory size: {len(orc_ai.memory)}")
        print(f"🏆 Victories: {orc_ai.victories}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create Empire AI opponent (baseline)
    empire_ai = WarhammerAIAgent(state_size=50, action_size=15)
    
    # Run test battles
    orc_wins = 0
    print(f"\n🎮 Running {num_games} test battles...")
    
    for game in range(num_games):
        battle = TrainingBattle()
        
        # Set very low epsilon for testing (minimal exploration)
        original_epsilon = orc_ai.epsilon
        orc_ai.epsilon = 0.01  # Almost pure exploitation
        
        # Orc vs Empire
        winner = battle.run_battle(orc_ai, empire_ai, faction1='orc', faction2='empire')
        
        # Restore epsilon
        orc_ai.epsilon = original_epsilon
        
        if winner == 'orc':
            orc_wins += 1
        
        if (game + 1) % 20 == 0:
            current_rate = (orc_wins / (game + 1)) * 100
            print(f"After {game + 1} games: Orc win rate = {current_rate:.1f}%")
    
    final_rate = (orc_wins / num_games) * 100
    print(f"\n🎯 FINAL RESULTS")
    print("=" * 40)
    print(f"Games played: {num_games}")
    print(f"Orc wins: {orc_wins}")
    print(f"Orc win rate: {final_rate:.2f}%")
    print(f"Model training epsilon: {orc_ai.epsilon:.6f}")
    print(f"Model memory experiences: {len(orc_ai.memory)}")
    
    # Performance evaluation
    print(f"\n📊 PERFORMANCE ANALYSIS")
    print("=" * 40)
    if final_rate > 45:
        print("🔥 EXCELLENT: AI shows significant improvement!")
    elif final_rate > 35:
        print("🎯 GOOD: AI shows meaningful improvement!")
    elif final_rate > 25:
        print("📈 MODERATE: AI shows some improvement!")
    else:
        print("⚠️  BASELINE: AI performance is at baseline level")
    
    # Test compatibility with original system
    print(f"\n🔧 COMPATIBILITY TEST")
    print("=" * 40)
    
    # Can we load this model into a fresh WarhammerAIAgent?
    try:
        test_ai = WarhammerAIAgent(state_size=50, action_size=15)
        test_ai.load_model('orc_ai_fixed_31.pth')
        print("✅ Model is compatible with original WarhammerAIAgent")
        print(f"✅ Loaded epsilon: {test_ai.epsilon:.6f}")
        print(f"✅ Loaded memory: {len(test_ai.memory)} experiences")
        print(f"✅ Loaded victories: {test_ai.victories}")
    except Exception as e:
        print(f"❌ Compatibility issue: {e}")
    
    return final_rate

if __name__ == "__main__":
    test_fixed_model() 