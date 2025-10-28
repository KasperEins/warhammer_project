#!/usr/bin/env python3
"""
Quick test of the final improved Orc AI model.
"""

import torch
import numpy as np
from warhammer_ai_agent import WarhammerAIAgent as WarhammerAI
from mass_training_system import TrainingBattle

def test_improved_model(num_games=100):
    """Test the improved model."""
    print("Loading improved Orc AI model...")
    
    # Load the final improved model
    orc_ai = WarhammerAI(state_size=50, action_size=15)
    try:
        checkpoint = torch.load('orc_ai_improved_50000.pth', map_location='cpu')
        orc_ai.q_network.load_state_dict(checkpoint['q_network_state'])
        orc_ai.epsilon = checkpoint.get('epsilon', 0.01)  # Low epsilon for testing
        print(f"Loaded model with epsilon: {orc_ai.epsilon:.6f}")
        
        # Check memory if available
        if 'memory' in checkpoint:
            print(f"Experience buffer size: {len(checkpoint['memory'])}")
        else:
            print("No memory found in checkpoint")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create Empire AI opponent
    empire_ai = WarhammerAI(state_size=50, action_size=15)
    
    # Run test battles
    orc_wins = 0
    print(f"\nRunning {num_games} test battles...")
    
    for game in range(num_games):
        battle = TrainingBattle()
        
        # Orc vs Empire
        winner = battle.run_battle(orc_ai, empire_ai, faction1='orc', faction2='empire')
        
        if winner == 'orc':
            orc_wins += 1
        
        if (game + 1) % 20 == 0:
            current_rate = (orc_wins / (game + 1)) * 100
            print(f"After {game + 1} games: Orc win rate = {current_rate:.1f}%")
    
    final_rate = (orc_wins / num_games) * 100
    print(f"\n=== FINAL RESULTS ===")
    print(f"Games played: {num_games}")
    print(f"Orc wins: {orc_wins}")
    print(f"Orc win rate: {final_rate:.2f}%")
    print(f"Orc model epsilon: {orc_ai.epsilon:.6f}")

if __name__ == "__main__":
    test_improved_model() 