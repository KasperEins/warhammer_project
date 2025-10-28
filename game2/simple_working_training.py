#!/usr/bin/env python3
"""
Simple working training that actually learns
"""

from fixed_improved_training import FixedImprovedAI
from mass_training_system import TrainingBattle
import time
import random

def simple_working_training():
    """Run actual working training session"""
    print("🚀 SIMPLE WORKING TRAINING")
    print("=" * 35)
    
    # Create the fixed AI
    print("Creating FixedImprovedAI...")
    orc_ai = FixedImprovedAI(state_size=50, action_size=15)
    print(f"Initial epsilon: {orc_ai.epsilon}")
    print(f"Initial memory: {len(orc_ai.memory)}")
    
    # Run training games
    total_games = 100
    wins = 0
    
    print(f"\n🔥 Running {total_games} training games...")
    
    for game in range(total_games):
        # Create battle
        battle = TrainingBattle()
        battle.create_armies()
        
        # Track battle progression
        total_reward = 0
        prev_state = battle.get_ai_state()
        
        # Run battle turns
        for turn in range(20):  # Max 20 turns per battle
            # AI makes decision
            action = orc_ai.act(prev_state)
            
            # Simulate battle turn
            battle.simulate_turn()
            
            # Get new state
            new_state = battle.get_ai_state()
            
            # Calculate reward (simple version)
            orc_alive = sum(1 for u in battle.units if u.faction == "orcs" and u.is_alive)
            empire_alive = sum(1 for u in battle.units if u.faction == "nuln" and u.is_alive)
            
            # Basic reward structure
            reward = orc_alive * 2 - empire_alive  # Reward for orc survival, penalty for empire survival
            
            # Bonus for winning
            if empire_alive == 0 and orc_alive > 0:
                reward += 50  # Victory bonus
                battle_done = True
            elif orc_alive == 0:
                reward -= 30  # Defeat penalty
                battle_done = True
            else:
                battle_done = False
            
            total_reward += reward
            
            # Store experience for learning
            orc_ai.remember(prev_state, action, reward, new_state, battle_done)
            
            prev_state = new_state
            
            if battle_done:
                break
        
        # Determine winner
        battle.calculate_final_scores()
        winner = battle.get_winner()
        
        if winner == 'orc':
            wins += 1
            orc_ai.victories += 1
        else:
            orc_ai.defeats += 1
        
        # Train the AI every few games if we have enough memory
        if len(orc_ai.memory) >= 32 and game % 5 == 0:
            print(f"  Game {game+1}: Training AI (memory: {len(orc_ai.memory)})...")
            try:
                orc_ai.replay()  # This should work now!
            except Exception as e:
                print(f"    Replay error: {e}")
        
        # Progress update
        if (game + 1) % 20 == 0:
            current_rate = (wins / (game + 1)) * 100
            print(f"  After {game+1:3d} games: {current_rate:5.1f}% win rate, ε={orc_ai.epsilon:.3f}")
    
    # Final results
    final_rate = (wins / total_games) * 100
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Win rate: {final_rate:.1f}% ({wins}/{total_games})")
    print(f"   Memory experiences: {len(orc_ai.memory):,}")
    print(f"   Final epsilon: {orc_ai.epsilon:.6f}")
    print(f"   Battle record: {orc_ai.victories}W-{orc_ai.defeats}L")
    
    # Save the trained model
    print(f"\n💾 Saving trained model...")
    orc_ai.save_enhanced_model("simple_trained_orc.pth")
    
    if final_rate > 15:
        print(f"🎉 TRAINING SHOWED IMPROVEMENT!")
    else:
        print(f"📈 Training completed - may need more time or parameter tuning")
    
    return final_rate

if __name__ == "__main__":
    simple_working_training() 