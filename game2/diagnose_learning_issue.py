#!/usr/bin/env python3
"""
🔬 ORC AI LEARNING DIAGNOSTICS
=============================
Diagnose why the Orc AI hit a learning plateau
"""

import torch
import json
import glob
import numpy as np
from mass_training_system import TrainingAI, TrainingBattle
import random

def load_latest_orc_model():
    """Load the latest Orc AI model"""
    model_files = glob.glob("orc_ai_mass_trained_*.pth")
    if not model_files:
        return None, 0
    
    # Get the latest model
    game_counts = []
    for file in model_files:
        try:
            game_count = int(file.split('_')[-1].split('.')[0])
            game_counts.append((game_count, file))
        except:
            continue
    
    if not game_counts:
        return None, 0
    
    latest_count, latest_file = max(game_counts)
    
    # Load the model
    ai = TrainingAI()
    ai.load_model(latest_file)
    
    return ai, latest_count

def test_ai_behavior(ai, num_tests=100):
    """Test AI decision making in various scenarios"""
    print("🧪 TESTING AI DECISION MAKING")
    print("=" * 40)
    
    action_counts = {}
    state_variety = []
    
    for i in range(num_tests):
        # Create different battle scenarios
        battle = TrainingBattle()
        battle.create_armies()
        
        # Simulate some battle progression
        for turn in range(random.randint(1, 4)):
            battle.simulate_turn()
        
        state = battle.get_ai_state()
        action = ai.act(state)
        
        # Track actions
        action_counts[action] = action_counts.get(action, 0) + 1
        state_variety.append(np.mean(state))
    
    print(f"📊 Action Distribution (over {num_tests} tests):")
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items()):
        percentage = (count / total_actions) * 100
        print(f"   Action {action}: {count} times ({percentage:.1f}%)")
    
    print(f"\n🎯 Action Diversity: {len(action_counts)} different actions used")
    
    # Check if AI is too deterministic
    max_action_percent = max(action_counts.values()) / total_actions * 100
    if max_action_percent > 80:
        print(f"⚠️  AI is too deterministic! Using action {max(action_counts, key=action_counts.get)} {max_action_percent:.1f}% of the time")
    elif max_action_percent > 60:
        print(f"🟡 AI is somewhat repetitive, using one action {max_action_percent:.1f}% of the time")
    else:
        print(f"✅ AI shows good action diversity")
    
    return action_counts

def test_learning_capability(ai):
    """Test if AI can still learn from experiences"""
    print("\n🧠 TESTING LEARNING CAPABILITY")
    print("=" * 40)
    
    # Store original memory size
    original_memory_size = len(ai.memory)
    print(f"📝 Current memory size: {original_memory_size}")
    
    # Test memory storage
    dummy_experience = ([0.5] * 50, 5, 10.0, [0.6] * 50, False)
    ai.remember(*dummy_experience)
    
    new_memory_size = len(ai.memory)
    if new_memory_size > original_memory_size:
        print("✅ AI can store new experiences")
    else:
        print("❌ AI memory not updating properly")
    
    # Test if replay works
    if len(ai.memory) > 32:
        try:
            ai.replay(batch_size=32)
            print("✅ AI can perform experience replay")
        except Exception as e:
            print(f"❌ Replay failed: {e}")
    else:
        print("⚠️  Not enough experiences for replay testing")
    
    return new_memory_size

def analyze_epsilon_and_exploration(ai):
    """Analyze exploration vs exploitation balance"""
    print("\n🎲 EXPLORATION ANALYSIS")
    print("=" * 40)
    
    epsilon = ai.epsilon
    print(f"🔍 Current epsilon: {epsilon:.6f}")
    
    if epsilon < 0.01:
        print("❌ CRITICAL: Epsilon too low! No exploration happening")
        print("   Recommendation: Reset epsilon to 0.3-0.5")
    elif epsilon < 0.1:
        print("⚠️  Epsilon very low. Minimal exploration")
        print("   Recommendation: Increase epsilon to 0.2-0.3")
    elif epsilon < 0.3:
        print("🟡 Epsilon moderate. Some exploration happening")
    else:
        print("✅ Epsilon good for exploration")
    
    return epsilon

def check_battle_environment():
    """Check if the training environment is working correctly"""
    print("\n⚔️  BATTLE ENVIRONMENT CHECK")
    print("=" * 40)
    
    battle = TrainingBattle()
    battle.create_armies()
    
    # Check initial state
    initial_empire = len([u for u in battle.units if u.faction == "nuln" and u.is_alive])
    initial_orcs = len([u for u in battle.units if u.faction == "orcs" and u.is_alive])
    
    print(f"🔵 Empire units: {initial_empire}")
    print(f"🟢 Orc units: {initial_orcs}")
    
    # Simulate some turns
    for turn in range(3):
        battle.simulate_turn()
    
    final_empire = len([u for u in battle.units if u.faction == "nuln" and u.is_alive])
    final_orcs = len([u for u in battle.units if u.faction == "orcs" and u.is_alive])
    
    print(f"After 3 turns:")
    print(f"🔵 Empire units: {final_empire}")
    print(f"🟢 Orc units: {final_orcs}")
    
    if initial_empire == final_empire and initial_orcs == final_orcs:
        print("❌ PROBLEM: No battle progression! Units not taking damage")
    else:
        print("✅ Battle simulation working")
    
    # Test winner determination
    winner = battle.get_winner()
    print(f"🏆 Battle winner: {winner}")

def main():
    """Run full diagnostics"""
    print("🔬 ORC AI LEARNING DIAGNOSTICS")
    print("=" * 50)
    
    # Load latest model
    ai, game_count = load_latest_orc_model()
    
    if not ai:
        print("❌ No Orc AI model found!")
        return
    
    print(f"✅ Loaded Orc AI from {game_count:,} games")
    print()
    
    # Run diagnostics
    action_counts = test_ai_behavior(ai)
    memory_size = test_learning_capability(ai)
    epsilon = analyze_epsilon_and_exploration(ai)
    check_battle_environment()
    
    # Final recommendations
    print("\n🎯 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 50)
    
    max_action_use = max(action_counts.values()) / sum(action_counts.values()) * 100
    
    if epsilon < 0.1:
        print("🔧 PRIMARY ISSUE: Epsilon too low")
        print("   ➡️  Reset epsilon to 0.3 for renewed exploration")
    
    if max_action_use > 70:
        print("🔧 SECONDARY ISSUE: AI too deterministic")
        print("   ➡️  Increase exploration or reset learning")
    
    if memory_size < 1000:
        print("🔧 ISSUE: Limited experience memory")
        print("   ➡️  AI needs more diverse experiences")
    
    print("\n💡 SUGGESTED FIXES:")
    print("1. Reset epsilon to 0.3-0.5")
    print("2. Increase learning rate temporarily")
    print("3. Add reward shaping for better learning signals")
    print("4. Consider curriculum learning (easier opponents first)")

if __name__ == "__main__":
    main() 