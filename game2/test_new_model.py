#!/usr/bin/env python3
"""
Test the newly trained model from simple_working_training.py
"""

import torch
from warhammer_ai_agent import WarhammerAIAgent
from mass_training_system import TrainingBattle
from collections import deque

def test_new_trained_model():
    """Test the newly trained model."""
    print("🆕 TESTING NEWLY TRAINED MODEL")
    print("=" * 40)
    
    # Load the NEW trained model
    print("Loading simple_trained_orc.pth...")
    try:
        checkpoint = torch.load('simple_trained_orc.pth', map_location='cpu', weights_only=False)
        print(f"✅ New model loaded:")
        print(f"   Memory experiences: {len(checkpoint.get('memory', []))}")
        print(f"   Epsilon: {checkpoint.get('epsilon', 'Unknown')}")
        print(f"   Training record: {checkpoint.get('victories', 0)}W-{checkpoint.get('defeats', 0)}L")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create and load the trained AI
    orc_ai = WarhammerAIAgent(state_size=50, action_size=15)
    
    # Load network states
    orc_ai.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    orc_ai.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    orc_ai.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    orc_ai.epsilon = 0.01  # Low exploration for testing
    
    # Restore memory
    if 'memory' in checkpoint and checkpoint['memory']:
        print(f"🧠 Restoring {len(checkpoint['memory'])} experiences...")
        orc_ai.memory = deque(checkpoint['memory'], maxlen=checkpoint.get('memory_maxlen', 50000))
    
    print(f"✅ AI ready for testing:")
    print(f"   Testing epsilon: {orc_ai.epsilon}")
    print(f"   Memory size: {len(orc_ai.memory):,}")
    
    # Test different scenarios
    test_scenarios = [
        ("Exploitation Only", 0.0),      # Pure learned behavior
        ("Low Exploration", 0.01),       # Minimal exploration
        ("Some Exploration", 0.05),      # Light exploration
    ]
    
    print(f"\n🎮 TESTING SCENARIOS")
    print("=" * 30)
    
    all_results = {}
    
    for scenario_name, test_epsilon in test_scenarios:
        print(f"\n🔬 {scenario_name} (ε={test_epsilon})")
        print("-" * 25)
        
        orc_ai.epsilon = test_epsilon
        wins = 0
        
        for i in range(30):  # 30 battles per scenario
            battle = TrainingBattle()
            battle.create_armies()
            
            # Run battle
            for turn in range(20):
                state = battle.get_ai_state()
                action = orc_ai.act(state)
                battle.simulate_turn()
                
                # Check if battle ended
                orc_alive = any(u.is_alive and u.faction == "orcs" for u in battle.units)
                empire_alive = any(u.is_alive and u.faction == "nuln" for u in battle.units)
                
                if not orc_alive or not empire_alive:
                    break
            
            battle.calculate_final_scores()
            winner = battle.get_winner()
            
            if winner == 'orc':
                wins += 1
            
            if (i + 1) % 10 == 0:
                current_rate = (wins / (i + 1)) * 100
                print(f"   After {i + 1:2d} battles: {current_rate:5.1f}%")
        
        final_rate = (wins / 30) * 100
        all_results[scenario_name] = final_rate
        print(f"   FINAL: {final_rate:.1f}% ({wins}/30)")
    
    # Compare with baseline
    print(f"\n📊 RESULTS COMPARISON")
    print("=" * 30)
    
    best_scenario = max(all_results, key=all_results.get)
    best_rate = all_results[best_scenario]
    
    for scenario, rate in all_results.items():
        improvement = "🔥 EXCELLENT" if rate > 20 else "🎯 IMPROVED" if rate > 10 else "📈 SOME LEARNING" if rate > 5 else "⚠️ BASELINE"
        print(f"   {scenario:15}: {rate:5.1f}% {improvement}")
    
    print(f"\n🏆 BEST: {best_scenario} at {best_rate:.1f}%")
    
    # Analysis
    print(f"\n🧪 TRAINING ANALYSIS")
    print("=" * 25)
    print(f"   Memory from training: {len(orc_ai.memory):,} experiences")
    print(f"   Training win rate was: 16.0% (reference)")
    
    if best_rate > 15:
        print("   ✅ Model shows REAL learning - training was effective!")
    elif best_rate > 10:
        print("   📈 Model shows SOME learning - needs more training")
    elif best_rate > 5:
        print("   🤔 Minimal learning detected - check parameters")
    else:
        print("   ⚠️ No significant learning - training may need fixes")
    
    return all_results

if __name__ == "__main__":
    test_new_trained_model() 