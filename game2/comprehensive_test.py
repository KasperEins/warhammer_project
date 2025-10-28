#!/usr/bin/env python3
"""
Comprehensive test of the fixed improved training system
"""

import torch
from warhammer_ai_agent import WarhammerAIAgent
from mass_training_system import TrainingBattle
from collections import deque

def comprehensive_test():
    """Comprehensive test of the trained model."""
    print("🎯 COMPREHENSIVE FIXED MODEL TEST")
    print("=" * 50)
    
    # Load checkpoint
    print("Loading trained model...")
    checkpoint = torch.load('orc_ai_fixed_31.pth', map_location='cpu', weights_only=False)
    
    print(f"✅ Checkpoint loaded:")
    print(f"   Memory experiences: {len(checkpoint.get('memory', []))}")
    print(f"   Epsilon: {checkpoint.get('epsilon', 'Unknown')}")
    print(f"   Victories: {checkpoint.get('victories', 0)}")
    print(f"   Defeats: {checkpoint.get('defeats', 0)}")
    
    # Create and load the trained AI
    orc_ai = WarhammerAIAgent(state_size=50, action_size=15)
    
    # Load network states
    orc_ai.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    orc_ai.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    orc_ai.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    orc_ai.epsilon = checkpoint.get('epsilon', 0.4)
    
    # IMPORTANT: Restore the memory experiences
    if 'memory' in checkpoint and checkpoint['memory']:
        print(f"🧠 Restoring {len(checkpoint['memory'])} experiences to memory...")
        orc_ai.memory = deque(checkpoint['memory'], maxlen=checkpoint.get('memory_maxlen', 50000))
        print(f"   Memory restored: {len(orc_ai.memory)} experiences")
    
    # Load other training data
    orc_ai.scores = checkpoint.get('scores', [])
    orc_ai.successful_strategies = checkpoint.get('successful_strategies', [])
    orc_ai.victories = checkpoint.get('victories', 0)
    orc_ai.defeats = checkpoint.get('defeats', 0)
    orc_ai.draws = checkpoint.get('draws', 0)
    
    print(f"✅ Trained AI loaded completely:")
    print(f"   Training epsilon: {orc_ai.epsilon:.6f}")
    print(f"   Memory size: {len(orc_ai.memory):,}")
    print(f"   Battle record: {orc_ai.victories}W-{orc_ai.defeats}L-{orc_ai.draws}D")
    
    # Create baseline AI for comparison
    baseline_ai = WarhammerAIAgent(state_size=50, action_size=15)
    baseline_ai.epsilon = 0.01  # Minimal exploration for fair test
    
    # Test scenarios
    test_scenarios = [
        ("Low Exploration", 0.01),   # Pure exploitation
        ("Medium Exploration", 0.1), # Some exploration
        ("Training Level", orc_ai.epsilon)  # Original training level
    ]
    
    print(f"\n🎮 TESTING DIFFERENT EXPLORATION LEVELS")
    print("=" * 50)
    
    overall_results = {}
    
    for scenario_name, test_epsilon in test_scenarios:
        print(f"\n🔬 {scenario_name} Test (ε={test_epsilon})")
        print("-" * 30)
        
        # Store original epsilon
        original_epsilon = orc_ai.epsilon
        orc_ai.epsilon = test_epsilon
        
        wins = 0
        for i in range(50):  # More battles for better statistics
            battle = TrainingBattle()
            battle.create_armies()
            
            # Battle simulation
            turns = 0
            while turns < 25 and battle.battle_state == "active":
                state = battle.get_ai_state()
                
                # Both AIs act
                orc_action = orc_ai.act(state)
                baseline_action = baseline_ai.act(state)
                
                battle.simulate_turn()
                turns += 1
                
                # Check battle end
                empire_alive = any(u.is_alive and u.faction == "nuln" for u in battle.units)
                orc_alive = any(u.is_alive and u.faction == "orcs" for u in battle.units)
                
                if not empire_alive or not orc_alive:
                    break
            
            battle.calculate_final_scores()
            winner = battle.get_winner()
            
            if winner == 'orc':
                wins += 1
            
            if (i + 1) % 10 == 0:
                current_rate = (wins / (i + 1)) * 100
                print(f"   After {i + 1:2d} battles: {current_rate:5.1f}% win rate")
        
        # Restore epsilon
        orc_ai.epsilon = original_epsilon
        
        final_rate = (wins / 50) * 100
        overall_results[scenario_name] = final_rate
        
        print(f"   FINAL: {final_rate:.1f}% ({wins}/50)")
    
    # Summary
    print(f"\n📊 OVERALL RESULTS SUMMARY")
    print("=" * 50)
    for scenario, rate in overall_results.items():
        status = "🔥 EXCELLENT" if rate > 40 else "🎯 GOOD" if rate > 30 else "📈 IMPROVED" if rate > 20 else "⚠️ BASELINE"
        print(f"   {scenario:15}: {rate:5.1f}% {status}")
    
    # Best performance
    best_scenario = max(overall_results, key=overall_results.get)
    best_rate = overall_results[best_scenario]
    
    print(f"\n🏆 BEST PERFORMANCE: {best_scenario} at {best_rate:.1f}%")
    
    # Training effectiveness analysis
    print(f"\n🧪 TRAINING EFFECTIVENESS ANALYSIS")
    print("=" * 50)
    print(f"   Memory utilization: {len(orc_ai.memory):,}/50,000 ({len(orc_ai.memory)/50000*100:.1f}%)")
    
    if best_rate > 25:
        print("   ✅ Training was EFFECTIVE - AI shows meaningful improvement!")
    elif best_rate > 15:
        print("   📈 Training shows SOME improvement - needs more training time")
    else:
        print("   ⚠️ Training needs optimization - consider adjusting parameters")
    
    return overall_results

if __name__ == "__main__":
    comprehensive_test() 