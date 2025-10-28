#!/usr/bin/env python3
"""
Quick Detailed Battle Analysis
"""

import torch
import numpy as np
import random
import time
from warhammer_ai_agent import WarhammerAIAgent
from mass_training_system import TrainingBattle

def analyze_ai_decision(agent, state, action, faction_name):
    """Analyze AI decision with Q-values"""
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = agent.q_network(state_tensor).squeeze().numpy()
    
    # Calculate confidence
    sorted_q = np.sort(q_values)
    confidence = float(sorted_q[-1] - sorted_q[-2]) if len(q_values) > 1 else 0.0
    
    # Get top 3 actions
    top_indices = np.argsort(q_values)[-3:][::-1]
    
    return {
        'faction': faction_name,
        'chosen_action': int(action),
        'chosen_q_value': float(q_values[action]),
        'confidence': confidence,
        'top_3_q_values': [(int(idx), float(q_values[idx])) for idx in top_indices],
        'all_q_values': [float(q) for q in q_values]
    }

def get_action_name(action):
    """Get action name"""
    actions = {
        0: "Move North", 1: "Move NE", 2: "Move East", 3: "Move SE",
        4: "Move South", 5: "Move SW", 6: "Move West", 7: "Move NW",
        8: "Artillery Strike", 9: "Cavalry Charge", 10: "Defensive Formation",
        11: "Flanking Maneuver", 12: "Mass Shooting", 13: "Special Tactic A",
        14: "Special Tactic B"
    }
    return actions.get(action, f"Action {action}")

def run_detailed_battle():
    print("🔍 DETAILED AI BATTLE ANALYSIS")
    print("=" * 60)
    
    # Load trained agents
    empire_agent = WarhammerAIAgent(state_size=50, action_size=15, lr=0.001)
    empire_agent.epsilon = 0.0
    
    orc_agent = WarhammerAIAgent(state_size=50, action_size=15, lr=0.002)
    orc_agent.epsilon = 0.0
    
    # Load models
    try:
        checkpoint = torch.load('empire_massive_300k_final.pth', map_location='cpu', weights_only=False)
        empire_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        print("✅ Empire AI loaded")
    except:
        checkpoint = torch.load('empire_massive_300000.pth', map_location='cpu', weights_only=False)
        empire_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        print("✅ Empire AI loaded from 300k")
    
    try:
        checkpoint = torch.load('orc_massive_300k_final.pth', map_location='cpu', weights_only=False)
        orc_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        print("✅ Orc AI loaded")
    except:
        checkpoint = torch.load('orc_massive_300000.pth', map_location='cpu', weights_only=False)
        orc_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        print("✅ Orc AI loaded from 300k")
    
    # Initialize battle
    battle = TrainingBattle()
    battle.create_armies()
    
    # Initial state
    empire_units = [u for u in battle.units if u.faction == "nuln" and u.is_alive]
    orc_units = [u for u in battle.units if u.faction == "orcs" and u.is_alive]
    
    print(f"\n📊 INITIAL DEPLOYMENT:")
    print(f"Empire: {len(empire_units)} units")
    print(f"Orc: {len(orc_units)} units")
    
    turn = 1
    max_turns = 15
    
    # Battle log
    battle_log = []
    
    while turn <= max_turns:
        print(f"\n⚔️ TURN {turn} DETAILED ANALYSIS")
        print("-" * 50)
        
        # Get battle state
        state_vector = battle.get_ai_state()
        print(f"State vector: {len(state_vector)} features")
        print(f"First 10 state values: {[f'{x:.3f}' for x in state_vector[:10]]}")
        
        # Pre-turn unit counts
        pre_empire = len([u for u in battle.units if u.faction == "nuln" and u.is_alive])
        pre_orc = len([u for u in battle.units if u.faction == "orcs" and u.is_alive])
        
        # Empire AI decision
        empire_action = empire_agent.act(state_vector)
        empire_analysis = analyze_ai_decision(empire_agent, state_vector, empire_action, "Empire")
        
        print(f"\n🔵 EMPIRE AI DECISION:")
        print(f"   Chosen: {get_action_name(empire_action)} (Action {empire_action})")
        print(f"   Q-Value: {empire_analysis['chosen_q_value']:.4f}")
        print(f"   Confidence: {empire_analysis['confidence']:.4f}")
        print(f"   Top 3 choices:")
        for i, (act, q_val) in enumerate(empire_analysis['top_3_q_values']):
            print(f"     {i+1}. {get_action_name(act)} (Q={q_val:.4f})")
        
        # Orc AI decision
        orc_action = orc_agent.act(state_vector)
        orc_analysis = analyze_ai_decision(orc_agent, state_vector, orc_action, "Orc")
        
        print(f"\n🟢 ORC AI DECISION:")
        print(f"   Chosen: {get_action_name(orc_action)} (Action {orc_action})")
        print(f"   Q-Value: {orc_analysis['chosen_q_value']:.4f}")
        print(f"   Confidence: {orc_analysis['confidence']:.4f}")
        print(f"   Top 3 choices:")
        for i, (act, q_val) in enumerate(orc_analysis['top_3_q_values']):
            print(f"     {i+1}. {get_action_name(act)} (Q={q_val:.4f})")
        
        # Use actions to influence battle
        random.seed(empire_action * turn + orc_action * turn * 2)
        
        # Execute battle turn
        battle.simulate_turn()
        
        # Post-turn analysis
        post_empire = len([u for u in battle.units if u.faction == "nuln" and u.is_alive])
        post_orc = len([u for u in battle.units if u.faction == "orcs" and u.is_alive])
        
        empire_casualties = pre_empire - post_empire
        orc_casualties = pre_orc - post_orc
        
        print(f"\n📈 TURN {turn} BATTLE RESULTS:")
        print(f"Empire casualties: {empire_casualties} units")
        print(f"Orc casualties: {orc_casualties} units")
        print(f"Empire remaining: {post_empire} units")
        print(f"Orc remaining: {post_orc} units")
        print(f"Battle momentum: {post_empire - post_orc:+d}")
        
        # Log turn data
        turn_data = {
            'turn': turn,
            'pre_battle': {'empire': pre_empire, 'orc': pre_orc},
            'post_battle': {'empire': post_empire, 'orc': post_orc},
            'casualties': {'empire': empire_casualties, 'orc': orc_casualties},
            'empire_decision': empire_analysis,
            'orc_decision': orc_analysis,
            'state_summary': {
                'first_10_features': list(state_vector[:10]),
                'state_mean': float(np.mean(state_vector)),
                'state_std': float(np.std(state_vector))
            }
        }
        battle_log.append(turn_data)
        
        # Check victory conditions
        if post_empire == 0 and post_orc > 0:
            print(f"\n🏆 ORC VICTORY! Empire eliminated on turn {turn}")
            break
        elif post_orc == 0 and post_empire > 0:
            print(f"\n🏆 EMPIRE VICTORY! Orcs eliminated on turn {turn}")
            break
        elif post_empire == 0 and post_orc == 0:
            print(f"\n💀 MUTUAL ANNIHILATION on turn {turn}")
            break
        
        turn += 1
        time.sleep(0.5)  # Brief pause
    
    # Handle time limit
    if turn > max_turns:
        print(f"\n⏰ BATTLE REACHED TURN LIMIT")
        battle.calculate_final_scores()
        winner = battle.get_winner()
        print(f"Final winner by points: {winner}")
    
    # Battle summary
    print(f"\n📋 DETAILED BATTLE SUMMARY:")
    print(f"Duration: {len(battle_log)} turns")
    
    total_empire_casualties = sum(t['casualties']['empire'] for t in battle_log)
    total_orc_casualties = sum(t['casualties']['orc'] for t in battle_log)
    
    empire_avg_confidence = sum(t['empire_decision']['confidence'] for t in battle_log) / len(battle_log)
    orc_avg_confidence = sum(t['orc_decision']['confidence'] for t in battle_log) / len(battle_log)
    
    print(f"Total Empire casualties: {total_empire_casualties}")
    print(f"Total Orc casualties: {total_orc_casualties}")
    print(f"Empire AI avg confidence: {empire_avg_confidence:.4f}")
    print(f"Orc AI avg confidence: {orc_avg_confidence:.4f}")
    
    # Action patterns
    empire_actions = [t['empire_decision']['chosen_action'] for t in battle_log]
    orc_actions = [t['orc_decision']['chosen_action'] for t in battle_log]
    
    from collections import Counter
    empire_counter = Counter(empire_actions)
    orc_counter = Counter(orc_actions)
    
    print(f"\n🎲 ACTION PATTERNS:")
    print(f"Empire most used: {empire_counter.most_common(3)}")
    print(f"Orc most used: {orc_counter.most_common(3)}")
    
    # Save detailed log
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"detailed_battle_log_{timestamp}.json"
    
    with open(log_file, 'w') as f:
        json.dump(battle_log, f, indent=2)
    
    print(f"\n💾 Detailed log saved to: {log_file}")
    
    return battle_log

if __name__ == "__main__":
    run_detailed_battle() 