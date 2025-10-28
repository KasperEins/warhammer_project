#!/usr/bin/env python3
"""
Detailed Battle Logger - Comprehensive AI Battle Analysis
Captures every aspect of AI vs AI battles for improvement
"""

import torch
import numpy as np
import random
import time
import json
from datetime import datetime
from warhammer_ai_agent import WarhammerAIAgent
from mass_training_system import TrainingBattle

class DetailedBattleLogger:
    """Comprehensive battle logging and analysis system"""
    
    def __init__(self):
        self.empire_agent = None
        self.orc_agent = None
        
    def load_trained_agents(self):
        """Load the fully trained agents with detailed logging"""
        print("🔍 LOADING AGENTS FOR DETAILED ANALYSIS")
        print("=" * 50)
        
        # Load Empire AI
        self.empire_agent = WarhammerAIAgent(state_size=50, action_size=15, lr=0.001)
        self.empire_agent.epsilon = 0.0
        
        try:
            checkpoint = torch.load('empire_massive_300k_final.pth', map_location='cpu', weights_only=False)
            if 'q_network_state_dict' in checkpoint:
                self.empire_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            print("✅ Empire AI loaded - entering analysis mode")
        except FileNotFoundError:
            try:
                checkpoint = torch.load('empire_massive_300000.pth', map_location='cpu', weights_only=False)
                if 'q_network_state_dict' in checkpoint:
                    self.empire_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                print("✅ Empire AI loaded from 300k checkpoint")
            except FileNotFoundError:
                print("❌ Empire model not found")
                return False
        
        # Load Orc AI
        self.orc_agent = WarhammerAIAgent(state_size=50, action_size=15, lr=0.002)
        self.orc_agent.epsilon = 0.0
        
        try:
            checkpoint = torch.load('orc_massive_300k_final.pth', map_location='cpu', weights_only=False)
            if 'q_network_state_dict' in checkpoint:
                self.orc_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            print("✅ Orc AI loaded - entering analysis mode")
        except FileNotFoundError:
            try:
                checkpoint = torch.load('orc_massive_300000.pth', map_location='cpu', weights_only=False)
                if 'q_network_state_dict' in checkpoint:
                    self.orc_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                print("✅ Orc AI loaded from 300k checkpoint")
            except FileNotFoundError:
                print("❌ Orc model not found")
                return False
        
        return True
    
    def analyze_ai_decision(self, agent, state, action, faction_name):
        """Analyze AI decision-making process"""
        # Get Q-values for all actions
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent.q_network(state_tensor).squeeze().numpy()
        
        # Analyze decision
        decision_analysis = {
            'faction': faction_name,
            'chosen_action': int(action),
            'chosen_q_value': float(q_values[action]),
            'all_q_values': [float(q) for q in q_values],
            'top_3_actions': [],
            'confidence': 0.0,
            'exploration': agent.epsilon > 0
        }
        
        # Get top 3 actions
        top_3_indices = np.argsort(q_values)[-3:][::-1]
        for idx in top_3_indices:
            decision_analysis['top_3_actions'].append({
                'action': int(idx),
                'q_value': float(q_values[idx]),
                'description': self.get_action_description(idx)
            })
        
        # Calculate confidence (difference between best and second-best)
        if len(q_values) > 1:
            sorted_q = np.sort(q_values)
            decision_analysis['confidence'] = float(sorted_q[-1] - sorted_q[-2])
        
        return decision_analysis
    
    def get_action_description(self, action):
        """Get detailed description of action"""
        action_descriptions = {
            0: "Move North - Advance army formation northward",
            1: "Move Northeast - Diagonal advance northeast", 
            2: "Move East - Advance army formation eastward",
            3: "Move Southeast - Diagonal advance southeast",
            4: "Move South - Retreat or reposition southward",
            5: "Move Southwest - Diagonal retreat southwest",
            6: "Move West - Advance army formation westward", 
            7: "Move Northwest - Diagonal advance northwest",
            8: "Artillery Strike - Concentrated artillery barrage",
            9: "Cavalry Charge - Aggressive cavalry assault",
            10: "Defensive Formation - Form defensive line",
            11: "Flanking Maneuver - Execute flanking attack",
            12: "Mass Shooting - Coordinated ranged volley",
            13: "Special Tactic A - Advanced battlefield maneuver",
            14: "Special Tactic B - Elite unit deployment"
        }
        return action_descriptions.get(action, f"Unknown Action {action}")
    
    def analyze_battle_state(self, battle, turn):
        """Comprehensive analysis of current battle state"""
        state = battle.get_ai_state()
        
        # Count living units by faction
        empire_units = [u for u in battle.units if u.faction == "nuln" and u.is_alive]
        orc_units = [u for u in battle.units if u.faction == "orcs" and u.is_alive]
        
        # Calculate faction strengths
        empire_strength = sum(getattr(u, 'models', 1) for u in empire_units)
        orc_strength = sum(getattr(u, 'models', 1) for u in orc_units)
        
        # Calculate total points remaining
        empire_points = sum(getattr(u, 'points_value', 100) for u in empire_units)
        orc_points = sum(getattr(u, 'points_value', 100) for u in orc_units)
        
        battle_analysis = {
            'turn': turn,
            'empire_units_alive': len(empire_units),
            'orc_units_alive': len(orc_units),
            'empire_total_models': empire_strength,
            'orc_total_models': orc_strength,
            'empire_points_remaining': empire_points,
            'orc_points_remaining': orc_points,
            'empire_advantage': empire_strength - orc_strength,
            'point_ratio': empire_points / max(orc_points, 1),
            'battle_phase': self.determine_battle_phase(empire_strength, orc_strength, turn),
            'state_vector': state.tolist(),
            'unit_details': {
                'empire': self.get_unit_details(empire_units),
                'orc': self.get_unit_details(orc_units)
            }
        }
        
        return battle_analysis
    
    def get_unit_details(self, units):
        """Get detailed information about units"""
        unit_details = []
        for unit in units:
            detail = {
                'name': unit.name,
                'models': getattr(unit, 'models', 1),
                'max_models': getattr(unit, 'starting_models', getattr(unit, 'models', 1)),
                'health_percent': (getattr(unit, 'models', 1) / max(getattr(unit, 'starting_models', 1), 1)) * 100,
                'is_alive': unit.is_alive,
                'unit_type': getattr(unit, 'unit_type', 'unknown'),
                'points_value': getattr(unit, 'points_value', 100)
            }
            unit_details.append(detail)
        return unit_details
    
    def determine_battle_phase(self, empire_str, orc_str, turn):
        """Determine current phase of battle"""
        total_str = empire_str + orc_str
        
        if turn <= 3:
            return "Opening"
        elif turn <= 8:
            if total_str > 30:
                return "Early Engagement" 
            else:
                return "Heavy Combat"
        elif turn <= 15:
            if total_str > 15:
                return "Mid Battle"
            else:
                return "Decisive Phase"
        else:
            return "Endgame"
    
    def run_detailed_battle(self, battle_num):
        """Run a single battle with maximum detail logging"""
        print(f"\n{'='*80}")
        print(f"🔍 DETAILED BATTLE ANALYSIS #{battle_num}")
        print(f"{'='*80}")
        
        # Initialize battle
        battle = TrainingBattle()
        battle.create_armies()
        
        # Initialize logging structures
        battle_log = {
            'battle_number': battle_num,
            'start_time': datetime.now().isoformat(),
            'turns': [],
            'final_result': None,
            'battle_summary': {}
        }
        
        turn = 1
        max_turns = 25
        
        # Log initial state
        initial_analysis = self.analyze_battle_state(battle, 0)
        print(f"\n📊 INITIAL DEPLOYMENT")
        print(f"Empire: {initial_analysis['empire_units_alive']} units, {initial_analysis['empire_total_models']} models")
        print(f"Orc: {initial_analysis['orc_units_alive']} units, {initial_analysis['orc_total_models']} models")
        
        while turn <= max_turns:
            print(f"\n⚔️ TURN {turn} ANALYSIS")
            print("-" * 40)
            
            # Pre-turn battle state
            pre_turn_state = self.analyze_battle_state(battle, turn)
            state_vector = battle.get_ai_state()
            
            # Empire AI Decision
            empire_action = self.empire_agent.act(state_vector)
            empire_decision = self.analyze_ai_decision(self.empire_agent, state_vector, empire_action, "Empire")
            
            print(f"🔵 Empire Decision: {self.get_action_description(empire_action)}")
            print(f"   Q-Value: {empire_decision['chosen_q_value']:.3f}, Confidence: {empire_decision['confidence']:.3f}")
            if len(empire_decision['top_3_actions']) > 1:
                print(f"   Top alternative: {empire_decision['top_3_actions'][1]['description']} ({empire_decision['top_3_actions'][1]['q_value']:.3f})")
            
            # Orc AI Decision  
            orc_action = self.orc_agent.act(state_vector)
            orc_decision = self.analyze_ai_decision(self.orc_agent, state_vector, orc_action, "Orc")
            
            print(f"🟢 Orc Decision: {self.get_action_description(orc_action)}")
            print(f"   Q-Value: {orc_decision['chosen_q_value']:.3f}, Confidence: {orc_decision['confidence']:.3f}")
            if len(orc_decision['top_3_actions']) > 1:
                print(f"   Top alternative: {orc_decision['top_3_actions'][1]['description']} ({orc_decision['top_3_actions'][1]['q_value']:.3f})")
            
            # Use actions to influence battle
            random.seed(empire_action * turn + orc_action * turn * 2)
            
            # Execute battle turn
            battle.simulate_turn()
            
            # Post-turn analysis
            post_turn_state = self.analyze_battle_state(battle, turn)
            
            # Calculate turn effects
            empire_model_loss = pre_turn_state['empire_total_models'] - post_turn_state['empire_total_models']
            orc_model_loss = pre_turn_state['orc_total_models'] - post_turn_state['orc_total_models']
            
            print(f"\n📈 TURN {turn} RESULTS:")
            print(f"Empire casualties: {empire_model_loss} models")
            print(f"Orc casualties: {orc_model_loss} models")
            print(f"Empire remaining: {post_turn_state['empire_units_alive']} units ({post_turn_state['empire_total_models']} models)")
            print(f"Orc remaining: {post_turn_state['orc_units_alive']} units ({post_turn_state['orc_total_models']} models)")
            print(f"Battle phase: {post_turn_state['battle_phase']}")
            
            # Store turn log
            turn_log = {
                'turn': turn,
                'pre_turn_state': pre_turn_state,
                'empire_decision': empire_decision,
                'orc_decision': orc_decision,
                'post_turn_state': post_turn_state,
                'casualties': {
                    'empire_models_lost': empire_model_loss,
                    'orc_models_lost': orc_model_loss
                },
                'battle_momentum': post_turn_state['empire_advantage']
            }
            battle_log['turns'].append(turn_log)
            
            # Check victory conditions
            empire_alive = post_turn_state['empire_units_alive']
            orc_alive = post_turn_state['orc_units_alive']
            
            if empire_alive == 0 and orc_alive > 0:
                print(f"\n🏆 ORC VICTORY! Empire eliminated on turn {turn}")
                battle_log['final_result'] = {'winner': 'Orc', 'turn': turn, 'type': 'Total Victory'}
                break
            elif orc_alive == 0 and empire_alive > 0:
                print(f"\n🏆 EMPIRE VICTORY! Orcs eliminated on turn {turn}")
                battle_log['final_result'] = {'winner': 'Empire', 'turn': turn, 'type': 'Total Victory'}
                break
            elif empire_alive == 0 and orc_alive == 0:
                print(f"\n💀 MUTUAL ANNIHILATION on turn {turn}")
                battle_log['final_result'] = {'winner': 'Draw', 'turn': turn, 'type': 'Mutual Destruction'}
                break
            
            turn += 1
            time.sleep(0.3)  # Brief pause for readability
        
        # Handle time limit
        if turn > max_turns:
            battle.calculate_final_scores()
            winner = battle.get_winner()
            final_state = self.analyze_battle_state(battle, max_turns)
            
            print(f"\n⏰ BATTLE DURATION LIMIT REACHED")
            print(f"Final score - Empire: {final_state['empire_points_remaining']}, Orc: {final_state['orc_points_remaining']}")
            
            battle_log['final_result'] = {
                'winner': winner if winner else 'Draw',
                'turn': max_turns,
                'type': 'Time Limit',
                'final_scores': {
                    'empire_points': final_state['empire_points_remaining'],
                    'orc_points': final_state['orc_points_remaining']
                }
            }
        
        # Generate battle summary
        battle_summary = self.generate_battle_summary(battle_log)
        battle_log['battle_summary'] = battle_summary
        
        print(f"\n📋 BATTLE SUMMARY:")
        print(f"Winner: {battle_log['final_result']['winner']}")
        print(f"Duration: {len(battle_log['turns'])} turns")
        print(f"Total Empire casualties: {battle_summary['total_empire_casualties']}")
        print(f"Total Orc casualties: {battle_summary['total_orc_casualties']}")
        print(f"Empire AI avg confidence: {battle_summary['empire_avg_confidence']:.3f}")
        print(f"Orc AI avg confidence: {battle_summary['orc_avg_confidence']:.3f}")
        
        return battle_log
    
    def generate_battle_summary(self, battle_log):
        """Generate comprehensive battle summary"""
        turns = battle_log['turns']
        
        total_empire_casualties = sum(turn['casualties']['empire_models_lost'] for turn in turns)
        total_orc_casualties = sum(turn['casualties']['orc_models_lost'] for turn in turns)
        
        empire_confidences = [turn['empire_decision']['confidence'] for turn in turns]
        orc_confidences = [turn['orc_decision']['confidence'] for turn in turns]
        
        empire_actions = [turn['empire_decision']['chosen_action'] for turn in turns]
        orc_actions = [turn['orc_decision']['chosen_action'] for turn in turns]
        
        return {
            'total_empire_casualties': total_empire_casualties,
            'total_orc_casualties': total_orc_casualties,
            'empire_avg_confidence': sum(empire_confidences) / len(empire_confidences) if empire_confidences else 0,
            'orc_avg_confidence': sum(orc_confidences) / len(orc_confidences) if orc_confidences else 0,
            'empire_action_distribution': {str(action): empire_actions.count(action) for action in set(empire_actions)},
            'orc_action_distribution': {str(action): orc_actions.count(action) for action in set(orc_actions)},
            'battle_duration': len(turns),
            'decisive_turn': self.find_decisive_turn(turns) if len(turns) > 1 else 1
        }
    
    def find_decisive_turn(self, turns):
        """Find the turn where battle momentum shifted decisively"""
        max_momentum_shift = 0
        decisive_turn = 1
        
        for i in range(1, len(turns)):
            prev_momentum = turns[i-1]['battle_momentum']
            curr_momentum = turns[i]['battle_momentum']
            momentum_shift = abs(curr_momentum - prev_momentum)
            
            if momentum_shift > max_momentum_shift:
                max_momentum_shift = momentum_shift
                decisive_turn = i + 1
        
        return decisive_turn
    
    def save_detailed_logs(self, battle_logs):
        """Save all battle logs to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_battle_logs_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(battle_logs, f, indent=2)
        
        print(f"\n💾 Detailed logs saved to {filename}")
        return filename
    
    def run_detailed_analysis_campaign(self, num_battles=3):
        """Run detailed analysis campaign"""
        print("🔍 LAUNCHING DETAILED BATTLE ANALYSIS CAMPAIGN")
        print("=" * 60)
        
        if not self.load_trained_agents():
            print("❌ Failed to load models")
            return None, None
        
        all_battle_logs = []
        
        for battle_num in range(1, num_battles + 1):
            battle_log = self.run_detailed_battle(battle_num)
            all_battle_logs.append(battle_log)
        
        # Save logs
        log_file = self.save_detailed_logs(all_battle_logs)
        
        # Generate overall analysis
        self.generate_campaign_analysis(all_battle_logs)
        
        return all_battle_logs, log_file
    
    def generate_campaign_analysis(self, all_battle_logs):
        """Generate analysis across all battles"""
        print(f"\n🎯 CAMPAIGN ANALYSIS ACROSS {len(all_battle_logs)} BATTLES")
        print("=" * 60)
        
        empire_wins = sum(1 for log in all_battle_logs if log['final_result']['winner'] == 'Empire')
        orc_wins = sum(1 for log in all_battle_logs if log['final_result']['winner'] == 'Orc')
        draws = sum(1 for log in all_battle_logs if log['final_result']['winner'] == 'Draw')
        
        avg_duration = sum(len(log['turns']) for log in all_battle_logs) / len(all_battle_logs)
        
        # AI Performance Analysis
        all_empire_confidences = []
        all_orc_confidences = []
        
        for log in all_battle_logs:
            all_empire_confidences.extend([turn['empire_decision']['confidence'] for turn in log['turns']])
            all_orc_confidences.extend([turn['orc_decision']['confidence'] for turn in log['turns']])
        
        print(f"📊 Win Rates:")
        print(f"   Empire: {empire_wins}/{len(all_battle_logs)} ({empire_wins/len(all_battle_logs)*100:.1f}%)")
        print(f"   Orc: {orc_wins}/{len(all_battle_logs)} ({orc_wins/len(all_battle_logs)*100:.1f}%)")
        print(f"   Draws: {draws}/{len(all_battle_logs)} ({draws/len(all_battle_logs)*100:.1f}%)")
        
        print(f"\n🧠 AI Decision Analysis:")
        print(f"   Empire avg confidence: {sum(all_empire_confidences)/len(all_empire_confidences):.3f}")
        print(f"   Orc avg confidence: {sum(all_orc_confidences)/len(all_orc_confidences):.3f}")
        print(f"   Average battle duration: {avg_duration:.1f} turns")
        
        # Action pattern analysis
        print(f"\n🎲 Most Common Actions:")
        empire_actions = []
        orc_actions = []
        
        for log in all_battle_logs:
            for turn in log['turns']:
                empire_actions.append(turn['empire_decision']['chosen_action'])
                orc_actions.append(turn['orc_decision']['chosen_action'])
        
        from collections import Counter
        empire_counter = Counter(empire_actions)
        orc_counter = Counter(orc_actions)
        
        print(f"   Empire top 3: {empire_counter.most_common(3)}")
        print(f"   Orc top 3: {orc_counter.most_common(3)}")

def main():
    """Run detailed battle analysis"""
    logger = DetailedBattleLogger()
    battle_logs, log_file = logger.run_detailed_analysis_campaign(num_battles=3)
    
    if battle_logs and log_file:
        print(f"\n✅ DETAILED ANALYSIS COMPLETE")
        print(f"📁 Full logs saved in: {log_file}")
        print(f"🔍 Use these logs to:")
        print(f"   • Identify AI decision patterns")
        print(f"   • Find training improvements")
        print(f"   • Optimize battle mechanics")
        print(f"   • Debug AI behavior")

if __name__ == "__main__":
    main() 