#!/usr/bin/env python3
"""
🏛️ BIAS RL WARHAMMER: THE OLD WORLD AI TRAINER
===============================================

Advanced AI trainer using imitation learning and bias RL approach
inspired by the MLPrague conference methodology.

Key Features:
- Expert demonstration recording
- Imitation learning from expert gameplay
- Bias RL for tactical decision making
- Reward shaping for TOW-specific metrics
- Progressive learning from expert to autonomous
"""

import random
import time
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import deque
import pickle

@dataclass
class ExpertDemo:
    """Expert demonstration for imitation learning"""
    army_composition: Dict[str, int]
    tactical_decisions: List[str]
    battle_outcome: str
    effectiveness_score: float
    turn_decisions: List[Dict] = None

@dataclass
class TOWGameState:
    """Complete game state for decision making"""
    own_army: Dict[str, int]
    enemy_army: Dict[str, int]
    battle_phase: str
    turn_number: int
    casualties: Dict[str, int]
    tactical_position: float
    morale_state: float

class ExpertDemonstrationRecorder:
    """Records expert TOW gameplay for imitation learning"""
    
    def __init__(self):
        self.demonstrations = []
        self.current_demo = None
        
    def start_recording(self, army_comp: Dict[str, int]):
        """Start recording a new expert demonstration"""
        self.current_demo = {
            'army_composition': army_comp.copy(),
            'tactical_decisions': [],
            'game_states': [],
            'turn_decisions': []
        }
        
    def record_tactical_decision(self, decision: str, game_state: TOWGameState):
        """Record a tactical decision during battle"""
        if self.current_demo:
            self.current_demo['tactical_decisions'].append(decision)
            self.current_demo['game_states'].append(game_state)
            
    def record_turn_decision(self, phase: str, action: str, context: Dict):
        """Record turn-based decisions"""
        if self.current_demo:
            self.current_demo['turn_decisions'].append({
                'phase': phase,
                'action': action,
                'context': context
            })
    
    def finish_recording(self, outcome: str, effectiveness: float):
        """Finish recording and save demonstration"""
        if self.current_demo:
            demo = ExpertDemo(
                army_composition=self.current_demo['army_composition'],
                tactical_decisions=self.current_demo['tactical_decisions'],
                battle_outcome=outcome,
                effectiveness_score=effectiveness,
                turn_decisions=self.current_demo['turn_decisions']
            )
            self.demonstrations.append(demo)
            self.current_demo = None
            
    def save_demonstrations(self, filename: str):
        """Save expert demonstrations to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.demonstrations, f)
        print(f"💾 Saved {len(self.demonstrations)} expert demonstrations to {filename}")
        
    def load_demonstrations(self, filename: str):
        """Load expert demonstrations from file"""
        try:
            with open(filename, 'rb') as f:
                self.demonstrations = pickle.load(f)
            print(f"📂 Loaded {len(self.demonstrations)} expert demonstrations from {filename}")
            return True
        except FileNotFoundError:
            print(f"⚠️ No expert demonstrations found at {filename}")
            return False

class TOWImitationAgent:
    """Imitation learning agent that learns from expert demonstrations"""
    
    def __init__(self, faction: str):
        self.faction = faction
        self.army_preferences = {}
        self.tactical_weights = {}
        
        # Unit databases
        if faction == "Orc & Goblin Tribes":
            self.available_units = {
                "Orc Big Boss": {"points": 80, "combat": 4, "role": "character"},
                "Orc Boyz": {"points": 12, "combat": 3, "numbers": True},
                "Night Goblins": {"points": 8, "combat": 2, "cheap": True},
                "Orc Arrer Boyz": {"points": 14, "combat": 3, "ranged": True},
                "Wolf Riders": {"points": 18, "combat": 3, "fast": True},
                "Trolls": {"points": 35, "combat": 5, "elite": True}
            }
        else:  # Nuln
            self.available_units = {
                "Engineer": {"points": 65, "combat": 4, "role": "character"},
                "Handgunners": {"points": 12, "combat": 4, "ranged": True, "armor_piercing": True},
                "Crossbowmen": {"points": 10, "combat": 3, "ranged": True},
                "Swordsmen": {"points": 8, "combat": 4, "armor": True},
                "Great Cannon": {"points": 100, "combat": 8, "artillery": True},
                "Outriders": {"points": 24, "combat": 4, "fast": True, "ranged": True}
            }
    
    def learn_from_expert_demos(self, demonstrations: List[ExpertDemo]):
        """Learn from expert demonstrations"""
        print(f"🧠 Learning from expert demonstrations for {self.faction}...")
        
        faction_demos = [d for d in demonstrations if self._is_faction_demo(d)]
        successful_demos = [d for d in faction_demos if d.effectiveness_score > 0.6]
        
        if not successful_demos:
            print(f"⚠️ No successful demonstrations for {self.faction}")
            return
        
        # Learn army composition patterns
        for demo in successful_demos:
            for unit, count in demo.army_composition.items():
                if unit in self.available_units:
                    if unit not in self.army_preferences:
                        self.army_preferences[unit] = []
                    self.army_preferences[unit].append(count)
        
        # Calculate preferred ranges
        for unit in self.army_preferences:
            counts = self.army_preferences[unit]
            self.army_preferences[unit] = {
                'avg': sum(counts) / len(counts),
                'min': min(counts),
                'max': max(counts),
                'weight': len(counts) / len(successful_demos)
            }
        
        print(f"✅ Learned patterns from {len(successful_demos)} successful demonstrations")
    
    def _is_faction_demo(self, demo: ExpertDemo) -> bool:
        """Check if demonstration belongs to this faction"""
        demo_units = set(demo.army_composition.keys())
        faction_units = set(self.available_units.keys())
        overlap = len(demo_units.intersection(faction_units))
        return overlap > len(demo_units) / 2
    
    def generate_expert_biased_army(self) -> Dict[str, int]:
        """Generate army using expert bias"""
        army = {}
        
        # Add character first
        char_units = [u for u, s in self.available_units.items() if s.get('role') == 'character']
        if char_units:
            army[char_units[0]] = 1
        
        # Add other units based on expert preferences
        for unit, stats in self.available_units.items():
            if unit not in army:
                army[unit] = 0
            
            if unit in self.army_preferences:
                prefs = self.army_preferences[unit]
                # Use expert average with some variation
                base_count = int(prefs['avg'])
                variation = random.randint(-1, 1)
                army[unit] = max(0, base_count + variation)
            else:
                # Random count for units without expert data
                army[unit] = random.randint(0, 2)
        
        return {k: v for k, v in army.items() if v > 0}

class BiasRLTOWTrainer:
    """Main bias RL trainer"""
    
    def __init__(self):
        self.orc_imitator = TOWImitationAgent("Orc & Goblin Tribes")
        self.nuln_imitator = TOWImitationAgent("City-State of Nuln")
        self.alpha = 0.5  # Bias strength
        self.exploration_rate = 0.2
    
    def create_expert_demonstrations(self, num_demos: int = 100) -> List[ExpertDemo]:
        """Create expert demonstrations for different tactical scenarios"""
        print(f"📝 Creating {num_demos} expert demonstrations...")
        
        demonstrations = []
        scenarios = ["balanced", "ranged_heavy", "melee_focus", "elite_units", "horde"]
        
        for i in range(num_demos):
            scenario = random.choice(scenarios)
            faction = random.choice(["Orc & Goblin Tribes", "City-State of Nuln"])
            
            # Create expert army for scenario
            army = self._create_expert_army(faction, scenario)
            
            # Generate expert tactical decisions
            tactics = self._get_expert_tactics(scenario)
            
            # Calculate effectiveness based on scenario appropriateness
            effectiveness = self._calculate_effectiveness(army, tactics, scenario)
            
            demo = ExpertDemo(
                army_composition=army,
                tactical_decisions=tactics,
                battle_outcome="Victory" if effectiveness > 0.6 else "Defeat",
                effectiveness_score=effectiveness
            )
            
            demonstrations.append(demo)
        
        # Save demonstrations
        with open('expert_demos.pkl', 'wb') as f:
            pickle.dump(demonstrations, f)
        
        print(f"💾 Saved {len(demonstrations)} expert demonstrations")
        return demonstrations
    
    def _create_expert_army(self, faction: str, scenario: str) -> Dict[str, int]:
        """Create expert army composition for scenario"""
        if faction == "Orc & Goblin Tribes":
            base = {"Orc Big Boss": 1}
            
            if scenario == "ranged_heavy":
                base.update({"Orc Arrer Boyz": 4, "Night Goblins": 2, "Orc Boyz": 2})
            elif scenario == "melee_focus":
                base.update({"Orc Boyz": 5, "Trolls": 2, "Wolf Riders": 1})
            elif scenario == "elite_units":
                base.update({"Trolls": 3, "Wolf Riders": 3, "Orc Boyz": 1})
            elif scenario == "horde":
                base.update({"Night Goblins": 6, "Orc Boyz": 4, "Orc Arrer Boyz": 1})
            else:  # balanced
                base.update({"Orc Boyz": 3, "Night Goblins": 2, "Orc Arrer Boyz": 2, "Wolf Riders": 1})
        
        else:  # Nuln
            base = {"Engineer": 1}
            
            if scenario == "ranged_heavy":
                base.update({"Handgunners": 4, "Crossbowmen": 3, "Great Cannon": 1})
            elif scenario == "elite_units":
                base.update({"Outriders": 3, "Handgunners": 2, "Great Cannon": 1})
            else:  # balanced/other
                base.update({"Swordsmen": 3, "Handgunners": 2, "Crossbowmen": 1, "Outriders": 1})
        
        return base
    
    def _get_expert_tactics(self, scenario: str) -> List[str]:
        """Get expert tactics for scenario"""
        base_tactics = ["maintain_formation", "focus_fire", "protect_flanks"]
        
        scenario_tactics = {
            "ranged_heavy": ["establish_firing_lines", "maximize_range"],
            "melee_focus": ["aggressive_advance", "coordinate_charges"],
            "elite_units": ["preserve_elites", "tactical_strikes"],
            "horde": ["swarm_tactics", "overwhelm_enemy"]
        }
        
        return base_tactics + scenario_tactics.get(scenario, ["balanced_approach"])
    
    def _calculate_effectiveness(self, army: Dict, tactics: List[str], scenario: str) -> float:
        """Calculate effectiveness of army/tactics for scenario"""
        base_effectiveness = 0.5
        
        # Army composition bonuses
        total_units = sum(army.values())
        
        if scenario == "ranged_heavy":
            ranged_units = army.get("Orc Arrer Boyz", 0) + army.get("Handgunners", 0) + army.get("Crossbowmen", 0)
            if ranged_units / total_units > 0.4:
                base_effectiveness += 0.3
        
        elif scenario == "melee_focus":
            melee_units = army.get("Orc Boyz", 0) + army.get("Trolls", 0) + army.get("Swordsmen", 0)
            if melee_units / total_units > 0.5:
                base_effectiveness += 0.3
        
        # Tactical bonuses
        if "establish_firing_lines" in tactics and scenario == "ranged_heavy":
            base_effectiveness += 0.2
        if "aggressive_advance" in tactics and scenario == "melee_focus":
            base_effectiveness += 0.2
        
        return min(1.0, base_effectiveness + random.uniform(-0.1, 0.1))
    
    def train_with_bias_rl(self, num_battles: int = 20000):
        """Train using bias RL methodology"""
        print("🧠 STARTING BIAS RL TRAINING")
        print("=" * 40)
        
        # Step 1: Create or load expert demonstrations
        try:
            with open('expert_demos.pkl', 'rb') as f:
                demonstrations = pickle.load(f)
            print(f"📂 Loaded {len(demonstrations)} expert demonstrations")
        except FileNotFoundError:
            demonstrations = self.create_expert_demonstrations(150)
        
        # Step 2: Train imitation agents
        self.orc_imitator.learn_from_expert_demos(demonstrations)
        self.nuln_imitator.learn_from_expert_demos(demonstrations)
        
        # Step 3: Bias RL training loop
        print(f"🎯 Starting bias RL training for {num_battles} battles...")
        
        results = []
        start_time = time.time()
        
        for battle in range(num_battles):
            # Generate armies using bias
            if random.random() > self.exploration_rate:
                # Use expert bias
                orc_army = self.orc_imitator.generate_expert_biased_army()
                nuln_army = self.nuln_imitator.generate_expert_biased_army()
            else:
                # Random exploration
                orc_army = self._generate_random_army("Orc & Goblin Tribes")
                nuln_army = self._generate_random_army("City-State of Nuln")
            
            # Simulate battle with reward shaping
            result = self._simulate_battle_with_bias(orc_army, nuln_army)
            results.append(result)
            
            # Progress updates
            if (battle + 1) % 5000 == 0:
                elapsed = time.time() - start_time
                rate = (battle + 1) / elapsed
                
                recent = results[-1000:] if len(results) >= 1000 else results
                orc_wins = sum(1 for r in recent if r['winner'] == 'Orcs')
                balance = orc_wins / len(recent)
                
                print(f"📊 Battle {battle + 1}/{num_battles}")
                print(f"   ⚡ Rate: {rate:.1f} battles/sec")
                print(f"   ⚖️ Balance: {balance:.1%} Orc | {1-balance:.1%} Nuln")
                print(f"   🧬 Exploration: {self.exploration_rate:.2f}")
                
                # Adapt exploration rate
                if balance < 0.4 or balance > 0.6:
                    self.exploration_rate = min(0.4, self.exploration_rate + 0.05)
                else:
                    self.exploration_rate = max(0.05, self.exploration_rate * 0.98)
        
        self._report_training_results(results, time.time() - start_time)
        return results
    
    def _generate_random_army(self, faction: str) -> Dict[str, int]:
        """Generate random army for exploration"""
        if faction == "Orc & Goblin Tribes":
            return {
                "Orc Big Boss": 1,
                "Orc Boyz": random.randint(1, 4),
                "Night Goblins": random.randint(0, 3),
                "Orc Arrer Boyz": random.randint(0, 3),
                "Wolf Riders": random.randint(0, 2),
                "Trolls": random.randint(0, 2)
            }
        else:
            return {
                "Engineer": 1,
                "Handgunners": random.randint(1, 3),
                "Crossbowmen": random.randint(0, 2),
                "Swordsmen": random.randint(1, 3),
                "Great Cannon": random.randint(0, 1),
                "Outriders": random.randint(0, 2)
            }
    
    def _simulate_battle_with_bias(self, orc_army: Dict, nuln_army: Dict) -> Dict:
        """Simulate battle with expert bias and reward shaping"""
        # Calculate base strength
        orc_strength = self._calculate_army_strength(orc_army, self.orc_imitator.available_units)
        nuln_strength = self._calculate_army_strength(nuln_army, self.nuln_imitator.available_units)
        
        # Apply expert bias (reward shaping)
        orc_bias = self._calculate_expert_bias(orc_army, self.orc_imitator)
        nuln_bias = self._calculate_expert_bias(nuln_army, self.nuln_imitator)
        
        # Final strengths with bias
        final_orc = orc_strength + (orc_bias * self.alpha)
        final_nuln = nuln_strength + (nuln_bias * self.alpha)
        
        # Battle resolution with randomness
        random_factor = random.uniform(0.85, 1.15)
        winner = "Orcs" if final_orc * random_factor > final_nuln else "Nuln"
        
        return {
            'winner': winner,
            'orc_strength': orc_strength,
            'nuln_strength': nuln_strength,
            'orc_bias': orc_bias,
            'nuln_bias': nuln_bias,
            'final_orc': final_orc,
            'final_nuln': final_nuln
        }
    
    def _calculate_army_strength(self, army: Dict, unit_data: Dict) -> float:
        """Calculate raw army combat strength"""
        strength = 0
        for unit, count in army.items():
            if unit in unit_data:
                stats = unit_data[unit]
                unit_strength = stats['combat'] * count
                
                # Apply modifiers
                if stats.get('elite'):
                    unit_strength *= 1.5
                elif stats.get('numbers'):
                    unit_strength *= 1.2
                elif stats.get('artillery'):
                    unit_strength *= 2.0
                elif stats.get('armor_piercing'):
                    unit_strength *= 1.4
                elif stats.get('armor'):
                    unit_strength *= 1.3
                elif stats.get('ranged'):
                    unit_strength *= 1.2
                
                strength += unit_strength
        
        return strength
    
    def _calculate_expert_bias(self, army: Dict, imitator: TOWImitationAgent) -> float:
        """Calculate expert bias for reward shaping"""
        bias = 0
        
        for unit, count in army.items():
            if unit in imitator.army_preferences:
                prefs = imitator.army_preferences[unit]
                optimal = prefs['avg']
                
                # Reward armies close to expert preferences
                deviation = abs(count - optimal)
                if deviation <= 1:
                    bias += prefs['weight'] * 15
                elif deviation <= 2:
                    bias += prefs['weight'] * 8
        
        return bias
    
    def _report_training_results(self, results: List[Dict], training_time: float):
        """Generate training report"""
        total_battles = len(results)
        orc_wins = sum(1 for r in results if r['winner'] == 'Orcs')
        
        print("\n🎉 BIAS RL TRAINING COMPLETE!")
        print("=" * 40)
        print(f"⏱️ Time: {training_time:.1f} seconds")
        print(f"⚡ Rate: {total_battles/training_time:.1f} battles/sec")
        print(f"⚖️ Final balance: {orc_wins/total_battles:.1%} Orc | {1-orc_wins/total_battles:.1%} Nuln")
        print(f"🧠 Bias strength (alpha): {self.alpha}")
        print(f"🎯 Final exploration rate: {self.exploration_rate:.2f}")
        
        # Show learned army preferences
        print("\n🏆 LEARNED EXPERT PREFERENCES:")
        print("-" * 30)
        print("🧌 Orc preferences:")
        for unit, prefs in self.orc_imitator.army_preferences.items():
            print(f"  {unit}: {prefs['avg']:.1f} (weight: {prefs['weight']:.2f})")
        
        print("\n🏰 Nuln preferences:")
        for unit, prefs in self.nuln_imitator.army_preferences.items():
            print(f"  {unit}: {prefs['avg']:.1f} (weight: {prefs['weight']:.2f})")
        
        # Save results
        with open('bias_rl_results.json', 'w') as f:
            json.dump({
                'total_battles': total_battles,
                'orc_win_rate': orc_wins / total_battles,
                'training_time': training_time,
                'final_alpha': self.alpha,
                'final_exploration': self.exploration_rate
            }, f, indent=2)
        
        print("\n💾 Results saved to bias_rl_results.json")

def main():
    """Main training function"""
    print("🏛️ BIAS RL WARHAMMER: THE OLD WORLD AI TRAINER")
    print("=" * 50)
    print("🧠 Using expert demonstrations + imitation learning")
    print("🎯 Bias RL approach from MLPrague conference")
    print()
    
    trainer = BiasRLTOWTrainer()
    
    print("🎮 TRAINING OPTIONS:")
    print("1. Quick bias RL (10,000 battles)")
    print("2. Standard bias RL (30,000 battles)")
    print("3. Intensive bias RL (100,000 battles)")
    print("4. Tune bias strength")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        trainer.train_with_bias_rl(10000)
    elif choice == '2':
        trainer.train_with_bias_rl(30000)
    elif choice == '3':
        trainer.train_with_bias_rl(100000)
    elif choice == '4':
        alpha = float(input("Enter bias strength (0.1-1.0): "))
        trainer.alpha = alpha
        trainer.train_with_bias_rl(20000)
    else:
        print("Running standard training...")
        trainer.train_with_bias_rl(30000)

if __name__ == "__main__":
    main() 