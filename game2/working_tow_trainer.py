#!/usr/bin/env python3
"""
🏛️ WORKING WARHAMMER: THE OLD WORLD AI TRAINER
===============================================

A working AI trainer that learns optimal army compositions through
meaningful tactical battle simulation.
"""

import random
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class BattleResult:
    """Results from a training battle"""
    winner: str
    orc_casualties: int
    nuln_casualties: int
    battle_length: int
    tactics_score: float

class TOWAgent:
    """AI agent that learns army compositions"""
    
    def __init__(self, faction: str):
        self.faction = faction
        self.wins = 0
        self.battles = 0
        self.army_preferences = {}
        self.best_armies = []
        
        # Initialize army units based on faction - BALANCED stats
        if faction == "Orc & Goblin Tribes":
            self.available_units = {
                "Orc Big Boss": {"points": 80, "combat": 4, "leadership": 7},
                "Orc Boyz": {"points": 12, "combat": 3, "numbers": True},
                "Night Goblins": {"points": 8, "combat": 2, "cheap": True},
                "Orc Arrer Boyz": {"points": 14, "combat": 3, "ranged": True},
                "Wolf Riders": {"points": 18, "combat": 3, "fast": True},
                "Trolls": {"points": 35, "combat": 5, "elite": True}
            }
        else:  # Nuln - BUFFED to be competitive
            self.available_units = {
                "Engineer": {"points": 65, "combat": 4, "leadership": 9},
                "Handgunners": {"points": 12, "combat": 4, "ranged": True, "armor_piercing": True},
                "Crossbowmen": {"points": 10, "combat": 3, "ranged": True},
                "Swordsmen": {"points": 8, "combat": 4, "reliable": True, "armor": True},
                "Great Cannon": {"points": 100, "combat": 8, "artillery": True, "devastating": True},
                "Outriders": {"points": 24, "combat": 4, "fast": True, "ranged": True}
            }
    
    def generate_army(self, points_limit: int = 2000) -> Dict[str, int]:
        """Generate an army composition"""
        army = {}
        remaining_points = points_limit
        
        # Start with required characters
        if self.faction == "Orc & Goblin Tribes":
            army["Orc Big Boss"] = 1
            remaining_points -= self.available_units["Orc Big Boss"]["points"]
        else:
            army["Engineer"] = 1
            remaining_points -= self.available_units["Engineer"]["points"]
        
        # Add units based on preferences and available points
        for unit, stats in self.available_units.items():
            if unit not in army:
                army[unit] = 0
            
            # Determine how many of this unit to include
            unit_preference = self.army_preferences.get(unit, 0.5)
            base_count = max(0, int(unit_preference * 5))
            
            # Add randomness for exploration
            if random.random() < 0.3:
                base_count += random.randint(-1, 2)
            
            # Ensure we can afford it
            unit_cost = stats["points"]
            max_affordable = remaining_points // unit_cost
            final_count = min(base_count, max_affordable)
            
            army[unit] += final_count
            remaining_points -= final_count * unit_cost
        
        return {k: v for k, v in army.items() if v > 0}
    
    def learn_from_battle(self, army_used: Dict, result: BattleResult):
        """Learn from battle outcome"""
        self.battles += 1
        won = (result.winner == self.faction)
        
        if won:
            self.wins += 1
            # Reinforce successful army choices
            for unit, count in army_used.items():
                current_pref = self.army_preferences.get(unit, 0.5)
                self.army_preferences[unit] = min(1.0, current_pref + 0.1)
            
            # Store successful armies
            army_score = result.tactics_score
            self.best_armies.append((army_used.copy(), army_score))
            self.best_armies.sort(key=lambda x: x[1], reverse=True)
            self.best_armies = self.best_armies[:5]  # Keep top 5
        else:
            # Reduce preference for unsuccessful choices
            for unit, count in army_used.items():
                current_pref = self.army_preferences.get(unit, 0.5)
                self.army_preferences[unit] = max(0.0, current_pref - 0.05)
    
    def get_win_rate(self) -> float:
        return self.wins / max(1, self.battles)

class WorkingTOWTrainer:
    """Main training system"""
    
    def __init__(self):
        self.orc_agent = TOWAgent("Orc & Goblin Tribes")
        self.nuln_agent = TOWAgent("City-State of Nuln")
    
    def simulate_tactical_battle(self, orc_army: Dict, nuln_army: Dict) -> BattleResult:
        """Simulate a tactical battle with meaningful mechanics"""
        # Calculate army strengths
        orc_combat = self._calculate_army_strength(orc_army, self.orc_agent.available_units)
        nuln_combat = self._calculate_army_strength(nuln_army, self.nuln_agent.available_units)
        
        # Simulate battle phases
        battle_length = 0
        orc_strength = orc_combat
        nuln_strength = nuln_combat
        
        # Pre-battle (ranged phase)
        orc_ranged = self._get_ranged_power(orc_army, self.orc_agent.available_units)
        nuln_ranged = self._get_ranged_power(nuln_army, self.nuln_agent.available_units)
        
        orc_casualties = max(0, nuln_ranged - orc_strength * 0.1)
        nuln_casualties = max(0, orc_ranged - nuln_strength * 0.1)
        
        orc_strength -= orc_casualties * 0.2
        nuln_strength -= nuln_casualties * 0.2
        battle_length += 1
        
        # Melee combat phases
        while battle_length < 6 and orc_strength > 0 and nuln_strength > 0:
            # Combat resolution
            orc_damage = orc_strength * random.uniform(0.8, 1.2)
            nuln_damage = nuln_strength * random.uniform(0.8, 1.2)
            
            round_orc_casualties = nuln_damage * 0.15
            round_nuln_casualties = orc_damage * 0.15
            
            orc_casualties += round_orc_casualties
            nuln_casualties += round_nuln_casualties
            
            orc_strength = max(0, orc_strength - round_orc_casualties)
            nuln_strength = max(0, nuln_strength - round_nuln_casualties)
            
            battle_length += 1
            
            # Leadership test
            if random.random() < 0.2:  # 20% chance of morale check
                if orc_strength < orc_combat * 0.5:
                    orc_strength *= 0.8  # Morale penalty
                if nuln_strength < nuln_combat * 0.5:
                    nuln_strength *= 0.8
        
        # Determine winner
        if orc_strength > nuln_strength:
            winner = "Orc & Goblin Tribes"
        elif nuln_strength > orc_strength:
            winner = "City-State of Nuln"
        else:
            winner = "Draw"
        
        # Calculate tactics score based on efficiency
        total_starting = orc_combat + nuln_combat
        total_remaining = orc_strength + nuln_strength
        tactics_score = total_remaining / total_starting
        
        return BattleResult(
            winner=winner,
            orc_casualties=int(orc_casualties),
            nuln_casualties=int(nuln_casualties),
            battle_length=battle_length,
            tactics_score=tactics_score
        )
    
    def _calculate_army_strength(self, army: Dict, unit_stats: Dict) -> float:
        """Calculate total army combat strength"""
        strength = 0
        for unit, count in army.items():
            if unit in unit_stats:
                base_combat = unit_stats[unit]["combat"]
                unit_strength = base_combat * count
                
                # Apply unit type modifiers
                if unit_stats[unit].get("elite"):
                    unit_strength *= 1.5
                elif unit_stats[unit].get("numbers"):
                    unit_strength *= 1.2
                elif unit_stats[unit].get("armor"):
                    unit_strength *= 1.3  # Defensive bonus
                elif unit_stats[unit].get("devastating"):
                    unit_strength *= 2.0  # Artillery power
                elif unit_stats[unit].get("armor_piercing"):
                    unit_strength *= 1.4  # Effective vs armor
                
                strength += unit_strength
        return strength
    
    def _get_ranged_power(self, army: Dict, unit_stats: Dict) -> float:
        """Calculate ranged combat power"""
        ranged_power = 0
        for unit, count in army.items():
            if unit in unit_stats and unit_stats[unit].get("ranged"):
                base_combat = unit_stats[unit]["combat"]
                unit_ranged = base_combat * count * 0.8  # Improved ranged modifier
                
                # Special ranged bonuses
                if unit_stats[unit].get("artillery"):
                    unit_ranged *= 2.5  # Artillery is devastating at range
                elif unit_stats[unit].get("armor_piercing"):
                    unit_ranged *= 1.5  # Gunpowder weapons
                
                ranged_power += unit_ranged
        return ranged_power
    
    def run_training(self, num_battles: int = 10000):
        """Run the training session"""
        print("🏛️ WORKING TOW AI TRAINING")
        print("=" * 40)
        print(f"🎯 Training for {num_battles} battles")
        print("⚔️ Using tactical battle simulation")
        print("🧠 Agents will learn optimal army compositions")
        print()
        
        start_time = time.time()
        
        for battle in range(num_battles):
            # Generate armies
            orc_army = self.orc_agent.generate_army()
            nuln_army = self.nuln_agent.generate_army()
            
            # Simulate battle
            result = self.simulate_tactical_battle(orc_army, nuln_army)
            
            # Agents learn
            self.orc_agent.learn_from_battle(orc_army, result)
            self.nuln_agent.learn_from_battle(nuln_army, result)
            
            # Progress report
            if (battle + 1) % 2000 == 0:
                elapsed = time.time() - start_time
                rate = (battle + 1) / elapsed
                
                print(f"📊 Battle {battle + 1}/{num_battles}")
                print(f"   ⚡ Rate: {rate:.1f} battles/sec") 
                print(f"   🧌 Orc Win Rate: {self.orc_agent.get_win_rate():.1%}")
                print(f"   🏰 Nuln Win Rate: {self.nuln_agent.get_win_rate():.1%}")
                print()
        
        training_time = time.time() - start_time
        
        # Final report
        print("🎉 TRAINING COMPLETE!")
        print("=" * 30)
        print(f"⏱️ Training time: {training_time:.1f} seconds")
        print(f"⚡ Rate: {num_battles/training_time:.1f} battles/sec")
        print(f"🧌 Final Orc Win Rate: {self.orc_agent.get_win_rate():.1%}")
        print(f"🏰 Final Nuln Win Rate: {self.nuln_agent.get_win_rate():.1%}")
        print()
        
        # Show best armies
        print("🏆 BEST ARMY COMPOSITIONS DISCOVERED:")
        print("=" * 45)
        
        print("🧌 Best Orc Armies:")
        for i, (army, score) in enumerate(self.orc_agent.best_armies[:3]):
            print(f"  {i+1}. Score {score:.3f}: {army}")
        
        print("\n🏰 Best Nuln Armies:")
        for i, (army, score) in enumerate(self.nuln_agent.best_armies[:3]):
            print(f"  {i+1}. Score {score:.3f}: {army}")
        
        print()
        
        # Save results
        self.save_training_results()
        
        return {
            'orc_win_rate': self.orc_agent.get_win_rate(),
            'nuln_win_rate': self.nuln_agent.get_win_rate(),
            'training_time': training_time,
            'battles_per_second': num_battles / training_time
        }
    
    def save_training_results(self):
        """Save training results to file"""
        results = {
            'orc_preferences': self.orc_agent.army_preferences,
            'nuln_preferences': self.nuln_agent.army_preferences,
            'orc_best_armies': self.orc_agent.best_armies,
            'nuln_best_armies': self.nuln_agent.best_armies,
            'orc_win_rate': self.orc_agent.get_win_rate(),
            'nuln_win_rate': self.nuln_agent.get_win_rate()
        }
        
        with open('tow_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("💾 Training results saved to tow_training_results.json")

def main():
    """Main function"""
    trainer = WorkingTOWTrainer()
    
    print("🎮 TOW AI TRAINING MENU")
    print("=" * 25)
    print("1. Quick Training (5,000 battles)")
    print("2. Standard Training (25,000 battles)")
    print("3. Intensive Training (100,000 battles)")
    print("4. Custom Training")
    
    choice = input("\nSelect training mode (1-4): ").strip()
    
    if choice == '1':
        trainer.run_training(5000)
    elif choice == '2':
        trainer.run_training(25000)
    elif choice == '3':
        trainer.run_training(100000)
    elif choice == '4':
        battles = int(input("Number of battles: "))
        trainer.run_training(battles)
    else:
        print("Invalid choice, running quick training...")
        trainer.run_training(5000)

if __name__ == "__main__":
    main() 