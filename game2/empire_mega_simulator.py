#!/usr/bin/env python3
"""
🏛️ EMPIRE MEGA SIMULATOR - 300,000 BATTLES
Test Empire armies against all enemy types to find the ultimate 750pt list
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time
from empire_750pt_optimizer import Empire750Optimizer, Unit, ArmyComposition

@dataclass
class EnemyArmy:
    name: str
    faction: str
    units: List[Dict]
    total_points: int
    strengths: List[str]
    weaknesses: List[str]
    difficulty: float  # 1-10 rating

@dataclass
class BattleResult:
    empire_army: str
    enemy_army: str
    empire_victory: bool
    victory_margin: float
    battle_duration: int
    empire_casualties: float
    enemy_casualties: float

class EmpireMegaSimulator:
    """300,000 battle simulation system"""
    
    def __init__(self):
        self.optimizer = Empire750Optimizer()
        self.enemy_armies = self._create_enemy_database()
        self.battle_count = 0
        
    def _create_enemy_database(self) -> List[EnemyArmy]:
        """Create comprehensive enemy army database"""
        enemies = []
        
        # ORCS & GOBLINS - Horde armies
        enemies.append(EnemyArmy(
            name="Orc Horde",
            faction="Orcs & Goblins",
            units=[
                {"name": "Orc Warboss", "count": 1, "points": 120},
                {"name": "Orc Boys", "count": 30, "points": 180},
                {"name": "Orc Boys", "count": 25, "points": 150},
                {"name": "Goblin Archers", "count": 20, "points": 100},
                {"name": "Orc Boar Boyz", "count": 5, "points": 110},
                {"name": "Rock Lobber", "count": 1, "points": 80}
            ],
            total_points=740,
            strengths=["Numbers", "Melee Combat", "Cheap Units"],
            weaknesses=["Low Leadership", "Poor Shooting", "Animosity"],
            difficulty=6.5
        ))
        
        enemies.append(EnemyArmy(
            name="Goblin Swarm",
            faction="Orcs & Goblins",
            units=[
                {"name": "Goblin Big Boss", "count": 1, "points": 80},
                {"name": "Night Goblins", "count": 40, "points": 200},
                {"name": "Goblin Archers", "count": 20, "points": 100},
                {"name": "Goblin Wolf Riders", "count": 10, "points": 140},
                {"name": "Goblin Fanatics", "count": 6, "points": 90},
                {"name": "Doom Diver Catapult", "count": 1, "points": 80},
                {"name": "Snotling Pump Wagon", "count": 1, "points": 50}
            ],
            total_points=740,
            strengths=["Overwhelming Numbers", "Cheap", "Unpredictable"],
            weaknesses=["Terrible Leadership", "Weak Individual Units"],
            difficulty=5.0
        ))
        
        # DWARFS - Elite defensive armies
        enemies.append(EnemyArmy(
            name="Dwarf Gunline",
            faction="Dwarfs",
            units=[
                {"name": "Dwarf Lord", "count": 1, "points": 140},
                {"name": "Dwarf Warriors", "count": 20, "points": 160},
                {"name": "Dwarf Quarrellers", "count": 15, "points": 165},
                {"name": "Dwarf Thunderers", "count": 10, "points": 130},
                {"name": "Dwarf Cannon", "count": 1, "points": 120},
                {"name": "Organ Gun", "count": 1, "points": 120}
            ],
            total_points=735,
            strengths=["Heavy Armor", "Superior Shooting", "High Leadership"],
            weaknesses=["Slow Movement", "Expensive", "Low Model Count"],
            difficulty=8.5
        ))
        
        enemies.append(EnemyArmy(
            name="Dwarf Ironbreakers",
            faction="Dwarfs",
            units=[
                {"name": "Dwarf Thane", "count": 1, "points": 100},
                {"name": "Ironbreakers", "count": 15, "points": 225},
                {"name": "Longbeards", "count": 15, "points": 180},
                {"name": "Dwarf Rangers", "count": 10, "points": 120},
                {"name": "Grudge Thrower", "count": 1, "points": 80},
                {"name": "Flame Cannon", "count": 1, "points": 70}
            ],
            total_points=775,
            strengths=["Elite Infantry", "Unbreakable Units", "Anti-Armor"],
            weaknesses=["Very Expensive", "Low Numbers", "Slow"],
            difficulty=9.0
        ))
        
        # HIGH ELVES - Elite magic armies
        enemies.append(EnemyArmy(
            name="High Elf Spearhost",
            faction="High Elves",
            units=[
                {"name": "Elf Prince", "count": 1, "points": 120},
                {"name": "Elf Mage", "count": 1, "points": 100},
                {"name": "Elf Spearmen", "count": 20, "points": 200},
                {"name": "Elf Archers", "count": 15, "points": 165},
                {"name": "Silver Helms", "count": 5, "points": 110},
                {"name": "Repeater Bolt Thrower", "count": 2, "points": 140}
            ],
            total_points=735,
            strengths=["High Initiative", "Magic", "Excellent Shooting"],
            weaknesses=["Low Toughness", "Expensive", "Fragile"],
            difficulty=8.0
        ))
        
        enemies.append(EnemyArmy(
            name="High Elf Magic Heavy",
            faction="High Elves",
            units=[
                {"name": "Archmage", "count": 1, "points": 185},
                {"name": "Elf Mage", "count": 1, "points": 100},
                {"name": "Elf Spearmen", "count": 16, "points": 160},
                {"name": "Elf Archers", "count": 10, "points": 110},
                {"name": "Phoenix Guard", "count": 10, "points": 150},
                {"name": "Eagle Claw Bolt Thrower", "count": 1, "points": 70}
            ],
            total_points=775,
            strengths=["Devastating Magic", "Ward Saves", "High Leadership"],
            weaknesses=["Magic Dependent", "Low Model Count", "Expensive"],
            difficulty=9.5
        ))
        
        # DARK ELVES - Fast aggressive armies
        enemies.append(EnemyArmy(
            name="Dark Elf Raiders",
            faction="Dark Elves",
            units=[
                {"name": "Dreadlord", "count": 1, "points": 130},
                {"name": "Dark Elf Spearmen", "count": 20, "points": 180},
                {"name": "Dark Elf Crossbows", "count": 15, "points": 150},
                {"name": "Dark Riders", "count": 5, "points": 110},
                {"name": "Cold One Knights", "count": 5, "points": 125},
                {"name": "Reaper Bolt Thrower", "count": 1, "points": 70}
            ],
            total_points=765,
            strengths=["Fast Movement", "Good Shooting", "Fear Causing"],
            weaknesses=["Hatred Rules", "Fragile", "Leadership Issues"],
            difficulty=7.5
        ))
        
        # CHAOS - Elite heavy armies
        enemies.append(EnemyArmy(
            name="Chaos Warriors",
            faction="Warriors of Chaos",
            units=[
                {"name": "Chaos Lord", "count": 1, "points": 150},
                {"name": "Chaos Warriors", "count": 12, "points": 216},
                {"name": "Chaos Warriors", "count": 10, "points": 180},
                {"name": "Chaos Knights", "count": 4, "points": 160},
                {"name": "Chaos Hounds", "count": 10, "points": 60}
            ],
            total_points=766,
            strengths=["Heavy Armor", "High Toughness", "Fear"],
            weaknesses=["Very Expensive", "Low Numbers", "Slow"],
            difficulty=9.5
        ))
        
        # SKAVEN - Swarm with weapons teams
        enemies.append(EnemyArmy(
            name="Skaven Horde",
            faction="Skaven",
            units=[
                {"name": "Skaven Warlord", "count": 1, "points": 90},
                {"name": "Clanrats", "count": 30, "points": 180},
                {"name": "Clanrats", "count": 25, "points": 150},
                {"name": "Skaven Slaves", "count": 20, "points": 40},
                {"name": "Jezzails", "count": 5, "points": 100},
                {"name": "Ratling Gun", "count": 1, "points": 60},
                {"name": "Warp Lightning Cannon", "count": 1, "points": 90}
            ],
            total_points=710,
            strengths=["Numbers", "Weapon Teams", "Cheap Units"],
            weaknesses=["Poor Leadership", "Unreliable", "Fragile"],
            difficulty=6.0
        ))
        
        # VAMPIRE COUNTS - Undead hordes
        enemies.append(EnemyArmy(
            name="Vampire Horde",
            faction="Vampire Counts",
            units=[
                {"name": "Vampire Lord", "count": 1, "points": 175},
                {"name": "Skeletons", "count": 25, "points": 125},
                {"name": "Zombies", "count": 30, "points": 90},
                {"name": "Ghouls", "count": 15, "points": 120},
                {"name": "Dire Wolves", "count": 10, "points": 80},
                {"name": "Black Knights", "count": 5, "points": 115}
            ],
            total_points=705,
            strengths=["Undead", "Magic", "Fear", "Regeneration"],
            weaknesses=["Crumble", "Magic Dependent", "No Shooting"],
            difficulty=8.0
        ))
        
        # BRETONNIANS - Heavy cavalry
        enemies.append(EnemyArmy(
            name="Bretonnian Lance",
            faction="Bretonnia",
            units=[
                {"name": "Bretonnian Lord", "count": 1, "points": 140},
                {"name": "Knights of the Realm", "count": 8, "points": 232},
                {"name": "Knights Errant", "count": 6, "points": 138},
                {"name": "Men-at-Arms", "count": 20, "points": 100},
                {"name": "Peasant Bowmen", "count": 15, "points": 90},
                {"name": "Trebuchet", "count": 1, "points": 90}
            ],
            total_points=790,
            strengths=["Heavy Cavalry", "Devastating Charge", "High Leadership"],
            weaknesses=["Expensive", "Peasant Troops", "Limited Flexibility"],
            difficulty=8.5
        ))
        
        return enemies
    
    def simulate_battle(self, empire_army: ArmyComposition, enemy_army: EnemyArmy) -> BattleResult:
        """Simulate a single battle between Empire and enemy army"""
        
        # Calculate army strengths
        empire_strength = self._calculate_empire_strength(empire_army)
        enemy_strength = self._calculate_enemy_strength(enemy_army)
        
        # Apply tactical modifiers based on matchup
        empire_modifier = self._get_empire_tactical_modifier(empire_army, enemy_army)
        enemy_modifier = self._get_enemy_tactical_modifier(enemy_army, empire_army)
        
        # Add randomness (dice rolls, luck, etc.)
        empire_dice = random.uniform(0.7, 1.3)
        enemy_dice = random.uniform(0.7, 1.3)
        
        # Calculate final battle strength
        empire_final = empire_strength * empire_modifier * empire_dice
        enemy_final = enemy_strength * enemy_modifier * enemy_dice
        
        # Determine winner and margin
        empire_victory = empire_final > enemy_final
        if empire_victory:
            victory_margin = (empire_final - enemy_final) / enemy_final
        else:
            victory_margin = (enemy_final - empire_final) / empire_final
        
        # Calculate casualties (more detailed simulation)
        empire_casualties = self._calculate_casualties(empire_army, enemy_army, empire_victory)
        enemy_casualties = self._calculate_casualties_enemy(enemy_army, empire_army, not empire_victory)
        
        # Battle duration (turns)
        battle_duration = random.randint(4, 8)
        
        return BattleResult(
            empire_army=f"Empire Army {id(empire_army)}",
            enemy_army=enemy_army.name,
            empire_victory=empire_victory,
            victory_margin=victory_margin,
            battle_duration=battle_duration,
            empire_casualties=empire_casualties,
            enemy_casualties=enemy_casualties
        )
    
    def _calculate_empire_strength(self, army: ArmyComposition) -> float:
        """Calculate Empire army battle strength"""
        total_strength = 0.0
        
        for unit_name, size, upgrades in army.units:
            unit = self.optimizer.empire_units[unit_name]
            
            # Base unit strength
            unit_strength = unit.effectiveness * size
            
            # Upgrade bonuses
            for upgrade in upgrades:
                if upgrade in unit.upgrades:
                    unit_strength *= 1.1  # 10% bonus per upgrade
            
            # Role-based bonuses
            if unit.battlefield_role == 'Artillery':
                unit_strength *= 1.2  # Artillery is very effective
            elif unit.battlefield_role == 'Heavy Cavalry':
                unit_strength *= 1.15
            elif unit.battlefield_role == 'Elite Infantry':
                unit_strength *= 1.1
            
            total_strength += unit_strength
        
        # Synergy bonus
        total_strength += army.synergy_bonus * 10
        
        return total_strength
    
    def _calculate_enemy_strength(self, enemy: EnemyArmy) -> float:
        """Calculate enemy army battle strength"""
        base_strength = enemy.difficulty * 50  # Base strength from difficulty
        
        # Add unit-based strength
        for unit in enemy.units:
            unit_strength = unit["points"] * 0.1  # Points to strength conversion
            base_strength += unit_strength
        
        return base_strength
    
    def _get_empire_tactical_modifier(self, empire_army: ArmyComposition, enemy: EnemyArmy) -> float:
        """Get Empire tactical advantage/disadvantage vs specific enemy"""
        modifier = 1.0
        
        empire_roles = [self.optimizer.empire_units[unit[0]].battlefield_role for unit in empire_army.units]
        
        # Artillery advantage vs hordes
        if "Artillery" in empire_roles:
            if "Numbers" in enemy.strengths or "Overwhelming Numbers" in enemy.strengths:
                modifier += 0.3  # Artillery destroys hordes
            if "Heavy Armor" in enemy.strengths:
                modifier += 0.2  # Artillery penetrates armor
        
        # Ranged advantage vs low armor
        ranged_count = sum(1 for role in empire_roles if role == "Ranged")
        if ranged_count >= 2:
            if "Low Toughness" in enemy.weaknesses or "Fragile" in enemy.weaknesses:
                modifier += 0.25
        
        # Heavy cavalry vs shooting armies
        if "Heavy Cavalry" in empire_roles:
            if "Superior Shooting" in enemy.strengths:
                modifier -= 0.2  # Cavalry vulnerable to shooting
            if "Low Numbers" in enemy.weaknesses:
                modifier += 0.15  # Cavalry good vs elite armies
        
        # Magic defense
        magic_count = sum(1 for unit in empire_army.units if "Wizard" in unit[0])
        if magic_count > 0:
            if "Magic" in enemy.strengths or "Devastating Magic" in enemy.strengths:
                modifier += 0.2  # Magic defense
        
        return max(0.5, min(2.0, modifier))  # Cap between 0.5x and 2.0x
    
    def _get_enemy_tactical_modifier(self, enemy: EnemyArmy, empire_army: ArmyComposition) -> float:
        """Get enemy tactical advantage vs Empire"""
        modifier = 1.0
        
        empire_roles = [self.optimizer.empire_units[unit[0]].battlefield_role for unit in empire_army.units]
        
        # Enemy advantages
        if "Numbers" in enemy.strengths:
            infantry_count = sum(1 for role in empire_roles if role == "Infantry Block")
            if infantry_count < 2:
                modifier += 0.2  # Horde advantage vs low infantry
        
        if "Magic" in enemy.strengths:
            magic_count = sum(1 for unit in empire_army.units if "Wizard" in unit[0])
            if magic_count == 0:
                modifier += 0.3  # Magic advantage vs no magic defense
        
        if "Heavy Armor" in enemy.strengths:
            ranged_count = sum(1 for role in empire_roles if role == "Ranged")
            if ranged_count >= 2:
                modifier -= 0.15  # Heavy armor reduces shooting effectiveness
        
        return max(0.5, min(2.0, modifier))
    
    def _calculate_casualties(self, empire_army: ArmyComposition, enemy: EnemyArmy, victory: bool) -> float:
        """Calculate Empire army casualties"""
        base_casualties = 0.3 if victory else 0.7
        
        # Modify based on enemy type
        if "Numbers" in enemy.strengths:
            base_casualties += 0.1  # Hordes cause more casualties
        if "Heavy Armor" in enemy.strengths:
            base_casualties += 0.15  # Elite armies hit hard
        
        return min(1.0, base_casualties + random.uniform(-0.1, 0.1))
    
    def _calculate_casualties_enemy(self, enemy: EnemyArmy, empire_army: ArmyComposition, victory: bool) -> float:
        """Calculate enemy army casualties"""
        base_casualties = 0.3 if victory else 0.7
        
        empire_roles = [self.optimizer.empire_units[unit[0]].battlefield_role for unit in empire_army.units]
        
        # Artillery causes heavy casualties
        if "Artillery" in empire_roles:
            base_casualties += 0.2
        
        return min(1.0, base_casualties + random.uniform(-0.1, 0.1))
    
    def run_mega_simulation(self, num_battles: int = 300000) -> Dict:
        """Run massive simulation with 300,000 battles"""
        print(f"🏛️ EMPIRE MEGA SIMULATOR")
        print(f"=" * 80)
        print(f"🎯 Running {num_battles:,} battles...")
        print(f"⚔️ Testing against {len(self.enemy_armies)} enemy army types")
        print()
        
        # Generate Empire army variants
        print("🔧 Generating Empire army variants...")
        empire_armies = self.optimizer.generate_optimal_armies(20)  # 20 different armies
        print(f"✅ Generated {len(empire_armies)} Empire army variants")
        print()
        
        # Prepare battle combinations
        battles_per_combo = num_battles // (len(empire_armies) * len(self.enemy_armies))
        print(f"📊 {battles_per_combo:,} battles per army combination")
        print()
        
        # Run simulations
        results = {}
        total_battles = 0
        start_time = time.time()
        
        for i, empire_army in enumerate(empire_armies):
            army_results = []
            
            print(f"🏛️ Testing Empire Army #{i+1}: {empire_army.effectiveness_score:.1f} effectiveness")
            
            for enemy in self.enemy_armies:
                enemy_results = []
                
                # Run battles for this combination
                for battle in range(battles_per_combo):
                    result = self.simulate_battle(empire_army, enemy)
                    enemy_results.append(result)
                    total_battles += 1
                    
                    if total_battles % 10000 == 0:
                        elapsed = time.time() - start_time
                        rate = total_battles / elapsed
                        print(f"   ⚡ {total_battles:,} battles completed ({rate:.0f} battles/sec)")
                
                # Calculate stats vs this enemy
                victories = sum(1 for r in enemy_results if r.empire_victory)
                win_rate = victories / len(enemy_results)
                avg_margin = np.mean([r.victory_margin for r in enemy_results if r.empire_victory])
                avg_casualties = np.mean([r.empire_casualties for r in enemy_results])
                
                army_results.append({
                    'enemy': enemy.name,
                    'battles': len(enemy_results),
                    'victories': victories,
                    'win_rate': win_rate,
                    'avg_victory_margin': avg_margin if victories > 0 else 0,
                    'avg_casualties': avg_casualties,
                    'enemy_difficulty': enemy.difficulty
                })
            
            # Calculate overall army performance
            total_victories = sum(r['victories'] for r in army_results)
            total_army_battles = sum(r['battles'] for r in army_results)
            overall_win_rate = total_victories / total_army_battles
            
            # Weight by enemy difficulty
            weighted_score = sum(r['win_rate'] * r['enemy_difficulty'] for r in army_results) / sum(r['enemy_difficulty'] for r in army_results)
            
            results[f"Army_{i+1}"] = {
                'army_composition': empire_army,
                'enemy_results': army_results,
                'total_battles': total_army_battles,
                'total_victories': total_victories,
                'overall_win_rate': overall_win_rate,
                'weighted_score': weighted_score,
                'effectiveness_score': empire_army.effectiveness_score
            }
            
            print(f"   📈 Win Rate: {overall_win_rate:.1%} | Weighted Score: {weighted_score:.3f}")
            print()
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Simulation completed in {elapsed_time:.1f} seconds")
        print(f"🎯 Total battles: {total_battles:,}")
        print(f"⚡ Average rate: {total_battles/elapsed_time:.0f} battles/second")
        print()
        
        return results
    
    def analyze_results(self, results: Dict):
        """Analyze and display simulation results"""
        print(f"📊 MEGA SIMULATION ANALYSIS")
        print(f"=" * 80)
        print()
        
        # Sort armies by weighted score
        sorted_armies = sorted(results.items(), key=lambda x: x[1]['weighted_score'], reverse=True)
        
        print(f"🏆 TOP PERFORMING EMPIRE ARMIES")
        print(f"-" * 50)
        
        for rank, (army_name, data) in enumerate(sorted_armies[:5], 1):
            army = data['army_composition']
            print(f"\n#{rank}. {army_name}")
            print(f"   Weighted Score: {data['weighted_score']:.3f}")
            print(f"   Overall Win Rate: {data['overall_win_rate']:.1%}")
            print(f"   Total Battles: {data['total_battles']:,}")
            print(f"   Army Points: {army.total_points}")
            print(f"   Effectiveness: {army.effectiveness_score:.1f}")
            
            # Show army composition
            print(f"   Units:")
            for unit_name, size, upgrades in army.units:
                upgrade_str = f" ({', '.join(upgrades)})" if upgrades else ""
                print(f"     - {size}x {unit_name}{upgrade_str}")
        
        print(f"\n🎯 ENEMY MATCHUP ANALYSIS")
        print(f"-" * 50)
        
        # Best army's performance vs each enemy
        best_army = sorted_armies[0][1]
        print(f"\nBest Army vs Each Enemy Type:")
        
        for enemy_result in best_army['enemy_results']:
            print(f"   vs {enemy_result['enemy']}: {enemy_result['win_rate']:.1%} " +
                  f"({enemy_result['victories']:,}/{enemy_result['battles']:,})")
        
        print(f"\n🔍 TACTICAL INSIGHTS")
        print(f"-" * 50)
        
        # Analyze what makes armies successful
        top_3_armies = [data['army_composition'] for _, data in sorted_armies[:3]]
        
        # Count unit types in top armies
        unit_frequency = {}
        for army in top_3_armies:
            for unit_name, size, upgrades in army.units:
                if unit_name not in unit_frequency:
                    unit_frequency[unit_name] = 0
                unit_frequency[unit_name] += 1
        
        print(f"\nMost Common Units in Top 3 Armies:")
        for unit, count in sorted(unit_frequency.items(), key=lambda x: x[1], reverse=True):
            if count >= 2:
                print(f"   - {unit}: appears in {count}/3 top armies")
        
        return sorted_armies[0][1]['army_composition']  # Return best army

def main():
    """Run the mega simulation"""
    simulator = EmpireMegaSimulator()
    
    print("🏛️ EMPIRE MEGA SIMULATOR STARTING...")
    print("⚔️ Preparing for 300,000 battle simulation")
    print()
    
    # Run simulation
    results = simulator.run_mega_simulation(300000)
    
    # Analyze results
    best_army = simulator.analyze_results(results)
    
    print(f"\n🏆 ULTIMATE EMPIRE 750PT ARMY DISCOVERED!")
    print(f"=" * 80)
    
    # Display the ultimate army
    simulator.optimizer.display_army(best_army, 1)

if __name__ == '__main__':
    main() 