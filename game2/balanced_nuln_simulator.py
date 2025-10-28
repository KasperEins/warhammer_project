#!/usr/bin/env python3

import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Unit:
    name: str
    points: int
    category: str
    min_size: int
    max_size: int
    effectiveness: float
    special_rules: List[str]
    per_1000_restriction: bool = False

class NulnArmyBuilder:
    def __init__(self):
        self.database = {
            # Characters
            "General of the Empire": Unit("General of the Empire", 85, "character", 1, 1, 4.0, ["Leadership"], False),
            "Captain of the Empire": Unit("Captain of the Empire", 45, "character", 1, 1, 2.5, ["Leadership"], False),
            "Empire Engineer": Unit("Empire Engineer", 45, "character", 1, 1, 2.0, ["Artillery Support"], False),
            "Battle Standard Bearer": Unit("Battle Standard Bearer", 25, "upgrade", 1, 1, 1.0, ["Banner"], False),
            "Full Plate Armour": Unit("Full Plate Armour", 8, "upgrade", 1, 1, 0.5, ["Armor"], False),
            
            # Core Units
            "Nuln State Troops (20)": Unit("Nuln State Troops (20)", 100, "core", 20, 20, 3.5, ["Infantry"], False),
            "Nuln State Troops (15)": Unit("Nuln State Troops (15)", 75, "core", 15, 15, 2.8, ["Infantry"], False),
            "Nuln Veteran State Troops (15)": Unit("Nuln Veteran State Troops (15)", 105, "core", 15, 15, 3.2, ["Veteran"], False),
            "Nuln State Handgunners (10)": Unit("Nuln State Handgunners (10)", 60, "core", 10, 10, 2.5, ["Handgun Drill"], False),
            "Nuln State Handgunners (15)": Unit("Nuln State Handgunners (15)", 90, "core", 15, 15, 3.5, ["Handgun Drill"], False),
            "Nuln Veteran Outriders (8)": Unit("Nuln Veteran Outriders (8)", 152, "core", 8, 8, 3.8, ["Fast Cavalry"], False),
            
            # Special Units
            "Empire Greatswords (12)": Unit("Empire Greatswords (12)", 144, "special", 12, 12, 4.2, ["Elite"], False),
            "Empire Knights (5)": Unit("Empire Knights (5)", 100, "special", 5, 5, 3.8, ["Heavy Cavalry"], False),
            "Great Cannon": Unit("Great Cannon", 125, "special", 1, 1, 3.5, ["Artillery"], False),
            "Great Cannon with Gun Limbers": Unit("Great Cannon with Gun Limbers", 135, "special", 1, 1, 4.0, ["Artillery", "Vanguard"], False),
            
            # Rare Units
            "Helblaster Volley Gun": Unit("Helblaster Volley Gun", 120, "rare", 1, 1, 3.8, ["Multi-shot"], False),
            "Helblaster with Gun Limbers": Unit("Helblaster with Gun Limbers", 135, "rare", 1, 1, 4.2, ["Multi-shot", "Vanguard"], False),
            
            # Mercenaries
            "Imperial Dwarfs (12)": Unit("Imperial Dwarfs (12)", 108, "mercenary", 12, 12, 3.6, ["Resilient"], False),
        }

    def calculate_effectiveness(self, army: List[str]) -> float:
        base_effectiveness = sum(self.database[unit].effectiveness for unit in army)
        
        # Apply Nuln faction bonuses
        nuln_bonus = 1.0
        
        # Handgun Drill bonus
        handgun_units = sum(1 for unit in army if "Handgunners" in unit)
        nuln_bonus += handgun_units * 0.1
        
        # Artillery synergy bonus
        artillery_count = sum(1 for unit in army if any(arty in unit for arty in ["Cannon", "Mortar", "Helblaster"]))
        engineer_count = sum(1 for unit in army if "Engineer" in unit)
        if artillery_count > 0 and engineer_count > 0:
            nuln_bonus += artillery_count * engineer_count * 0.15
        
        # Gun Limbers bonus
        gun_limber_units = sum(1 for unit in army if "Gun Limbers" in unit)
        nuln_bonus += gun_limber_units * 0.1
        
        # Veteran Outriders bonus
        veteran_outrider_units = sum(1 for unit in army if "Veteran Outriders" in unit)
        nuln_bonus += veteran_outrider_units * 0.15
        
        return base_effectiveness * nuln_bonus

def simulate_battle(army1: List[str], army2: List[str], builder: NulnArmyBuilder) -> bool:
    """Simulate battle between two armies (True if army1 wins)"""
    effectiveness1 = builder.calculate_effectiveness(army1)
    
    # Balanced effectiveness calculation for enemy armies
    enemy_effectiveness_map = {
        # Characters (strong but single models)
        "Big Boss": 4.0, "Warlord": 4.2, "Lord": 4.5, "Prince": 5.0, "Noble": 3.8, 
        "Thane": 4.0, "Oldblood": 4.8, "Highborn": 4.5, "Warboss": 3.5,
        
        # Infantry (moderate effectiveness, reasonable scaling)
        "Warriors": 0.12, "Boyz": 0.10, "Clanrats": 0.08, "Spearmen": 0.13, 
        "Men-at-Arms": 0.09, "Skeleton": 0.06, "Zombies": 0.04, "Temple Guard": 0.18,
        "White Lions": 0.16, "Hammerers": 0.15, "Stormvermin": 0.14, "Grave Guard": 0.13,
        
        # Elite Units
        "Knights": 0.35, "Silver Helms": 0.28, "Black Knights": 0.32, "Boar Boyz": 0.25,
        "Dragon Ogres": 1.2, "Chaos Spawn": 0.8,
        
        # Ranged Units
        "Archers": 0.11, "Crossbowmen": 0.12, "Quarrellers": 0.13, "Glade Guard": 0.14,
        "Shades": 0.15, "Spider Riders": 0.18, "Dark Riders": 0.22,
        
        # War Machines
        "Cannon": 3.5, "Organ Gun": 3.2, "Rock Lobber": 2.8, "Trebuchet": 3.0,
        "Bolt Thrower": 2.5, "Ratling Gun": 2.2, "Warpfire": 2.0, "Doom Diver": 2.3,
        
        # Monsters
        "Salamander": 2.8, "Stegadon": 4.5, "Treeman": 5.2, "Corpse Cart": 1.8
    }
    
    effectiveness2 = 0
    for unit in army2:
        unit_name = unit.split('(')[0].strip()
        
        # Get unit count
        if '(' in unit and ')' in unit:
            try:
                count = int(unit.split('(')[1].split(')')[0])
            except:
                count = 1
        else:
            count = 1
        
        # Find effectiveness per model
        unit_effectiveness = 0.1  # Default for unknown units
        for key, value in enemy_effectiveness_map.items():
            if key.lower() in unit_name.lower():
                unit_effectiveness = value
                break
        
        # Calculate total unit effectiveness
        if unit_effectiveness >= 1.0:  # Single model units (characters, monsters, machines)
            effectiveness2 += unit_effectiveness
        else:  # Multi-model units (infantry, cavalry)
            effectiveness2 += unit_effectiveness * count
    
    # Apply faction bonuses for enemy armies
    if any("Dwarf" in unit for unit in army2):
        effectiveness2 *= 1.15  # Dwarf resilience
    if any("Elf" in unit for unit in army2):
        effectiveness2 *= 1.10  # Elf precision
    if any("Chaos" in unit for unit in army2):
        effectiveness2 *= 1.12  # Chaos corruption
    if "Cannon" in str(army2) or "Gun" in str(army2):
        effectiveness2 *= 1.08  # Artillery support
    
    # Battle resolution with randomness (±20%)
    roll1 = random.uniform(0.8, 1.2)
    roll2 = random.uniform(0.8, 1.2)
    
    final_score1 = effectiveness1 * roll1
    final_score2 = effectiveness2 * roll2
    
    return final_score1 > final_score2

def generate_army_templates() -> List[List[str]]:
    """Generate balanced Nuln army templates"""
    return [
        # Artillery Support Build (578 pts)
        [
            "General of the Empire", "Full Plate Armour",
            "Empire Engineer", "Battle Standard Bearer",
            "Nuln State Troops (20)",
            "Nuln State Handgunners (10)",
            "Great Cannon with Gun Limbers",
            "Helblaster Volley Gun"
        ],
        
        # Mobile Gunline Build (492 pts)
        [
            "General of the Empire",
            "Empire Engineer",
            "Nuln State Troops (15)",
            "Nuln Veteran Outriders (8)",
            "Great Cannon with Gun Limbers"
        ],
        
        # Combined Arms Build (553 pts)
        [
            "Captain of the Empire", "Battle Standard Bearer",
            "Empire Engineer",
            "Nuln Veteran State Troops (15)",
            "Empire Knights (5)",
            "Great Cannon",
            "Imperial Dwarfs (12)"
        ]
    ]

def generate_enemy_armies() -> Dict[str, List[str]]:
    """Generate balanced enemy army types"""
    return {
        "Orc Horde": ["Orc Big Boss", "Orc Boyz (30)", "Orc Boyz (25)", "Orc Arrer Boyz (20)", "Rock Lobber", "Orc Boar Boyz (10)"],
        "Dwarf Gunline": ["Dwarf Thane", "Dwarf Warriors (20)", "Dwarf Quarrellers (15)", "Dwarf Cannon", "Dwarf Organ Gun", "Dwarf Hammerers (15)"],
        "High Elf Elite": ["Elf Prince", "Spearmen (20)", "Archers (15)", "White Lions (15)", "Silver Helms (8)", "Eagle Claw Bolt Thrower"],
        "Chaos Warriors": ["Chaos Lord", "Chaos Warriors (15)", "Chaos Marauders (20)", "Chaos Knights (6)", "Dragon Ogres (3)", "Chaos Spawn"],
        "Skaven Horde": ["Skaven Warlord", "Clanrats (30)", "Clanrats (25)", "Stormvermin (15)", "Warpfire Thrower", "Ratling Gun"]
    }

def run_balanced_simulation():
    """Run balanced 1 million battle simulation"""
    print("⚖️ BALANCED NULN ARMY SIMULATOR")
    print("="*50)
    print("🎯 Realistic battle calculations")
    print("📊 1,000,000 battles for statistical confidence")
    print("🔧 Proper enemy army scaling")
    print()
    
    builder = NulnArmyBuilder()
    templates = generate_army_templates()
    enemy_armies = generate_enemy_armies()
    
    start_time = time.time()
    total_battles = 0
    results = {}
    
    for template_idx, army in enumerate(templates):
        army_name = f"Nuln Build {template_idx + 1}"
        wins = 0
        total_enemy_battles = 0
        enemy_results = {}
        
        army_points = sum(builder.database[unit].points for unit in army)
        effectiveness = builder.calculate_effectiveness(army)
        
        print(f"🏛️ Testing {army_name} ({army_points} pts, effectiveness {effectiveness:.2f})")
        
        for enemy_type, enemy_army in enemy_armies.items():
            enemy_wins = 0
            battles_per_enemy = 200000  # 200k battles per matchup
            
            for battle in range(battles_per_enemy):
                if simulate_battle(army, enemy_army, builder):
                    wins += 1
                    enemy_wins += 1
                total_battles += 1
                total_enemy_battles += 1
            
            win_rate = enemy_wins / battles_per_enemy
            enemy_results[enemy_type] = win_rate
            print(f"   vs {enemy_type:.<20} {win_rate:.1%}")
        
        overall_win_rate = wins / total_enemy_battles
        results[army_name] = {
            'army': army,
            'win_rate': overall_win_rate,
            'points': army_points,
            'effectiveness': effectiveness,
            'enemy_results': enemy_results
        }
        
        print(f"   Overall Win Rate: {overall_win_rate:.1%}")
        print()
    
    elapsed_time = time.time() - start_time
    battles_per_sec = total_battles / elapsed_time
    
    print(f"⚡ Completed {total_battles:,} battles in {elapsed_time:.1f}s")
    print(f"🚀 Processing speed: {battles_per_sec:,.0f} battles/second")
    print()
    
    # Sort and display results
    sorted_results = sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True)
    
    print("🏆 BALANCED SIMULATION RESULTS")
    print("="*40)
    
    for i, (name, data) in enumerate(sorted_results):
        print(f"\n🥇 RANK #{i+1}: {name}")
        print(f"   💫 Win Rate: {data['win_rate']:.1%}")
        print(f"   ⚔️ Points: {data['points']}")
        print(f"   🔥 Effectiveness: {data['effectiveness']:.2f}")
        print(f"   🎯 Army Composition:")
        
        for unit in data['army']:
            unit_data = builder.database[unit]
            print(f"      • {unit} ({unit_data.points} pts)")

if __name__ == "__main__":
    run_balanced_simulation() 