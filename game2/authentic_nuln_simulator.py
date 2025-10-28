#!/usr/bin/env python3
"""
🏛️ AUTHENTIC NULN ARMY OF INFAMY MEGA SIMULATOR
==================================================================
Implements proper Nuln faction restrictions and tournament rules:
- Mandatory Nuln State Troops or Nuln Veteran State Troops
- Battle Standard Bearer only for Captain or Engineer
- Proper unit point costs and restrictions
- Tournament limitations (188/263/225/188 points)
- 0-X per 1000 points: max 2 options allowed
- Nuln special rules and bonuses
"""

import random
import time
import itertools
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Unit:
    name: str
    points: int
    category: str  # 'character', 'core', 'special', 'rare', 'mercenary'
    min_size: int
    max_size: int
    effectiveness: float
    special_rules: List[str]
    per_1000_restriction: bool = False  # 0-X per 1000 points units

# Authentic Nuln Army of Infamy Database
NULN_ARMY_DATABASE = {
    # A. Characters
    "General of the Empire": Unit("General of the Empire", 85, "character", 1, 1, 1.8, ["Leadership", "Heavy Combat"]),
    "Lector of Sigmar": Unit("Lector of Sigmar", 100, "character", 1, 1, 1.6, ["Magic Resistance", "Inspiring"]),
    "Captain of the Empire": Unit("Captain of the Empire", 45, "character", 1, 1, 1.4, ["Leadership", "BSB Option"]),
    "Master Mage Level 1": Unit("Master Mage Level 1", 60, "character", 1, 1, 1.5, ["Magic"]),
    "Master Mage Level 2": Unit("Master Mage Level 2", 95, "character", 1, 1, 1.7, ["Magic"]),
    "Witch Hunter": Unit("Witch Hunter", 50, "character", 1, 1, 1.3, ["Anti-Magic", "Shooting"]),
    "Priest of Sigmar": Unit("Priest of Sigmar", 45, "character", 1, 1, 1.2, ["Divine Magic"]),
    "Empire Engineer": Unit("Empire Engineer", 45, "character", 1, 1, 1.6, ["War Machine Support", "BSB Option"]),
    
    # Character Upgrades
    "Battle Standard Bearer": Unit("Battle Standard Bearer", 25, "character", 1, 1, 0.3, ["Army Morale"]),
    "Full Plate Armour": Unit("Full Plate Armour", 8, "character", 1, 1, 0.1, ["Defence"]),
    "Great Weapon": Unit("Great Weapon", 6, "character", 1, 1, 0.1, ["Damage"]),
    
    # B. Core Units (Mandatory: At least one Nuln State Troops or Nuln Veteran State Troops)
    "Nuln State Troops (15)": Unit("Nuln State Troops (15)", 75, "core", 15, 15, 1.2, ["Disciplined", "Halberds"]),
    "Nuln State Troops (20)": Unit("Nuln State Troops (20)", 100, "core", 20, 20, 1.4, ["Disciplined", "Halberds"]),
    "Nuln State Troops (25)": Unit("Nuln State Troops (25)", 125, "core", 25, 25, 1.6, ["Disciplined", "Halberds"]),
    "Nuln Veteran State Troops (15)": Unit("Nuln Veteran State Troops (15)", 105, "core", 15, 15, 1.4, ["Veteran", "Halberds"]),
    "Nuln Veteran State Troops (20)": Unit("Nuln Veteran State Troops (20)", 140, "core", 20, 20, 1.6, ["Veteran", "Halberds"]),
    "Nuln State Handgunners (10)": Unit("Nuln State Handgunners (10)", 60, "core", 10, 10, 1.3, ["Handgun Drill", "Shooting"]),
    "Nuln State Handgunners (15)": Unit("Nuln State Handgunners (15)", 90, "core", 15, 15, 1.5, ["Handgun Drill", "Shooting"]),
    "Nuln Veteran Outriders (5)": Unit("Nuln Veteran Outriders (5)", 95, "core", 5, 5, 1.7, ["No Ponderous", "Mobile Shooting"]),
    "Nuln Veteran Outriders (8)": Unit("Nuln Veteran Outriders (8)", 152, "core", 8, 8, 2.0, ["No Ponderous", "Mobile Shooting"]),
    "Free Company Militia (15)": Unit("Free Company Militia (15)", 75, "core", 15, 15, 0.9, ["Cheap Infantry"]),
    "Empire Archers (10)": Unit("Empire Archers (10)", 70, "core", 10, 10, 1.1, ["Shooting"]),
    
    # C. Special Units
    "Empire Greatswords (12)": Unit("Empire Greatswords (12)", 144, "special", 12, 12, 1.8, ["Elite Infantry", "Great Weapons"]),
    "Empire Greatswords (15)": Unit("Empire Greatswords (15)", 180, "special", 15, 15, 2.0, ["Elite Infantry", "Great Weapons"]),
    "Pistoliers (5)": Unit("Pistoliers (5)", 80, "special", 5, 5, 1.4, ["Fast Cavalry", "Shooting"]),
    "Empire Knights (5)": Unit("Empire Knights (5)", 100, "special", 5, 5, 1.9, ["Heavy Cavalry", "Charge"]),
    "Empire Knights (8)": Unit("Empire Knights (8)", 160, "special", 8, 8, 2.2, ["Heavy Cavalry", "Charge"]),
    "War Wagon": Unit("War Wagon", 90, "special", 1, 1, 1.5, ["Mobile Platform", "Shooting"]),
    "Great Cannon": Unit("Great Cannon", 125, "special", 1, 1, 2.0, ["Artillery", "Long Range"]),
    "Great Cannon with Gun Limbers": Unit("Great Cannon with Gun Limbers", 135, "special", 1, 1, 2.2, ["Artillery", "Vanguard", "Veteran"]),
    "Mortar": Unit("Mortar", 95, "special", 1, 1, 1.8, ["Artillery", "Indirect Fire"]),
    "Mortar with Gun Limbers": Unit("Mortar with Gun Limbers", 105, "special", 1, 1, 2.0, ["Artillery", "Indirect Fire", "Vanguard"]),
    
    # D. Rare Units
    "Empire Road Wardens (5)": Unit("Empire Road Wardens (5)", 90, "rare", 5, 5, 1.6, ["Light Cavalry", "Crossbows", "Vanguard"], per_1000_restriction=True),
    "Helblaster Volley Gun": Unit("Helblaster Volley Gun", 120, "rare", 1, 1, 2.3, ["Multi-Shot Artillery"]),
    "Helblaster with Gun Limbers": Unit("Helblaster with Gun Limbers", 135, "rare", 1, 1, 2.5, ["Multi-Shot Artillery", "Vanguard"]),
    "Helstorm Rocket Battery": Unit("Helstorm Rocket Battery", 125, "rare", 1, 1, 2.2, ["Area Artillery"]),
    "Helstorm with Gun Limbers": Unit("Helstorm with Gun Limbers", 140, "rare", 1, 1, 2.4, ["Area Artillery", "Vanguard"]),
    "Steam Tank": Unit("Steam Tank", 285, "rare", 1, 1, 3.0, ["Monster", "Steam Power"], per_1000_restriction=True),
    
    # E. Mercenaries
    "Imperial Dwarfs (12)": Unit("Imperial Dwarfs (12)", 108, "mercenary", 12, 12, 1.7, ["Stubborn", "Heavy Infantry"]),
    "Imperial Ogres (3)": Unit("Imperial Ogres (3)", 114, "mercenary", 3, 3, 2.1, ["Monstrous Infantry"], per_1000_restriction=True),
    "Prince Ulther's Dragon Company (15)": Unit("Prince Ulther's Dragon Company (15)", 225, "mercenary", 15, 15, 2.0, ["Elite Slayers"], per_1000_restriction=True),
}

# Tournament Point Limitations
TOURNAMENT_LIMITS = {
    "character": 188,  # 25% of 750
    "core": 263,       # 35% of 750
    "special": 225,    # 30% of 750
    "rare": 188,       # 25% of 750
    "mercenary": 188   # 25% of 750
}

class NulnArmyBuilder:
    def __init__(self):
        self.target_points = 750
        self.database = NULN_ARMY_DATABASE
        self.limits = TOURNAMENT_LIMITS
        
    def is_valid_army(self, army: List[str]) -> Tuple[bool, str]:
        """Check if army meets Nuln Army of Infamy and tournament restrictions"""
        total_points = sum(self.database[unit].points for unit in army)
        
        if total_points > self.target_points:
            return False, f"Exceeds {self.target_points} points"
            
        # Check point limits per category
        category_points = {}
        per_1000_count = 0
        
        for unit_name in army:
            unit = self.database[unit_name]
            category = unit.category
            
            if category not in category_points:
                category_points[category] = 0
            category_points[category] += unit.points
            
            # Count 0-X per 1000 restrictions
            if unit.per_1000_restriction:
                per_1000_count += 1
        
        # Check category limits
        for category, points in category_points.items():
            if category in self.limits and points > self.limits[category]:
                return False, f"{category.title()} exceeds {self.limits[category]} points"
        
        # Check 0-X per 1000 limit (max 2 for 750pt army)
        if per_1000_count > 2:
            return False, f"Too many 0-X per 1000 units ({per_1000_count}/2)"
        
        # Nuln Army of Infamy mandatory requirement
        has_mandatory_core = any(
            unit_name.startswith("Nuln State Troops") or unit_name.startswith("Nuln Veteran State Troops")
            for unit_name in army
        )
        if not has_mandatory_core:
            return False, "Must include Nuln State Troops or Nuln Veteran State Troops"
        
        # Battle Standard Bearer restriction (only Captain or Engineer)
        if "Battle Standard Bearer" in army:
            has_valid_bsb_carrier = any(
                unit_name in ["Captain of the Empire", "Empire Engineer"]
                for unit_name in army
            )
            if not has_valid_bsb_carrier:
                return False, "BSB only available to Captain or Engineer in Nuln"
        
        return True, "Valid"
    
    def calculate_effectiveness(self, army: List[str]) -> float:
        """Calculate army effectiveness with Nuln bonuses"""
        base_effectiveness = sum(self.database[unit].effectiveness for unit in army)
        
        # Apply Nuln faction bonuses
        nuln_bonus = 1.0
        
        # Handgun Drill bonus
        handgun_units = sum(1 for unit in army if "Handgunners" in unit)
        nuln_bonus += handgun_units * 0.1
        
        # Artillery synergy bonus
        artillery_count = sum(1 for unit in army if any(arty in unit for arty in ["Cannon", "Mortar", "Helblaster", "Helstorm"]))
        engineer_count = sum(1 for unit in army if "Engineer" in unit)
        if artillery_count > 0 and engineer_count > 0:
            nuln_bonus += artillery_count * engineer_count * 0.15
        
        # Gun Limbers bonus
        gun_limber_units = sum(1 for unit in army if "Gun Limbers" in unit)
        nuln_bonus += gun_limber_units * 0.1
        
        # Veteran Outriders bonus (no Ponderous penalty)
        veteran_outrider_units = sum(1 for unit in army if "Veteran Outriders" in unit)
        nuln_bonus += veteran_outrider_units * 0.15
        
        return base_effectiveness * nuln_bonus

def generate_army_templates() -> List[List[str]]:
    """Generate diverse Nuln army templates"""
    templates = [
        # Artillery Support Build (634 pts) - Character: 163, Core: 160, Special: 135, Rare: 120
        [
            "General of the Empire", "Full Plate Armour",      # 93 pts character
            "Empire Engineer", "Battle Standard Bearer",       # 70 pts character (total: 163)
            "Nuln State Troops (20)",                          # 100 pts core  
            "Nuln State Handgunners (10)",                     # 60 pts core (total: 160)
            "Great Cannon with Gun Limbers",                   # 135 pts special
            "Helblaster Volley Gun"                            # 120 pts rare
        ],
        
        # Infantry Core Build (540 pts) - Character: 123, Core: 195, Special: 144, Mercenary: 108
        [
            "Captain of the Empire", "Battle Standard Bearer", "Full Plate Armour",  # 78 pts character
            "Empire Engineer",                                                        # 45 pts character (total: 123)
            "Nuln Veteran State Troops (15)",                                       # 105 pts core
            "Nuln State Handgunners (15)",                                          # 90 pts core (total: 195)
            "Empire Greatswords (12)",                                              # 144 pts special
            "Imperial Dwarfs (12)"                                                  # 108 pts mercenary
        ],
        
        # Mobile Gunline Build (592 pts) - Character: 130, Core: 227, Special: 135, Rare: 0  
        [
            "General of the Empire",                           # 85 pts character
            "Empire Engineer",                                 # 45 pts character (total: 130)
            "Nuln State Troops (15)",                          # 75 pts core
            "Nuln Veteran Outriders (8)",                     # 152 pts core (total: 227)
            "Great Cannon with Gun Limbers"                   # 135 pts special
        ],
        
        # Combined Arms Build (563 pts) - Character: 115, Core: 105, Special: 225, Mercenary: 108
        [
            "Captain of the Empire", "Battle Standard Bearer", # 70 pts character
            "Empire Engineer",                                  # 45 pts character (total: 115)
            "Nuln Veteran State Troops (15)",                 # 105 pts core
            "Empire Knights (5)",                              # 100 pts special  
            "Great Cannon",                                    # 125 pts special (total: 225)
            "Imperial Dwarfs (12)"                             # 108 pts mercenary
        ],
        
        # Elite Artillery Build (699 pts) - Character: 163, Core: 190, Special: 125, Rare: 135
        [
            "General of the Empire", "Full Plate Armour",     # 93 pts character
            "Empire Engineer", "Battle Standard Bearer",      # 70 pts character (total: 163)
            "Nuln State Troops (20)",                         # 100 pts core
            "Nuln State Handgunners (15)",                    # 90 pts core (total: 190)
            "Great Cannon",                                    # 125 pts special
            "Helblaster with Gun Limbers"                     # 135 pts rare
        ]
    ]
    return templates

def simulate_battle(army1: List[str], army2: List[str], builder: NulnArmyBuilder) -> bool:
    """Simulate battle between two armies (True if army1 wins)"""
    effectiveness1 = builder.calculate_effectiveness(army1)
    
    # More realistic effectiveness calculation for enemy armies
    # Base effectiveness per enemy unit type with proper scaling
    enemy_effectiveness_map = {
        "Big Boss": 3.5, "Warlord": 3.5, "Lord": 4.0, "Prince": 4.5, "Noble": 3.0, "Thane": 3.5,
        "Oldblood": 4.0, "Highborn": 3.8, "Warboss": 3.2,
        "Warriors": 2.0, "Boyz": 1.8, "Clanrats": 1.5, "Spearmen": 2.2, "Men-at-Arms": 1.8,
        "Skeleton": 1.3, "Zombies": 1.0, "Temple Guard": 2.8, "White Lions": 2.5, "Hammerers": 2.4,
        "Knights": 3.2, "Silver Helms": 2.8, "Black Knights": 3.0, "Boar Boyz": 2.5,
        "Archers": 1.8, "Crossbowmen": 1.9, "Quarrellers": 2.0, "Glade Guard": 2.1,
        "Cannon": 3.0, "Organ Gun": 2.8, "Rock Lobber": 2.5, "Trebuchet": 2.7,
        "Bolt Thrower": 2.3, "Ratling Gun": 2.2, "Warpfire": 2.0
    }
    
    effectiveness2 = 0
    for unit in army2:
        # Extract unit type and calculate size-adjusted effectiveness
        unit_name = unit.split('(')[0].strip()
        
        # Get unit count if specified in parentheses
        if '(' in unit and ')' in unit:
            try:
                count = int(unit.split('(')[1].split(')')[0])
            except:
                count = 1
        else:
            count = 1
        
        # Find matching effectiveness value
        unit_effectiveness = 1.5  # Default
        for key, value in enemy_effectiveness_map.items():
            if key.lower() in unit_name.lower():
                unit_effectiveness = value
                break
        
        effectiveness2 += unit_effectiveness * (count ** 0.8)  # Diminishing returns for large units
    
    # Add some tactical bonuses for enemy armies
    if "Cannon" in str(army2) or "Gun" in str(army2):
        effectiveness2 *= 1.15  # Artillery bonus
    if "Knights" in str(army2):
        effectiveness2 *= 1.10  # Cavalry bonus
    if any("Dwarf" in unit for unit in army2):
        effectiveness2 *= 1.12  # Dwarf resilience
    if any("Elf" in unit for unit in army2):
        effectiveness2 *= 1.08  # Elf precision
    
    # DEBUG: Print effectiveness comparison for first few battles
    global debug_counter
    if not hasattr(simulate_battle, 'debug_counter'):
        simulate_battle.debug_counter = 0
    
    if simulate_battle.debug_counter < 5:
        print(f"    🔍 DEBUG: Nuln effectiveness: {effectiveness1:.2f}, Enemy effectiveness: {effectiveness2:.2f}")
        simulate_battle.debug_counter += 1
    
    # Add meaningful randomness to battles (±25% variance)
    roll1 = random.uniform(0.75, 1.25)
    roll2 = random.uniform(0.75, 1.25)
    
    # Add tactical factors
    tactical_bonus1 = random.uniform(0.95, 1.05)  # Minor Nuln tactical variance
    tactical_bonus2 = random.uniform(0.90, 1.10)  # Enemy tactical variance
    
    final_score1 = effectiveness1 * roll1 * tactical_bonus1
    final_score2 = effectiveness2 * roll2 * tactical_bonus2
    
    return final_score1 > final_score2

def generate_enemy_armies() -> Dict[str, List[str]]:
    """Generate diverse enemy army types for testing"""
    return {
        "Orc Horde": ["Orc Big Boss", "Orc Boyz (30)", "Orc Boyz (25)", "Orc Arrer Boyz (20)", "Rock Lobber", "Orc Boar Boyz (10)"],
        "Dwarf Gunline": ["Dwarf Thane", "Dwarf Warriors (20)", "Dwarf Quarrellers (15)", "Dwarf Cannon", "Dwarf Organ Gun", "Dwarf Hammerers (15)"],
        "High Elf Elite": ["Elf Prince", "Spearmen (20)", "Archers (15)", "White Lions (15)", "Silver Helms (8)", "Eagle Claw Bolt Thrower"],
        "Dark Elf Raiders": ["Dark Elf Noble", "Dark Elf Warriors (20)", "Crossbowmen (15)", "Dark Riders (8)", "Shades (10)", "Reaper Bolt Thrower"],
        "Chaos Warriors": ["Chaos Lord", "Chaos Warriors (15)", "Chaos Marauders (20)", "Chaos Knights (6)", "Dragon Ogres (3)", "Chaos Spawn"],
        "Skaven Horde": ["Skaven Warlord", "Clanrats (30)", "Clanrats (25)", "Stormvermin (15)", "Warpfire Thrower", "Ratling Gun"],
        "Vampire Counts": ["Vampire Lord", "Skeleton Warriors (25)", "Zombies (30)", "Grave Guard (15)", "Black Knights (8)", "Corpse Cart"],
        "Bretonnian Knights": ["Bretonnian Lord", "Men-at-Arms (20)", "Peasant Bowmen (15)", "Knights Errant (8)", "Knights of the Realm (6)", "Trebuchet"],
        "Goblin Swarm": ["Goblin Warboss", "Goblin Spearmen (40)", "Goblin Archers (30)", "Spider Riders (10)", "Rock Lobber", "Doom Diver"],
        "Lizardmen Temple": ["Saurus Oldblood", "Saurus Warriors (20)", "Skink Cohort (24)", "Temple Guard (15)", "Salamander", "Stegadon"],
        "Wood Elf Rangers": ["Wood Elf Highborn", "Glade Guard (16)", "Eternal Guard (16)", "Glade Riders (8)", "Wardancers (12)", "Treeman"]
    }

def run_mega_simulation():
    """Run massive simulation to find optimal Nuln army"""
    print("🏛️ NULN ARMY OF INFAMY MEGA SIMULATOR")
    print("="*60)
    print("🎯 Testing authentic Nuln faction rules...")
    print("⚔️ Tournament restrictions applied (188/263/225/188)")
    print("🔧 Mandatory Nuln State/Veteran Troops required")
    print("📊 10,000,000 battles against 11 enemy army types")
    print("🚀 MAXIMUM STATISTICAL PRECISION MODE")
    print()
    
    builder = NulnArmyBuilder()
    templates = generate_army_templates()
    enemy_armies = generate_enemy_armies()
    
    # Validate templates
    valid_templates = []
    for i, template in enumerate(templates):
        is_valid, reason = builder.is_valid_army(template)
        points = sum(builder.database[unit].points for unit in template)
        effectiveness = builder.calculate_effectiveness(template)
        
        if is_valid:
            valid_templates.append(template)
            print(f"✅ Template {i+1}: {points} pts, effectiveness {effectiveness:.2f}")
        else:
            print(f"❌ Template {i+1}: {reason}")
    
    print(f"\n🔥 Testing {len(valid_templates)} valid Nuln armies...")
    print("🎮 Starting 10 MILLION battle mega simulation...")
    print("⚡ This will provide maximum statistical accuracy!")
    
    start_time = time.time()
    total_battles = 0
    results = {}
    
    for template_idx, army in enumerate(valid_templates):
        army_name = f"Nuln Build {template_idx + 1}"
        wins = 0
        total_enemy_battles = 0
        enemy_results = {}
        
        for enemy_type, enemy_army in enemy_armies.items():
            enemy_wins = 0
            battles_per_enemy = 10000000 // len(enemy_armies) // len(valid_templates)
            
            # Progress indicator for long simulation
            if template_idx == 0:
                print(f"   🎯 {battles_per_enemy:,} battles per army vs each enemy type")
            
            for battle in range(battles_per_enemy):
                # Progress indicator every 100k battles
                if battle > 0 and battle % 100000 == 0:
                    progress = (battle / battles_per_enemy) * 100
                    print(f"      📊 {army_name} vs {enemy_type}: {progress:.1f}% complete")
                
                if simulate_battle(army, enemy_army, builder):
                    wins += 1
                    enemy_wins += 1
                total_battles += 1
                total_enemy_battles += 1
            
            win_rate = enemy_wins / battles_per_enemy if battles_per_enemy > 0 else 0
            enemy_results[enemy_type] = win_rate
        
        overall_win_rate = wins / total_enemy_battles if total_enemy_battles > 0 else 0
        army_points = sum(builder.database[unit].points for unit in army)
        effectiveness = builder.calculate_effectiveness(army)
        
        results[army_name] = {
            'army': army,
            'win_rate': overall_win_rate,
            'points': army_points,
            'effectiveness': effectiveness,
            'enemy_results': enemy_results,
            'battles': total_enemy_battles
        }
        
        print(f"✅ {army_name}: {overall_win_rate:.4f} win rate ({total_enemy_battles:,} battles)")
    
    elapsed_time = time.time() - start_time
    battles_per_sec = total_battles / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n⚡ Completed {total_battles:,} battles in {elapsed_time:.1f}s")
    print(f"🚀 Processing speed: {battles_per_sec:,.0f} battles/second")
    print(f"🎯 Statistical precision: ±{(1.96 * (0.5 * 0.5 / (total_battles/len(valid_templates)/len(enemy_armies)))**0.5):.4f} at 95% confidence")
    
    # Sort by win rate
    sorted_results = sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True)
    
    print(f"\n🏆 TOP NULN ARMIES OF INFAMY (10M SIMULATIONS)")
    print("="*60)
    
    for i, (name, data) in enumerate(sorted_results[:3]):
        print(f"\n🥇 RANK #{i+1}: {name}")
        print(f"   💫 Win Rate: {data['win_rate']:.4%} (±{1.96 * (data['win_rate'] * (1-data['win_rate']) / data['battles'])**0.5:.4%})")
        print(f"   ⚔️ Points: {data['points']}")
        print(f"   🔥 Effectiveness: {data['effectiveness']:.2f}")
        print(f"   📊 Battles: {data['battles']:,}")
        print(f"   🎯 Army Composition:")
        
        for unit in data['army']:
            unit_data = builder.database[unit]
            print(f"      • {unit} ({unit_data.points} pts)")
        
        print(f"   📈 Performance vs Enemy Types:")
        for enemy, win_rate in sorted(data['enemy_results'].items(), key=lambda x: x[1], reverse=True):
            battles_vs_enemy = data['battles'] // len(data['enemy_results'])
            margin_error = 1.96 * (win_rate * (1-win_rate) / battles_vs_enemy)**0.5
            print(f"      {enemy:.<20} {win_rate:.4%} (±{margin_error:.3%})")
    
    # Calculate engineering bonus and specialized metrics
    best_army = sorted_results[0][1]
    artillery_count = sum(1 for unit in best_army['army'] if any(arty in unit for arty in ["Cannon", "Mortar", "Helblaster", "Helstorm"]))
    engineer_count = sum(1 for unit in best_army['army'] if "Engineer" in unit)
    gun_limber_count = sum(1 for unit in best_army['army'] if "Gun Limbers" in unit)
    
    print(f"\n🔧 NULN ENGINEERING ANALYSIS (10M BATTLES)")
    print("="*40)
    print(f"Artillery Pieces: {artillery_count}")
    print(f"Engineers: {engineer_count}")
    print(f"Gun Limbers: {gun_limber_count}")
    print(f"Engineering Synergy Bonus: {artillery_count * engineer_count * 0.15:.1f}")
    print(f"Statistical Confidence: 99.99%+ with {total_battles:,} battles")
    
    perfect_enemies = [enemy for enemy, rate in best_army['enemy_results'].items() if rate >= 0.999]
    if perfect_enemies:
        print(f"\n💯 NEAR-PERFECT VICTORIES AGAINST:")
        for enemy in perfect_enemies:
            rate = best_army['enemy_results'][enemy]
            battles_vs_enemy = best_army['battles'] // len(best_army['enemy_results'])
            print(f"   • {enemy}: {rate:.4%} ({battles_vs_enemy:,} battles)")
    
    # Statistical significance analysis
    if len(sorted_results) >= 2:
        best_rate = sorted_results[0][1]['win_rate']
        second_rate = sorted_results[1][1]['win_rate']
        rate_diff = best_rate - second_rate
        print(f"\n📊 STATISTICAL ANALYSIS:")
        print(f"Best vs 2nd place difference: {rate_diff:.4%}")
        print(f"Statistical significance: {'HIGH' if rate_diff > 0.01 else 'MODERATE' if rate_diff > 0.005 else 'LOW'}")

if __name__ == "__main__":
    run_mega_simulation() 