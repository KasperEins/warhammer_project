#!/usr/bin/env python3

import random
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum

class CounterStrategy(Enum):
    ANTI_CAVALRY = "anti_cavalry"
    ANTI_ELITE = "anti_elite" 
    ANTI_MONSTER = "anti_monster"
    MOBILE_RESPONSE = "mobile_response"
    DEFENSIVE_CASTLE = "defensive_castle"

@dataclass
class Unit:
    name: str
    points: int
    category: str
    effectiveness: float
    unit_type: str
    special_rules: List[str]
    counter_bonuses: Dict[str, float]  # Bonuses vs specific enemy types

@dataclass
class Enemy:
    name: str
    army_type: str
    total_points: int
    primary_threat: str  # cavalry, elite_infantry, monsters, shooting
    faction_bonuses: Dict[str, float]
    weaknesses: List[str]  # What counters this army

class NulnCounterOptimizer:
    def __init__(self):
        self.database = self._create_enhanced_database()
        self.problem_enemies = self._create_problem_enemies()
        
    def _create_enhanced_database(self) -> Dict[str, Unit]:
        """Enhanced unit database with counter-strategy bonuses"""
        return {
            # Characters - Leadership and buffs
            "General of the Empire": Unit(
                "General of the Empire", 85, "character", 4.0, "character", 
                ["Leadership", "Inspiring"], 
                {"cavalry": 0.1, "elite": 0.15, "monsters": 0.1}
            ),
            "Empire Engineer": Unit(
                "Empire Engineer", 45, "character", 2.0, "character",
                ["Artillery Support", "Big Guns"],
                {"elite": 0.2, "monsters": 0.25}  # Artillery expertise
            ),
            "Captain of the Empire": Unit(
                "Captain of the Empire", 45, "character", 2.5, "character",
                ["Leadership", "Combat"],
                {"cavalry": 0.15, "elite": 0.1}
            ),
            
            # Enhanced Equipment
            "Full Plate Armour": Unit(
                "Full Plate Armour", 8, "upgrade", 0.3, "equipment",
                ["Armor Save"], {"elite": 0.1, "cavalry": 0.1}
            ),
            "Battle Standard Bearer": Unit(
                "Battle Standard Bearer", 25, "upgrade", 1.0, "character",
                ["Banner", "Reroll"], {"all": 0.1}
            ),
            
            # Core Units - Balanced foundation
            "Nuln State Troops (25)": Unit(
                "Nuln State Troops (25)", 125, "core", 4.0, "infantry",
                ["Steadfast", "Faction Troops", "Large Unit"],
                {"cavalry": 0.3, "elite": 0.1}  # Good vs cavalry charges
            ),
            "Nuln State Troops (20)": Unit(
                "Nuln State Troops (20)", 100, "core", 3.2, "infantry",
                ["Steadfast", "Faction Troops"], 
                {"cavalry": 0.25, "elite": 0.05}
            ),
            "Nuln Veteran State Troops (15)": Unit(
                "Nuln Veteran State Troops (15)", 105, "core", 3.0, "infantry",
                ["Veteran", "Steadfast"], 
                {"elite": 0.2, "cavalry": 0.15}
            ),
            "Nuln State Handgunners (15)": Unit(
                "Nuln State Handgunners (15)", 90, "core", 4.0, "infantry",
                ["Handgun Drill", "Armor Piercing", "Large Unit"],
                {"elite": 0.4, "monsters": 0.3, "cavalry": 0.2}  # Excellent vs armor
            ),
            "Nuln State Handgunners (10)": Unit(
                "Nuln State Handgunners (10)", 60, "core", 2.8, "infantry",
                ["Handgun Drill", "Armor Piercing"],
                {"elite": 0.35, "monsters": 0.25, "cavalry": 0.15}
            ),
            "Nuln Veteran Outriders (8)": Unit(
                "Nuln Veteran Outriders (8)", 152, "core", 4.2, "cavalry",
                ["Fast Cavalry", "Veteran", "Mobile Shooting"],
                {"cavalry": 0.4, "monsters": 0.2}  # Mobile counter-cavalry
            ),
            
            # Special Units - Specialized roles
            "Empire Greatswords (15)": Unit(
                "Empire Greatswords (15)", 180, "special", 5.5, "infantry",
                ["Elite", "Great Weapons", "Stubborn", "Large Unit"],
                {"elite": 0.3, "monsters": 0.4, "cavalry": 0.2}  # Elite killers
            ),
            "Empire Greatswords (12)": Unit(
                "Empire Greatswords (12)", 144, "special", 4.8, "infantry",
                ["Elite", "Great Weapons", "Stubborn"],
                {"elite": 0.25, "monsters": 0.35, "cavalry": 0.15}
            ),
            "Empire Knights (8)": Unit(
                "Empire Knights (8)", 160, "special", 6.0, "cavalry",
                ["Heavy Cavalry", "Lance", "Devastating Charge", "Large Unit"],
                {"cavalry": 0.5, "elite": 0.2}  # Counter-charge unit
            ),
            "Empire Knights (5)": Unit(
                "Empire Knights (5)", 100, "special", 4.5, "cavalry",
                ["Heavy Cavalry", "Lance", "Devastating Charge"],
                {"cavalry": 0.4, "elite": 0.15}
            ),
            "Great Cannon": Unit(
                "Great Cannon", 125, "special", 4.0, "war_machine",
                ["Artillery", "High Strength", "Armor Piercing"],
                {"elite": 0.3, "monsters": 0.5, "cavalry": 0.1}
            ),
            "Great Cannon with Gun Limbers": Unit(
                "Great Cannon with Gun Limbers", 135, "special", 4.5, "war_machine",
                ["Artillery", "Vanguard", "Mobile", "Armor Piercing"],
                {"elite": 0.35, "monsters": 0.55, "cavalry": 0.2}  # Mobile positioning
            ),
            
            # Rare Units - Game changers
            "Helblaster Volley Gun": Unit(
                "Helblaster Volley Gun", 120, "rare", 4.2, "war_machine",
                ["Multi-shot", "High Volume"],
                {"cavalry": 0.6, "elite": 0.2}  # Excellent vs cavalry
            ),
            "Helblaster with Gun Limbers": Unit(
                "Helblaster with Gun Limbers", 135, "rare", 4.8, "war_machine",
                ["Multi-shot", "Vanguard", "Mobile"],
                {"cavalry": 0.7, "elite": 0.25}  # Premier cavalry killer
            ),
            "Mortar": Unit(
                "Mortar", 90, "rare", 3.5, "war_machine",
                ["Artillery", "Indirect Fire", "Area Effect"],
                {"elite": 0.4, "cavalry": 0.3}  # Ignores armor
            ),
            
            # Mercenaries - Specialized support
            "Imperial Dwarfs (15)": Unit(
                "Imperial Dwarfs (15)", 135, "mercenary", 4.8, "infantry",
                ["Resilient", "Armor Save", "Stubborn", "Large Unit"],
                {"elite": 0.25, "monsters": 0.3, "cavalry": 0.4}  # Excellent anchor
            ),
            "Imperial Dwarfs (12)": Unit(
                "Imperial Dwarfs (12)", 108, "mercenary", 4.0, "infantry",
                ["Resilient", "Armor Save", "Stubborn"],
                {"elite": 0.2, "monsters": 0.25, "cavalry": 0.35}
            ),
        }
    
    def _create_problem_enemies(self) -> Dict[str, Enemy]:
        """Define our most problematic enemies with detailed analysis"""
        return {
            "Bretonnian Knights": Enemy(
                "Bretonnian Knights", "cavalry_mobile", 750,
                "cavalry", 
                {"chivalry": 1.20, "cavalry_charge": 1.25},
                ["anti_cavalry", "defensive_positioning", "volume_shooting"]
            ),
            "Chaos Warriors": Enemy(
                "Chaos Warriors", "elite_small", 760,
                "elite_infantry",
                {"favor_of_gods": 1.30, "fear": 1.15},
                ["armor_piercing", "volume_fire", "artillery"]
            ),
            "High Elf Elite": Enemy(
                "High Elf Elite", "elite_small", 730,
                "elite_infantry",
                {"always_strikes_first": 1.25, "martial_prowess": 1.10},
                ["artillery", "volume_shooting", "area_denial"]
            ),
            "Lizardmen Temple": Enemy(
                "Lizardmen Temple", "monster_mash", 780,
                "monsters",
                {"cold_blooded": 1.15, "scaly_skin": 1.10, "ancient_power": 1.20},
                ["high_strength", "artillery", "armor_piercing"]
            )
        }
    
    def generate_counter_strategies(self) -> Dict[str, List[List[str]]]:
        """Generate specialized army builds to counter each problem enemy"""
        strategies = {}
        
        # Anti-Cavalry builds (vs Bretonnians)
        strategies["Anti-Cavalry"] = [
            # Defensive Castle Strategy
            ["General of the Empire", "Full Plate Armour", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (25)", "Nuln State Handgunners (15)", 
             "Helblaster with Gun Limbers", "Imperial Dwarfs (15)"],
            
            # Mobile Counter-Cavalry
            ["General of the Empire", "Captain of the Empire", "Battle Standard Bearer",
             "Nuln Veteran State Troops (15)", "Nuln Veteran Outriders (8)",
             "Empire Knights (8)", "Helblaster Volley Gun"],
            
            # Volume Shooting Wall
            ["General of the Empire", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (20)", "Nuln State Handgunners (15)", "Nuln State Handgunners (10)",
             "Helblaster with Gun Limbers"]
        ]
        
        # Anti-Elite builds (vs Chaos/High Elves)
        strategies["Anti-Elite"] = [
            # Artillery Superiority
            ["General of the Empire", "Full Plate Armour", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (20)", "Nuln State Handgunners (15)",
             "Great Cannon with Gun Limbers", "Mortar"],
            
            # Elite vs Elite
            ["General of the Empire", "Captain of the Empire", "Battle Standard Bearer",
             "Empire Greatswords (15)", "Nuln State Handgunners (15)",
             "Great Cannon with Gun Limbers", "Imperial Dwarfs (12)"],
            
            # Mass Shooting
            ["General of the Empire", "Empire Engineer", "Battle Standard Bearer",
             "Nuln Veteran State Troops (15)", "Nuln State Handgunners (15)", "Nuln State Handgunners (10)",
             "Helblaster Volley Gun"]
        ]
        
        # Anti-Monster builds (vs Lizardmen)
        strategies["Anti-Monster"] = [
            # High Strength Focus
            ["General of the Empire", "Full Plate Armour", "Empire Engineer", "Battle Standard Bearer",
             "Empire Greatswords (15)", "Nuln State Handgunners (10)",
             "Great Cannon with Gun Limbers", "Great Cannon"],
            
            # Double Artillery
            ["General of the Empire", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (20)", "Nuln State Handgunners (10)",
             "Great Cannon with Gun Limbers", "Mortar"],
            
            # Combined Arms Anti-Monster
            ["General of the Empire", "Captain of the Empire", "Battle Standard Bearer",
             "Empire Greatswords (12)", "Nuln State Handgunners (15)",
             "Great Cannon", "Imperial Dwarfs (15)"]
        ]
        
        return strategies
    
    def calculate_counter_effectiveness(self, army_units: List[str], enemy: Enemy) -> float:
        """Calculate effectiveness with counter-strategy bonuses"""
        base_effectiveness = sum(self.database[unit].effectiveness for unit in army_units)
        
        # Apply counter-strategy bonuses
        counter_bonus = 1.0
        for unit_name in army_units:
            unit = self.database[unit_name]
            # Check for specific counter bonuses
            if enemy.primary_threat in unit.counter_bonuses:
                counter_bonus += unit.counter_bonuses[enemy.primary_threat]
            if "all" in unit.counter_bonuses:
                counter_bonus += unit.counter_bonuses["all"]
        
        # Apply faction bonuses
        faction_multiplier = 1.0
        
        # Nuln faction bonuses
        handgun_units = sum(1 for unit in army_units if "Handgunners" in unit)
        faction_multiplier += handgun_units * 0.08
        
        engineers = sum(1 for unit in army_units if "Engineer" in unit)
        artillery = sum(1 for unit in army_units if any(art in unit for art in ["Cannon", "Helblaster", "Mortar"]))
        if engineers > 0 and artillery > 0:
            faction_multiplier += min(engineers, artillery) * 0.12
        
        gun_limbers = sum(1 for unit in army_units if "Gun Limbers" in unit)
        faction_multiplier += gun_limbers * 0.06
        
        if artillery >= 2:
            faction_multiplier += 0.10
        
        return base_effectiveness * counter_bonus * faction_multiplier
    
    def calculate_enemy_effectiveness(self, enemy: Enemy) -> float:
        """Calculate enemy army effectiveness"""
        base_effectiveness = (enemy.total_points / 750.0) * 20.0
        
        total_bonus = 1.0
        for bonus in enemy.faction_bonuses.values():
            total_bonus *= bonus
            
        return base_effectiveness * total_bonus
    
    def simulate_counter_battle(self, army_units: List[str], enemy: Enemy) -> Tuple[bool, Dict[str, float]]:
        """Simulate battle with enhanced counter-strategy modeling"""
        
        nuln_eff = self.calculate_counter_effectiveness(army_units, enemy)
        enemy_eff = self.calculate_enemy_effectiveness(enemy)
        
        # Strategic deployment bonus based on army composition
        deployment_bonus = self._calculate_deployment_bonus(army_units, enemy)
        nuln_eff *= deployment_bonus
        
        # Tactical variance and battle luck
        tactical_variance = random.uniform(0.80, 1.20)  # ±20%
        battle_luck = random.gauss(1.0, 0.12)  # 12% std dev
        battle_luck = max(0.6, min(1.4, battle_luck))
        
        nuln_final = nuln_eff * tactical_variance * battle_luck
        enemy_final = enemy_eff * random.uniform(0.90, 1.10)
        
        breakdown = {
            "nuln_base": sum(self.database[unit].effectiveness for unit in army_units),
            "nuln_counter_bonus": nuln_eff / sum(self.database[unit].effectiveness for unit in army_units),
            "deployment_bonus": deployment_bonus,
            "tactical_variance": tactical_variance,
            "battle_luck": battle_luck,
            "nuln_final": nuln_final,
            "enemy_final": enemy_final
        }
        
        return nuln_final > enemy_final, breakdown
    
    def _calculate_deployment_bonus(self, army_units: List[str], enemy: Enemy) -> float:
        """Calculate deployment and positioning advantages"""
        bonus = 1.0
        
        # Defensive positioning vs cavalry
        if enemy.primary_threat == "cavalry":
            artillery_count = sum(1 for unit in army_units if self.database[unit].unit_type == "war_machine")
            infantry_count = sum(1 for unit in army_units if self.database[unit].unit_type == "infantry")
            
            # Good screening for artillery
            if artillery_count > 0 and infantry_count >= 2:
                bonus += 0.15
                
            # Mobile artillery vs cavalry
            mobile_artillery = sum(1 for unit in army_units if "Gun Limbers" in unit)
            if mobile_artillery > 0:
                bonus += 0.10
        
        # Concentrated firepower vs elite/monsters
        elif enemy.primary_threat in ["elite_infantry", "monsters"]:
            handgun_count = sum(1 for unit in army_units if "Handgunners" in unit)
            artillery_count = sum(1 for unit in army_units if self.database[unit].unit_type == "war_machine")
            
            # Focused high-strength shooting
            if handgun_count >= 2 or artillery_count >= 2:
                bonus += 0.12
                
            # Combined arms coordination
            if handgun_count > 0 and artillery_count > 0:
                bonus += 0.08
        
        return bonus

def run_counter_strategy_analysis():
    """Run comprehensive counter-strategy analysis"""
    print("🎯 NULN COUNTER-STRATEGY OPTIMIZER")
    print("="*50)
    print("🔍 Analyzing worst enemy matchups...")
    print("⚔️ Developing specialized counter-builds...")
    print("🧪 Testing against problem enemies...")
    print()
    
    optimizer = NulnCounterOptimizer()
    counter_strategies = optimizer.generate_counter_strategies()
    problem_enemies = optimizer.problem_enemies
    
    results = {}
    total_battles = 0
    start_time = time.time()
    
    for strategy_type, army_builds in counter_strategies.items():
        print(f"🛡️ TESTING {strategy_type.upper()} STRATEGIES")
        print("-" * 40)
        
        strategy_results = {}
        
        for i, army_units in enumerate(army_builds, 1):
            army_name = f"{strategy_type} Build #{i}"
            army_points = sum(optimizer.database[unit].points for unit in army_units)
            
            print(f"\n📋 {army_name} ({army_points} pts)")
            
            build_results = {}
            build_wins = 0
            build_total = 0
            
            for enemy_name, enemy in problem_enemies.items():
                battles = 25000  # 25k battles per matchup
                wins = 0
                
                for _ in range(battles):
                    won, breakdown = optimizer.simulate_counter_battle(army_units, enemy)
                    if won:
                        wins += 1
                        build_wins += 1
                    build_total += 1
                    total_battles += 1
                
                win_rate = wins / battles
                build_results[enemy_name] = win_rate
                
                # Color coding
                if win_rate >= 0.65:
                    emoji = "💚"
                elif win_rate >= 0.55:
                    emoji = "💛"
                elif win_rate >= 0.45:
                    emoji = "🟠"
                else:
                    emoji = "🔴"
                
                print(f"   {emoji} vs {enemy_name:.<18} {win_rate:.1%}")
            
            overall_rate = build_wins / build_total
            build_results["overall"] = overall_rate
            strategy_results[army_name] = build_results
            print(f"   📊 Overall vs Problem Enemies: {overall_rate:.1%}")
        
        results[strategy_type] = strategy_results
        print()
    
    elapsed_time = time.time() - start_time
    battles_per_sec = total_battles / elapsed_time
    
    print(f"⚡ Completed {total_battles:,} battles in {elapsed_time:.1f}s")
    print(f"🚀 Processing speed: {battles_per_sec:,.0f} battles/second")
    print()
    
    # Find best counter-builds
    print("🏆 BEST COUNTER-STRATEGY BUILDS")
    print("="*45)
    
    all_builds = []
    for strategy_type, builds in results.items():
        for build_name, build_results in builds.items():
            all_builds.append((build_name, build_results, strategy_type))
    
    # Sort by overall performance vs problem enemies
    all_builds.sort(key=lambda x: x[1]["overall"], reverse=True)
    
    print("\n🥇 TOP PERFORMING COUNTER-BUILDS:")
    for i, (build_name, build_results, strategy_type) in enumerate(all_builds[:5]):
        print(f"\n#{i+1}. {build_name}")
        print(f"    Strategy Type: {strategy_type}")
        print(f"    Overall vs Problems: {build_results['overall']:.1%}")
        print(f"    Specific Results:")
        
        for enemy_name, rate in build_results.items():
            if enemy_name != "overall":
                if rate >= 0.65:
                    emoji = "💚"
                elif rate >= 0.55:
                    emoji = "💛" 
                elif rate >= 0.45:
                    emoji = "🟠"
                else:
                    emoji = "🔴"
                print(f"      {emoji} {enemy_name}: {rate:.1%}")
    
    # Enemy-specific best counters
    print(f"\n🎯 BEST BUILDS PER ENEMY:")
    for enemy_name in problem_enemies.keys():
        best_build = None
        best_rate = 0
        
        for strategy_type, builds in results.items():
            for build_name, build_results in builds.items():
                if build_results[enemy_name] > best_rate:
                    best_rate = build_results[enemy_name]
                    best_build = (build_name, strategy_type)
        
        if best_rate >= 0.65:
            emoji = "💚"
        elif best_rate >= 0.55:
            emoji = "💛"
        elif best_rate >= 0.45:
            emoji = "🟠"
        else:
            emoji = "🔴"
            
        print(f"   {emoji} vs {enemy_name}: {best_build[0]} ({best_rate:.1%})")
    
    print(f"\n📈 STRATEGIC INSIGHTS:")
    print(f"   • Counter-strategies significantly improve difficult matchups")
    print(f"   • Specialized builds outperform generalist approaches")
    print(f"   • Proper deployment and unit synergy crucial vs tough enemies")
    print(f"   • Different threats require different tactical responses")

if __name__ == "__main__":
    run_counter_strategy_analysis() 