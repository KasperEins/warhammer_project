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
    min_size: int
    max_size: int
    effectiveness: float
    special_rules: List[str]
    per_1000_restriction: bool = False
    counter_bonuses: Dict[str, float] = None

@dataclass
class Enemy:
    name: str
    army_type: str
    total_points: int
    primary_threat: str
    faction_bonuses: Dict[str, float]
    weaknesses: List[str]

class AuthenticNulnCounterOptimizer:
    def __init__(self):
        self.database = self._create_authentic_database()
        self.problem_enemies = self._create_problem_enemies()
        
    def _create_authentic_database(self) -> Dict[str, Unit]:
        """Authentic Nuln Army of Infamy database with mandatory units"""
        return {
            # Characters
            "General of the Empire": Unit("General of the Empire", 85, "character", 1, 1, 4.0, ["Leadership"], False, {"cavalry": 0.1, "elite": 0.15}),
            "Captain of the Empire": Unit("Captain of the Empire", 45, "character", 1, 1, 2.5, ["Leadership"], False, {"cavalry": 0.15, "elite": 0.1}),
            "Empire Engineer": Unit("Empire Engineer", 45, "character", 1, 1, 2.0, ["Artillery Support"], False, {"elite": 0.2, "monsters": 0.25}),
            "Battle Standard Bearer": Unit("Battle Standard Bearer", 25, "upgrade", 1, 1, 1.0, ["Banner"], False, {"all": 0.1}),
            "Full Plate Armour": Unit("Full Plate Armour", 8, "upgrade", 1, 1, 0.3, ["Armor"], False, {"elite": 0.1}),
            
            # MANDATORY Core Units (must include at least one)
            "Nuln State Troops (20)": Unit("Nuln State Troops (20)", 100, "core", 10, 30, 3.2, ["Steadfast", "Mandatory"], False, {"cavalry": 0.25}),
            "Nuln State Troops (25)": Unit("Nuln State Troops (25)", 125, "core", 10, 30, 4.0, ["Steadfast", "Mandatory"], False, {"cavalry": 0.3}),
            "Nuln Veteran State Troops (15)": Unit("Nuln Veteran State Troops (15)", 105, "core", 10, 25, 3.0, ["Veteran", "Mandatory"], False, {"elite": 0.2, "cavalry": 0.15}),
            "Nuln Veteran State Troops (20)": Unit("Nuln Veteran State Troops (20)", 140, "core", 10, 25, 4.0, ["Veteran", "Mandatory"], False, {"elite": 0.25, "cavalry": 0.2}),
            
            # MANDATORY Halberdiers (Nuln special requirement)
            "Nuln State Halberdiers (15)": Unit("Nuln State Halberdiers (15)", 90, "core", 10, 30, 2.8, ["Halberd", "Anti-Cavalry", "Mandatory"], False, {"cavalry": 0.4, "monsters": 0.2}),
            "Nuln State Halberdiers (20)": Unit("Nuln State Halberdiers (20)", 120, "core", 10, 30, 3.6, ["Halberd", "Anti-Cavalry", "Mandatory"], False, {"cavalry": 0.45, "monsters": 0.25}),
            
            # Other Core
            "Nuln State Handgunners (10)": Unit("Nuln State Handgunners (10)", 60, "core", 5, 20, 2.8, ["Handgun Drill"], False, {"elite": 0.35, "monsters": 0.25}),
            "Nuln State Handgunners (15)": Unit("Nuln State Handgunners (15)", 90, "core", 5, 20, 4.0, ["Handgun Drill"], False, {"elite": 0.4, "monsters": 0.3}),
            "Nuln Veteran Outriders (8)": Unit("Nuln Veteran Outriders (8)", 152, "core", 5, 10, 4.2, ["Fast Cavalry", "No Ponderous"], True, {"cavalry": 0.4}),
            
            # Special Units
            "Empire Greatswords (12)": Unit("Empire Greatswords (12)", 144, "special", 5, 20, 4.8, ["Elite", "Great Weapons"], False, {"elite": 0.25, "monsters": 0.35}),
            "Empire Greatswords (15)": Unit("Empire Greatswords (15)", 180, "special", 5, 20, 5.5, ["Elite", "Great Weapons"], False, {"elite": 0.3, "monsters": 0.4}),
            "Empire Knights (5)": Unit("Empire Knights (5)", 100, "special", 3, 12, 4.5, ["Heavy Cavalry"], False, {"cavalry": 0.4, "elite": 0.15}),
            "Empire Knights (8)": Unit("Empire Knights (8)", 160, "special", 3, 12, 6.0, ["Heavy Cavalry"], False, {"cavalry": 0.5, "elite": 0.2}),
            "Empire War Wagon": Unit("Empire War Wagon", 90, "special", 1, 3, 3.5, ["Mobile Platform"], False, {"cavalry": 0.3, "elite": 0.2}),
            "Great Cannon": Unit("Great Cannon", 125, "special", 1, 4, 4.0, ["Artillery"], False, {"elite": 0.3, "monsters": 0.5}),
            "Great Cannon with Gun Limbers": Unit("Great Cannon with Gun Limbers", 135, "special", 1, 4, 4.5, ["Artillery", "Vanguard"], False, {"elite": 0.35, "monsters": 0.55}),
            "Mortar": Unit("Mortar", 90, "special", 1, 2, 3.5, ["Artillery"], False, {"elite": 0.4, "cavalry": 0.3}),
            
            # Rare Units  
            "Road Wardens (10)": Unit("Road Wardens (10)", 180, "rare", 5, 15, 5.0, ["Elite Cavalry"], True, {"cavalry": 0.6, "elite": 0.3}),
            "Helblaster Volley Gun": Unit("Helblaster Volley Gun", 120, "rare", 1, 2, 4.2, ["Multi-shot"], False, {"cavalry": 0.6, "elite": 0.2}),
            "Helblaster with Gun Limbers": Unit("Helblaster with Gun Limbers", 135, "rare", 1, 2, 4.8, ["Multi-shot", "Vanguard"], False, {"cavalry": 0.7, "elite": 0.25}),
            "Steam Tank": Unit("Steam Tank", 285, "rare", 1, 1, 8.0, ["Monster", "Terror"], True, {"cavalry": 0.8, "elite": 0.4, "monsters": 0.6}),
            
            # Mercenaries
            "Imperial Dwarfs (12)": Unit("Imperial Dwarfs (12)", 108, "mercenary", 10, 25, 4.0, ["Resilient"], False, {"elite": 0.2, "monsters": 0.25}),
            "Imperial Dwarfs (15)": Unit("Imperial Dwarfs (15)", 135, "mercenary", 10, 25, 4.8, ["Resilient"], False, {"elite": 0.25, "monsters": 0.3}),
            "Imperial Ogres (3)": Unit("Imperial Ogres (3)", 114, "mercenary", 3, 6, 4.5, ["Monstrous Infantry"], False, {"elite": 0.3, "cavalry": 0.2}),
        }
    
    def _create_problem_enemies(self) -> Dict[str, Enemy]:
        """Our most problematic enemies from previous analysis"""
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
    
    def is_valid_army(self, army: List[str]) -> Tuple[bool, str]:
        """Validate army against Nuln Army of Infamy restrictions"""
        total_points = sum(self.database[unit].points for unit in army)
        if total_points > 750:
            return False, f"Exceeds 750 points ({total_points})"
        
        # Check category limits (188/263/225/188)
        character_points = sum(self.database[unit].points for unit in army if self.database[unit].category == "character" or self.database[unit].category == "upgrade")
        core_points = sum(self.database[unit].points for unit in army if self.database[unit].category == "core")
        special_points = sum(self.database[unit].points for unit in army if self.database[unit].category == "special")
        rare_points = sum(self.database[unit].points for unit in army if self.database[unit].category == "rare")
        
        if character_points > 188:
            return False, f"Character limit exceeded ({character_points}/188)"
        if core_points > 263:
            return False, f"Core limit exceeded ({core_points}/263)"
        if special_points > 225:
            return False, f"Special limit exceeded ({special_points}/225)"
        if rare_points > 188:
            return False, f"Rare limit exceeded ({rare_points}/188)"
        
        # Check mandatory requirements
        has_mandatory_troops = any("Mandatory" in self.database[unit].special_rules for unit in army)
        if not has_mandatory_troops:
            return False, "Must include at least one regiment of Nuln State/Veteran Troops or Halberdiers"
        
        # Check 0-X per 1000 points (max 2 in 750pt games)
        per_1000_units = [unit for unit in army if self.database[unit].per_1000_restriction]
        if len(per_1000_units) > 2:
            return False, f"Too many 0-X per 1000 units ({len(per_1000_units)}/2)"
        
        # Check Battle Standard Bearer restriction
        if "Battle Standard Bearer" in army:
            has_captain_or_engineer = any(unit in army for unit in ["Captain of the Empire", "Empire Engineer"])
            if not has_captain_or_engineer:
                return False, "Battle Standard Bearer only available to Captain or Engineer"
        
        return True, "Valid"
    
    def generate_authentic_counter_strategies(self) -> Dict[str, List[List[str]]]:
        """Generate authentic counter-strategies using mandatory units"""
        strategies = {}
        
        # Anti-Cavalry builds (vs Bretonnians) - Emphasize halberdiers
        strategies["Anti-Cavalry"] = [
            # Halberdier Wall
            ["General of the Empire", "Full Plate Armour", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Halberdiers (20)", "Nuln State Handgunners (10)", 
             "Helblaster Volley Gun", "Imperial Dwarfs (12)"],
            
            # Mobile Anti-Cavalry  
            ["Captain of the Empire", "Battle Standard Bearer",
             "Nuln State Halberdiers (15)", "Nuln State Handgunners (15)",
             "Empire Knights (5)", "Great Cannon"],
            
            # Defensive Castle
            ["General of the Empire", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (20)", "Nuln State Halberdiers (15)", 
             "Great Cannon with Gun Limbers"]
        ]
        
        # Anti-Elite builds (vs Chaos/High Elves) - Artillery focus
        strategies["Anti-Elite"] = [
            # Artillery Supremacy
            ["General of the Empire", "Full Plate Armour", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (20)", "Nuln State Handgunners (10)",
             "Great Cannon", "Mortar"],
            
            # Elite vs Elite  
            ["General of the Empire", "Captain of the Empire", "Battle Standard Bearer",
             "Nuln Veteran State Troops (15)", "Empire Greatswords (12)",
             "Great Cannon", "Imperial Dwarfs (12)"],
            
            # Mass Shooting
            ["General of the Empire", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (20)", "Nuln State Handgunners (15)", 
             "Helblaster Volley Gun"]
        ]
        
        # Anti-Monster builds (vs Lizardmen) - High strength focus
        strategies["Anti-Monster"] = [
            # Double Artillery
            ["General of the Empire", "Full Plate Armour", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (20)", "Great Cannon", "Mortar"],
            
            # Steam Tank Power (uses 2 per-1000 slots)
            ["General of the Empire", "Empire Engineer",
             "Nuln State Halberdiers (15)", "Steam Tank"],
            
            # Combined Arms Anti-Monster
            ["General of the Empire", "Captain of the Empire", "Battle Standard Bearer",
             "Nuln Veteran State Troops (15)", "Empire Greatswords (12)",
             "Great Cannon", "Imperial Dwarfs (12)"]
        ]
        
        return strategies
    
    def calculate_counter_effectiveness(self, army_units: List[str], enemy: Enemy) -> float:
        """Calculate effectiveness with authentic faction bonuses and counters"""
        base_effectiveness = sum(self.database[unit].effectiveness for unit in army_units)
        
        # Apply counter-strategy bonuses
        counter_bonus = 1.0
        for unit_name in army_units:
            unit = self.database[unit_name]
            if unit.counter_bonuses:
                if enemy.primary_threat in unit.counter_bonuses:
                    counter_bonus += unit.counter_bonuses[enemy.primary_threat]
                if "all" in unit.counter_bonuses:
                    counter_bonus += unit.counter_bonuses["all"]
        
        # Authentic Nuln faction bonuses
        faction_multiplier = 1.0
        
        # Handgun Drill bonus
        handgun_units = sum(1 for unit in army_units if "Handgun Drill" in self.database[unit].special_rules)
        faction_multiplier += handgun_units * 0.08
        
        # Artillery Engineer synergy  
        engineers = sum(1 for unit in army_units if "Engineer" in unit)
        artillery = sum(1 for unit in army_units if "Artillery" in self.database[unit].special_rules)
        if engineers > 0 and artillery > 0:
            faction_multiplier += min(engineers, artillery) * 0.12
        
        # Gun Limbers mobility
        gun_limbers = sum(1 for unit in army_units if "Vanguard" in self.database[unit].special_rules)
        faction_multiplier += gun_limbers * 0.06
        
        # Big Guns Know No Fear
        if artillery >= 2:
            faction_multiplier += 0.10
        
        # Anti-cavalry specialist bonus
        halberdiers = sum(1 for unit in army_units if "Anti-Cavalry" in self.database[unit].special_rules)
        if enemy.primary_threat == "cavalry" and halberdiers > 0:
            faction_multiplier += halberdiers * 0.15
        
        return base_effectiveness * counter_bonus * faction_multiplier
    
    def calculate_enemy_effectiveness(self, enemy: Enemy) -> float:
        """Calculate enemy army effectiveness"""
        base_effectiveness = (enemy.total_points / 750.0) * 20.0
        
        total_bonus = 1.0
        for bonus in enemy.faction_bonuses.values():
            total_bonus *= bonus
            
        return base_effectiveness * total_bonus
    
    def simulate_authentic_counter_battle(self, army_units: List[str], enemy: Enemy) -> Tuple[bool, Dict[str, float]]:
        """Simulate battle with authentic Nuln rules"""
        
        nuln_eff = self.calculate_counter_effectiveness(army_units, enemy)
        enemy_eff = self.calculate_enemy_effectiveness(enemy)
        
        # Tactical variance and battle luck
        tactical_variance = random.uniform(0.80, 1.20)
        battle_luck = random.gauss(1.0, 0.12)
        battle_luck = max(0.6, min(1.4, battle_luck))
        
        nuln_final = nuln_eff * tactical_variance * battle_luck
        enemy_final = enemy_eff * random.uniform(0.90, 1.10)
        
        breakdown = {
            "nuln_base": sum(self.database[unit].effectiveness for unit in army_units),
            "nuln_counter_bonus": nuln_eff / sum(self.database[unit].effectiveness for unit in army_units) if sum(self.database[unit].effectiveness for unit in army_units) > 0 else 1.0,
            "tactical_variance": tactical_variance,
            "battle_luck": battle_luck,
            "nuln_final": nuln_final,
            "enemy_final": enemy_final
        }
        
        return nuln_final > enemy_final, breakdown

def run_authentic_counter_analysis():
    """Run authentic Nuln counter-strategy analysis"""
    print("🏛️ AUTHENTIC NULN ARMY OF INFAMY COUNTER-OPTIMIZER")
    print("="*55)
    print("⚔️ Using mandatory halberdiers and authentic restrictions")
    print("📋 Tournament limits: 188/263/225/188 points")
    print("🎯 Max 2 per-1000 units allowed in 750pt games")
    print("🔧 Battle Standard Bearer restricted to Captain/Engineer")
    print()
    
    optimizer = AuthenticNulnCounterOptimizer()
    counter_strategies = optimizer.generate_authentic_counter_strategies()
    problem_enemies = optimizer.problem_enemies
    
    results = {}
    total_battles = 0
    start_time = time.time()
    
    for strategy_type, army_builds in counter_strategies.items():
        print(f"🛡️ TESTING {strategy_type.upper()} STRATEGIES")
        print("-" * 40)
        
        strategy_results = {}
        
        for i, army_units in enumerate(army_builds, 1):
            # Validate army first
            is_valid, reason = optimizer.is_valid_army(army_units)
            if not is_valid:
                print(f"❌ {strategy_type} Build #{i}: INVALID - {reason}")
                continue
            
            army_name = f"{strategy_type} Build #{i}"
            army_points = sum(optimizer.database[unit].points for unit in army_units)
            
            print(f"\n📋 {army_name} ({army_points} pts) ✅")
            
            # Show mandatory units
            mandatory_units = [unit for unit in army_units if "Mandatory" in optimizer.database[unit].special_rules]
            halberdiers = [unit for unit in army_units if "Halberd" in optimizer.database[unit].special_rules]
            if mandatory_units or halberdiers:
                print(f"   Mandatory: {', '.join(mandatory_units + halberdiers)}")
            
            build_results = {}
            build_wins = 0
            build_total = 0
            
            for enemy_name, enemy in problem_enemies.items():
                battles = 25000
                wins = 0
                
                for _ in range(battles):
                    won, breakdown = optimizer.simulate_authentic_counter_battle(army_units, enemy)
                    if won:
                        wins += 1
                        build_wins += 1
                    build_total += 1
                    total_battles += 1
                
                win_rate = wins / battles
                build_results[enemy_name] = win_rate
                
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
    
    # Find best authentic builds
    print("🏆 BEST AUTHENTIC NULN COUNTER-BUILDS")
    print("="*40)
    
    all_builds = []
    for strategy_type, builds in results.items():
        for build_name, build_results in builds.items():
            all_builds.append((build_name, build_results, strategy_type))
    
    all_builds.sort(key=lambda x: x[1]["overall"], reverse=True)
    
    print("\n🥇 TOP AUTHENTIC BUILDS:")
    for i, (build_name, build_results, strategy_type) in enumerate(all_builds[:3]):
        print(f"\n#{i+1}. {build_name}")
        print(f"    Strategy Type: {strategy_type}")
        print(f"    Overall vs Problems: {build_results['overall']:.1%}")
        print(f"    ✅ Follows all Nuln Army of Infamy restrictions")
        
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
    
    print(f"\n📈 AUTHENTIC ARMY INSIGHTS:")
    print(f"   • Mandatory halberdiers excel vs cavalry threats")
    print(f"   • Authentic restrictions create balanced armies")
    print(f"   • Engineer + artillery synergy remains powerful")
    print(f"   • Must include Nuln State/Veteran Troops or Halberdiers")

if __name__ == "__main__":
    run_authentic_counter_analysis() 