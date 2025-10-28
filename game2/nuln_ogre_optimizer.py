#!/usr/bin/env python3

import random
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict

@dataclass
class Unit:
    name: str
    points: int
    category: str
    effectiveness: float
    special_rules: List[str]
    per_1000_restriction: bool = False
    counter_bonuses: Dict[str, float] = None

@dataclass
class Enemy:
    name: str
    points: int
    primary_threat: str
    faction_bonuses: Dict[str, float]
    key_weakness: str

class NulnOgreOptimizer:
    def __init__(self):
        self.database = self._create_ogre_database()
        self.enemies = self._create_enemy_database()
        
    def _create_ogre_database(self) -> Dict[str, Unit]:
        """Updated database with Imperial Ogres and Light Cannons"""
        return {
            # Characters (max 188pts)
            "General + Full Plate": Unit("General + Full Plate", 93, "character", 4.3, 
                                       ["Leadership", "Armor"], False, {"all": 0.15}),
            "Captain + BSB": Unit("Captain + BSB", 70, "character", 3.5,
                                ["Leadership", "Banner"], False, {"all": 0.12}),
            "Engineer + BSB": Unit("Engineer + BSB", 70, "character", 3.0,
                                 ["Artillery Support", "Banner"], False, {"elite": 0.2, "monsters": 0.25}),
            
            # Core Units (max 263pts)
            "Nuln State Troops (25)": Unit("Nuln State Troops (25)", 125, "core", 4.0,
                                          ["Mandatory", "Steadfast"], False, {"cavalry": 0.3}),
            "Nuln State Troops (20)": Unit("Nuln State Troops (20)", 100, "core", 3.2,
                                          ["Mandatory", "Steadfast"], False, {"cavalry": 0.25}),
            "Nuln Halberdiers (20)": Unit("Nuln Halberdiers (20)", 120, "core", 3.6,
                                         ["Mandatory", "Anti-Cavalry"], False, {"cavalry": 0.45, "monsters": 0.25}),
            "Nuln Halberdiers (15)": Unit("Nuln Halberdiers (15)", 90, "core", 2.8,
                                         ["Mandatory", "Anti-Cavalry"], False, {"cavalry": 0.4, "monsters": 0.2}),
            "Nuln Handgunners (15)": Unit("Nuln Handgunners (15)", 90, "core", 4.0,
                                         ["Handgun Drill"], False, {"elite": 0.4, "monsters": 0.3}),
            "Nuln Handgunners (10)": Unit("Nuln Handgunners (10)", 60, "core", 2.8,
                                         ["Handgun Drill"], False, {"elite": 0.35, "monsters": 0.25}),
            
            # Special Units (max 225pts)  
            "Great Cannon + Limbers": Unit("Great Cannon + Limbers", 135, "special", 4.5,
                                         ["Artillery", "Vanguard"], False, {"elite": 0.35, "monsters": 0.55}),
            "Great Cannon": Unit("Great Cannon", 125, "special", 4.0,
                               ["Artillery"], False, {"elite": 0.3, "monsters": 0.5}),
            "Light Cannon": Unit("Light Cannon", 85, "special", 3.2,
                               ["Artillery", "Mobile"], False, {"cavalry": 0.4, "elite": 0.25}),
            "Light Cannon + Limbers": Unit("Light Cannon + Limbers", 95, "special", 3.7,
                                         ["Artillery", "Mobile", "Vanguard"], False, {"cavalry": 0.45, "elite": 0.3}),
            "Mortar": Unit("Mortar", 90, "special", 3.5,
                         ["Artillery"], False, {"elite": 0.4, "cavalry": 0.3}),
            "Empire Greatswords (15)": Unit("Empire Greatswords (15)", 180, "special", 5.5,
                                          ["Elite"], False, {"elite": 0.3, "monsters": 0.4}),
            "Empire Greatswords (12)": Unit("Empire Greatswords (12)", 144, "special", 4.8,
                                          ["Elite"], False, {"elite": 0.25, "monsters": 0.35}),
            
            # Rare Units (max 188pts)
            "Helblaster + Limbers": Unit("Helblaster + Limbers", 135, "rare", 4.8,
                                       ["Multi-shot", "Vanguard"], False, {"cavalry": 0.7, "elite": 0.25}),
            "Helblaster Volley Gun": Unit("Helblaster Volley Gun", 120, "rare", 4.2,
                                        ["Multi-shot"], False, {"cavalry": 0.6, "elite": 0.2}),
            "Steam Tank": Unit("Steam Tank", 285, "rare", 8.0,
                             ["Monster", "Terror"], True, {"cavalry": 0.8, "elite": 0.4, "monsters": 0.6}),
            
            # Mercenaries - NEW OGRES!
            "Imperial Ogres (3)": Unit("Imperial Ogres (3)", 114, "mercenary", 4.5,
                                     ["Monstrous Infantry", "Fear"], False, {"elite": 0.3, "cavalry": 0.25, "monsters": 0.2}),
            "Imperial Ogres (4)": Unit("Imperial Ogres (4)", 152, "mercenary", 5.8,
                                     ["Monstrous Infantry", "Fear"], False, {"elite": 0.35, "cavalry": 0.3, "monsters": 0.25}),
            "Imperial Ogres (6)": Unit("Imperial Ogres (6)", 228, "mercenary", 8.5,
                                     ["Monstrous Infantry", "Fear"], False, {"elite": 0.4, "cavalry": 0.35, "monsters": 0.3}),
        }
    
    def _create_enemy_database(self) -> Dict[str, Enemy]:
        """Enemy database with key weaknesses identified"""
        return {
            "Bretonnian Knights": Enemy("Bretonnian Knights", 750, "cavalry",
                                      {"chivalry": 1.20, "charge": 1.25}, "volume_shooting"),
            "Chaos Warriors": Enemy("Chaos Warriors", 760, "elite_infantry", 
                                  {"favor": 1.30, "armor": 1.20}, "concentrated_artillery"),
            "High Elf Elite": Enemy("High Elf Elite", 730, "elite_infantry",
                                  {"martial_prowess": 1.25, "reflexes": 1.15}, "overwhelming_firepower"),
            "Lizardmen Temple": Enemy("Lizardmen Temple", 780, "monsters",
                                    {"cold_blooded": 1.15, "scales": 1.10, "power": 1.20}, "high_strength_focus"),
        }
    
    def generate_ogre_builds(self) -> Dict[str, List[List[str]]]:
        """Generate optimized builds using Imperial Ogres and Light Cannons"""
        builds = {}
        
        # Ogre-Enhanced Anti-Cavalry
        builds["Ogre Anti-Cavalry"] = [
            # Ogre + Helblaster Supremacy
            ["General + Full Plate", "Engineer + BSB", "Nuln Halberdiers (20)", 
             "Nuln Handgunners (15)", "Helblaster + Limbers", "Imperial Ogres (3)"],
            
            # Mobile Light Artillery  
            ["General + Full Plate", "Engineer + BSB", "Nuln Halberdiers (20)",
             "Light Cannon + Limbers", "Light Cannon", "Imperial Ogres (4)"],
             
            # Ogre Charge Support
            ["Captain + BSB", "Nuln Halberdiers (20)", "Nuln Handgunners (15)",
             "Light Cannon + Limbers", "Imperial Ogres (3)", "Helblaster Volley Gun"]
        ]
        
        # Ogre-Enhanced Anti-Elite  
        builds["Ogre Anti-Elite"] = [
            # Maximum Ogre Firepower
            ["General + Full Plate", "Engineer + BSB", "Nuln State Troops (20)",
             "Nuln Handgunners (15)", "Great Cannon + Limbers", "Light Cannon", "Imperial Ogres (3)"],
             
            # Elite vs Elite with Ogres
            ["General + Full Plate", "Captain + BSB", "Nuln Handgunners (15)",
             "Empire Greatswords (12)", "Light Cannon + Limbers", "Imperial Ogres (3)"],
             
            # Artillery + Ogre Support  
            ["General + Full Plate", "Engineer + BSB", "Nuln State Troops (20)",
             "Light Cannon + Limbers", "Light Cannon", "Imperial Ogres (4)"]
        ]
        
        # Ogre-Enhanced Anti-Monster
        builds["Ogre Anti-Monster"] = [
            # Heavy Artillery + Ogres
            ["General + Full Plate", "Engineer + BSB", "Nuln State Troops (20)",
             "Great Cannon + Limbers", "Great Cannon", "Imperial Ogres (3)"],
             
            # Monster vs Monster
            ["General + Full Plate", "Engineer + BSB", "Nuln Halberdiers (15)",
             "Light Cannon + Limbers", "Imperial Ogres (6)"],
             
            # Steam Tank + Ogres (Ultimate Power)
            ["Engineer + BSB", "Nuln State Troops (20)", "Imperial Ogres (4)", "Steam Tank"]
        ]
        
        # NEW: Ogre-Enhanced Anti-Chaos
        builds["Ogre Anti-Chaos"] = [
            # Triple Artillery + Ogres
            ["General + Full Plate", "Engineer + BSB", "Nuln State Troops (20)",
             "Great Cannon", "Light Cannon + Limbers", "Light Cannon", "Imperial Ogres (3)"],
             
            # Elite Breakthrough + Ogres  
            ["General + Full Plate", "Captain + BSB", "Nuln Handgunners (15)",
             "Empire Greatswords (12)", "Great Cannon + Limbers", "Imperial Ogres (3)"],
             
            # Maximum Ogre Assault
            ["General + Full Plate", "Engineer + BSB", "Nuln Handgunners (10)",
             "Light Cannon", "Imperial Ogres (6)"]
        ]
        
        return builds
    
    def is_valid_army(self, army: List[str]) -> Tuple[bool, str]:
        """Validate army construction"""
        total_points = sum(self.database[unit].points for unit in army)
        if total_points > 750:
            return False, f"Exceeds 750 points ({total_points})"
        
        # Category limits
        char_pts = sum(self.database[unit].points for unit in army if self.database[unit].category == "character")
        core_pts = sum(self.database[unit].points for unit in army if self.database[unit].category == "core") 
        special_pts = sum(self.database[unit].points for unit in army if self.database[unit].category == "special")
        rare_pts = sum(self.database[unit].points for unit in army if self.database[unit].category == "rare")
        
        if char_pts > 188: return False, f"Character limit ({char_pts}/188)"
        if core_pts > 263: return False, f"Core limit ({core_pts}/263)"
        if special_pts > 225: return False, f"Special limit ({special_pts}/225)"
        if rare_pts > 188: return False, f"Rare limit ({rare_pts}/188)"
        
        # Mandatory units
        has_mandatory = any("Mandatory" in self.database[unit].special_rules for unit in army)
        if not has_mandatory:
            return False, "Must include mandatory troops"
            
        # Per-1000 limit
        per_1000 = [unit for unit in army if self.database[unit].per_1000_restriction]
        if len(per_1000) > 2:
            return False, f"Too many 0-X per 1000 units ({len(per_1000)}/2)"
        
        return True, "Valid"
    
    def calculate_army_effectiveness(self, army_units: List[str], enemy: Enemy) -> float:
        """Calculate optimized army effectiveness with ogre bonuses"""
        base_eff = sum(self.database[unit].effectiveness for unit in army_units)
        
        # Counter bonuses
        counter_bonus = 1.0
        for unit_name in army_units:
            unit = self.database[unit_name]
            if unit.counter_bonuses:
                if enemy.primary_threat in unit.counter_bonuses:
                    counter_bonus += unit.counter_bonuses[enemy.primary_threat]
                if "all" in unit.counter_bonuses:
                    counter_bonus += unit.counter_bonuses["all"]
        
        # Enhanced Nuln faction synergies
        faction_mult = 1.0
        
        # Artillery synergies (enhanced)
        engineers = sum(1 for unit in army_units if "Engineer" in unit)
        artillery = sum(1 for unit in army_units if "Artillery" in self.database[unit].special_rules)
        if engineers > 0 and artillery > 0:
            faction_mult += min(engineers, artillery) * 0.18  # Boosted from analysis
        
        # Light cannon mobility bonus
        light_cannons = sum(1 for unit in army_units if "Mobile" in self.database[unit].special_rules)
        faction_mult += light_cannons * 0.08  # Mobile artillery advantage
        
        # Handgun synergies
        handgunners = sum(1 for unit in army_units if "Handgun Drill" in self.database[unit].special_rules)
        faction_mult += handgunners * 0.10
        
        # Multiple artillery bonus
        if artillery >= 2:
            faction_mult += 0.15
        if artillery >= 3:
            faction_mult += 0.12  # Triple artillery mastery!
        
        # NEW: Ogre synergies!
        ogres = sum(1 for unit in army_units if "Monstrous Infantry" in self.database[unit].special_rules)
        if ogres > 0:
            # Ogres provide fear, breakthrough power, and artillery protection
            faction_mult += ogres * 0.12  # Fear factor
            if artillery > 0:
                faction_mult += ogres * 0.08  # Ogres protect artillery
            # Ogre charge bonus vs elite enemies
            if enemy.primary_threat == "elite_infantry":
                faction_mult += ogres * 0.10  # Ogres crush elite infantry
        
        # Banner coordination
        banners = sum(1 for unit in army_units if "Banner" in self.database[unit].special_rules)
        faction_mult += banners * 0.08
        
        return base_eff * counter_bonus * faction_mult
    
    def calculate_enemy_effectiveness(self, enemy: Enemy) -> float:
        """Calculate enemy effectiveness"""
        base_eff = (enemy.points / 750.0) * 22.0
        
        faction_mult = 1.0
        for bonus in enemy.faction_bonuses.values():
            faction_mult *= bonus
            
        return base_eff * faction_mult
    
    def simulate_ogre_battle(self, army_units: List[str], enemy: Enemy) -> Tuple[bool, Dict[str, float]]:
        """Enhanced battle simulation with ogre advantages"""
        nuln_eff = self.calculate_army_effectiveness(army_units, enemy)
        enemy_eff = self.calculate_enemy_effectiveness(enemy)
        
        # Enhanced tactical variance for ogre armies
        base_variance = 0.85
        if any("Monstrous Infantry" in self.database[unit].special_rules for unit in army_units):
            base_variance += 0.06  # Ogres more reliable than humans
        if any("Elite" in self.database[unit].special_rules for unit in army_units):
            base_variance += 0.04
        if any("Terror" in self.database[unit].special_rules for unit in army_units):
            base_variance += 0.03
            
        tactical_var = random.uniform(base_variance, 1.15)
        battle_luck = random.gauss(1.0, 0.08)
        battle_luck = max(0.75, min(1.25, battle_luck))
        
        nuln_final = nuln_eff * tactical_var * battle_luck
        enemy_final = enemy_eff * random.uniform(0.92, 1.08)
        
        margin = (nuln_final - enemy_final) / enemy_final if enemy_final > 0 else 0
        
        return nuln_final > enemy_final, {
            "nuln_eff": nuln_eff,
            "enemy_eff": enemy_eff, 
            "margin": margin,
            "nuln_final": nuln_final,
            "enemy_final": enemy_final
        }

def run_ogre_analysis():
    """Run ultimate strategic analysis with Imperial Ogres"""
    print("💪 NULN IMPERIAL OGRE OPTIMIZER")
    print("="*45)
    print("🎯 Featuring Imperial Ogres with Light Cannons")
    print("🏆 Enhanced builds for tournament dominance")
    print("💥 Monstrous infantry + artillery synergies")
    print("🚀 Running 500,000 battles per matchup")
    print()
    
    optimizer = NulnOgreOptimizer()
    ogre_builds = optimizer.generate_ogre_builds()
    enemies = optimizer.enemies
    
    results = {}
    total_battles = 0
    start_time = time.time()
    
    for strategy_type, army_builds in ogre_builds.items():
        print(f"💪 {strategy_type.upper()}")
        print("-" * 40)
        
        strategy_results = {}
        
        for i, army_units in enumerate(army_builds, 1):
            is_valid, reason = optimizer.is_valid_army(army_units)
            if not is_valid:
                print(f"❌ Build #{i}: INVALID - {reason}")
                continue
                
            army_name = f"{strategy_type} Build #{i}"
            army_points = sum(optimizer.database[unit].points for unit in army_units)
            
            print(f"\n📋 {army_name} ({army_points} pts)")
            
            # Show ogres and artillery
            ogres = [unit for unit in army_units if "Monstrous Infantry" in optimizer.database[unit].special_rules]
            artillery = [unit for unit in army_units if "Artillery" in optimizer.database[unit].special_rules]
            if ogres:
                print(f"   💪 Ogres: {', '.join(ogres)}")
            if artillery:
                print(f"   🎯 Artillery: {', '.join(artillery)}")
            
            build_results = {}
            build_total_wins = 0
            build_total_battles = 0
            
            for enemy_name, enemy in enemies.items():
                battles = 500000  # Massive simulation
                wins = 0
                total_margin = 0
                
                for _ in range(battles):
                    won, breakdown = optimizer.simulate_ogre_battle(army_units, enemy)
                    if won:
                        wins += 1
                        total_margin += breakdown["margin"]
                    total_battles += 1
                    build_total_battles += 1
                
                win_rate = wins / battles
                avg_margin = total_margin / max(wins, 1)
                build_results[enemy_name] = {"win_rate": win_rate, "margin": avg_margin}
                
                if win_rate >= 0.5:
                    build_total_wins += wins
                
                # Enhanced result display
                if win_rate >= 0.80:
                    emoji = "🏆"
                elif win_rate >= 0.70:
                    emoji = "💚"
                elif win_rate >= 0.60:
                    emoji = "💛"
                elif win_rate >= 0.50:
                    emoji = "🟠"
                else:
                    emoji = "🔴"
                
                print(f"   {emoji} vs {enemy_name:.<20} {win_rate:.1%} (±{avg_margin:+.2f})")
            
            overall_rate = build_total_wins / build_total_battles if build_total_battles > 0 else 0
            strategy_results[army_name] = build_results
            print(f"   📊 Overall vs All Enemies: {overall_rate:.1%}")
        
        results[strategy_type] = strategy_results
        print()
    
    elapsed_time = time.time() - start_time
    battles_per_sec = total_battles / elapsed_time
    
    print(f"⚡ OGRE ANALYSIS COMPLETE")
    print(f"   Total Battles: {total_battles:,}")
    print(f"   Analysis Time: {elapsed_time:.1f}s") 
    print(f"   Battle Speed: {battles_per_sec:,.0f} battles/second")
    
    # Find ogre champions
    print(f"\n🏆 OGRE CHAMPIONS")
    print("="*30)
    
    all_builds = []
    for strategy_type, builds in results.items():
        for build_name, build_results in builds.items():
            total_score = sum(r["win_rate"] for r in build_results.values()) / len(build_results)
            chaos_score = build_results.get("Chaos Warriors", {}).get("win_rate", 0)
            all_builds.append((build_name, build_results, total_score, chaos_score, strategy_type))
    
    all_builds.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    for rank, (build_name, build_results, total_score, chaos_score, strategy_type) in enumerate(all_builds[:5], 1):
        print(f"\n💪 #{rank}. {build_name}")
        print(f"    Strategy: {strategy_type}")
        print(f"    Overall Score: {total_score:.1%}")
        print(f"    🔥 vs Chaos Warriors: {chaos_score:.1%}")
        print(f"    Performance:")
        
        for enemy_name, results_data in build_results.items():
            wr = results_data["win_rate"]
            margin = results_data["margin"]
            if wr >= 0.80:
                emoji = "🏆"
            elif wr >= 0.70:
                emoji = "💚"
            elif wr >= 0.60:
                emoji = "💛"
            elif wr >= 0.50:
                emoji = "🟠"
            else:
                emoji = "🔴"
            print(f"      {emoji} {enemy_name}: {wr:.1%} (±{margin:+.2f})")
    
    # Ogre strategic insights
    print(f"\n📈 OGRE STRATEGIC INSIGHTS:")
    print(f"   • Ogres provide fear and breakthrough power")
    print(f"   • Light cannons offer mobile artillery support")
    print(f"   • Ogres protect artillery from cavalry charges")
    print(f"   • Triple artillery + ogres dominates elite armies")
    print(f"   • Monstrous infantry excels vs elite infantry")

if __name__ == "__main__":
    run_ogre_analysis() 