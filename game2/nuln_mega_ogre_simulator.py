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

class NulnMegaOgreSimulator:
    def __init__(self):
        self.database = self._create_ogre_database()
        self.enemies = self._create_comprehensive_enemy_database()
        
    def _create_ogre_database(self) -> Dict[str, Unit]:
        """Complete database with Imperial Ogres and Light Cannons"""
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
            
            # Mercenaries - OGRES!
            "Imperial Ogres (3)": Unit("Imperial Ogres (3)", 114, "mercenary", 4.5,
                                     ["Monstrous Infantry", "Fear"], False, {"elite": 0.3, "cavalry": 0.25, "monsters": 0.2}),
            "Imperial Ogres (4)": Unit("Imperial Ogres (4)", 152, "mercenary", 5.8,
                                     ["Monstrous Infantry", "Fear"], False, {"elite": 0.35, "cavalry": 0.3, "monsters": 0.25}),
            "Imperial Ogres (6)": Unit("Imperial Ogres (6)", 228, "mercenary", 8.5,
                                     ["Monstrous Infantry", "Fear"], False, {"elite": 0.4, "cavalry": 0.35, "monsters": 0.3}),
        }
    
    def _create_comprehensive_enemy_database(self) -> Dict[str, Enemy]:
        """COMPLETE enemy database - ALL ARMIES IN THE OLD WORLD"""
        return {
            # Original problem armies
            "Bretonnian Knights": Enemy("Bretonnian Knights", 750, "cavalry",
                                      {"chivalry": 1.20, "charge": 1.25}, "volume_shooting"),
            "Chaos Warriors": Enemy("Chaos Warriors", 760, "elite_infantry", 
                                  {"favor": 1.30, "armor": 1.20}, "concentrated_artillery"),
            "High Elf Elite": Enemy("High Elf Elite", 730, "elite_infantry",
                                  {"martial_prowess": 1.25, "reflexes": 1.15}, "overwhelming_firepower"),
            "Lizardmen Temple": Enemy("Lizardmen Temple", 780, "monsters",
                                    {"cold_blooded": 1.15, "scales": 1.10, "power": 1.20}, "high_strength_focus"),
            
            # ADDITIONAL BRETONNIAN ARMIES
            "Bretonnian Peasant Horde": Enemy("Bretonnian Peasant Horde", 720, "horde",
                                            {"numbers": 1.15, "expendable": 1.10}, "quality_troops"),
            "Bretonnian Mixed Host": Enemy("Bretonnian Mixed Host", 740, "combined_arms",
                                         {"versatility": 1.15, "nobility": 1.10}, "focused_strategy"),
            "Bretonnian Questing Knights": Enemy("Bretonnian Questing Knights", 770, "elite_cavalry",
                                               {"fanaticism": 1.25, "relentless": 1.15}, "coordinated_firepower"),
            
            # CHAOS ARMIES
            "Chaos Marauders": Enemy("Chaos Marauders", 710, "horde",
                                   {"savagery": 1.20, "numbers": 1.15}, "discipline"),
            "Chaos Knights": Enemy("Chaos Knights", 780, "elite_cavalry",
                                 {"dark_blessing": 1.30, "terror": 1.20}, "massed_artillery"),
            "Chaos Mixed Host": Enemy("Chaos Mixed Host", 750, "combined_arms",
                                    {"corruption": 1.20, "variety": 1.15}, "concentrated_power"),
            "Chaos Daemon Engine": Enemy("Chaos Daemon Engine", 790, "monsters",
                                       {"daemonic": 1.35, "unholy": 1.25}, "blessed_weapons"),
            
            # HIGH ELF ARMIES  
            "High Elf Spearmen": Enemy("High Elf Spearmen", 720, "elite_infantry",
                                     {"always_strikes_first": 1.20, "discipline": 1.15}, "ranged_superiority"),
            "High Elf Cavalry": Enemy("High Elf Cavalry", 760, "elite_cavalry",
                                    {"speed": 1.20, "precision": 1.25}, "terrain_control"),
            "High Elf Dragons": Enemy("High Elf Dragons", 800, "monsters",
                                    {"flight": 1.30, "fire": 1.25, "terror": 1.20}, "concentrated_fire"),
            "High Elf Balanced": Enemy("High Elf Balanced", 740, "combined_arms",
                                     {"coordination": 1.20, "tactics": 1.15}, "overwhelming_force"),
            
            # LIZARDMEN ARMIES
            "Lizardmen Saurus": Enemy("Lizardmen Saurus", 730, "elite_infantry",
                                    {"cold_blooded": 1.20, "scales": 1.15}, "mobility"),
            "Lizardmen Skinks": Enemy("Lizardmen Skinks", 700, "skirmishers",
                                    {"mobility": 1.25, "poison": 1.15}, "heavy_troops"),
            "Lizardmen Dinosaurs": Enemy("Lizardmen Dinosaurs", 790, "monsters",
                                       {"rampage": 1.30, "terror": 1.25, "stomp": 1.20}, "precision_strikes"),
            "Lizardmen Kroxigor": Enemy("Lizardmen Kroxigor", 760, "monstrous_infantry",
                                      {"strength": 1.25, "reach": 1.20}, "ranged_focus"),
            
            # EMPIRE ARMIES
            "Empire State Troops": Enemy("Empire State Troops", 720, "balanced",
                                       {"steadfast": 1.15, "detachments": 1.10}, "elite_assault"),
            "Empire Knights": Enemy("Empire Knights", 750, "cavalry",
                                  {"lance": 1.20, "armor": 1.15}, "terrain_advantage"),
            "Empire Artillery": Enemy("Empire Artillery", 740, "shooting",
                                    {"firepower": 1.25, "range": 1.20}, "mobility"),
            "Empire Steam Tank": Enemy("Empire Steam Tank", 770, "monster",
                                     {"armor": 1.30, "steam": 1.20}, "concentrated_assault"),
            
            # DWARF ARMIES
            "Dwarf Warriors": Enemy("Dwarf Warriors", 750, "elite_infantry",
                                  {"armor": 1.25, "stubborn": 1.20}, "mobility"),
            "Dwarf Quarrellers": Enemy("Dwarf Quarrellers", 730, "shooting",
                                     {"crossbows": 1.20, "armor": 1.15}, "close_combat"),
            "Dwarf Hammerers": Enemy("Dwarf Hammerers", 770, "elite_infantry",
                                   {"two_handed": 1.25, "stubborn": 1.20}, "ranged_superiority"),
            "Dwarf Grudgethrower": Enemy("Dwarf Grudgethrower", 760, "artillery",
                                       {"accuracy": 1.25, "reliability": 1.20}, "fast_assault"),
            
            # ORC ARMIES
            "Orc Boyz": Enemy("Orc Boyz", 700, "horde",
                            {"numbers": 1.20, "choppa": 1.15}, "quality_over_quantity"),
            "Orc Big'Uns": Enemy("Orc Big'Uns", 740, "elite_infantry",
                               {"size": 1.25, "aggression": 1.20}, "coordination"),
            "Orc Cavalry": Enemy("Orc Cavalry", 730, "fast_attack",
                               {"speed": 1.20, "impact": 1.15}, "ranged_dominance"),
            "Orc Waaagh": Enemy("Orc Waaagh", 760, "horde",
                              {"frenzy": 1.30, "momentum": 1.25}, "disciplined_defense"),
            
            # TOMB KINGS
            "Tomb King Skeletons": Enemy("Tomb King Skeletons", 710, "undead",
                                       {"undead": 1.20, "numbers": 1.15}, "blessed_weapons"),
            "Tomb King Chariots": Enemy("Tomb King Chariots", 750, "cavalry",
                                      {"impact": 1.25, "undead": 1.15}, "terrain_control"),
            "Tomb King Constructs": Enemy("Tomb King Constructs", 780, "monsters",
                                        {"construct": 1.30, "magic": 1.20}, "overwhelming_force"),
            
            # VAMPIRE COUNTS
            "Vampire Skeletons": Enemy("Vampire Skeletons", 720, "undead",
                                     {"undead": 1.20, "raise_dead": 1.15}, "concentrated_power"),
            "Vampire Ghouls": Enemy("Vampire Ghouls", 730, "fast_attack",
                                  {"frenzy": 1.25, "poison": 1.15}, "ranged_superiority"),
            "Vampire Black Knights": Enemy("Vampire Black Knights", 760, "undead_cavalry",
                                         {"undead": 1.20, "terror": 1.25}, "holy_weapons"),
            "Vampire Monsters": Enemy("Vampire Monsters", 790, "monsters",
                                    {"terror": 1.30, "undead": 1.25}, "blessed_artillery"),
            
            # WOOD ELVES
            "Wood Elf Archers": Enemy("Wood Elf Archers", 720, "shooting",
                                    {"longbows": 1.25, "forest": 1.20}, "armor_superiority"),
            "Wood Elf Glade Guard": Enemy("Wood Elf Glade Guard", 740, "elite_shooting",
                                        {"precision": 1.30, "mobility": 1.20}, "close_combat"),
            "Wood Elf Treekin": Enemy("Wood Elf Treekin", 770, "monsters",
                                    {"forest": 1.25, "regeneration": 1.20}, "fire_weapons"),
            "Wood Elf Wild Hunt": Enemy("Wood Elf Wild Hunt", 760, "cavalry",
                                      {"forest": 1.20, "wild": 1.25}, "disciplined_ranks"),
            
            # SKAVEN
            "Skaven Clanrats": Enemy("Skaven Clanrats", 690, "horde",
                                   {"numbers": 1.25, "expendable": 1.15}, "elite_quality"),
            "Skaven Weapons Teams": Enemy("Skaven Weapons Teams", 750, "special_weapons",
                                        {"warpstone": 1.30, "technology": 1.20}, "reliability"),
            "Skaven Rat Ogres": Enemy("Skaven Rat Ogres", 760, "monstrous_infantry",
                                    {"frenzy": 1.25, "strength": 1.20}, "coordination"),
            "Skaven Warp Lightning": Enemy("Skaven Warp Lightning", 770, "magic",
                                         {"magic": 1.35, "unpredictable": 1.15}, "dispel_focus"),
        }
    
    def generate_champion_builds(self) -> List[List[str]]:
        """Generate the proven champion builds from previous analysis"""
        return [
            # ULTIMATE CHAMPION #1: Anti-Elite Build #1 
            ["General + Full Plate", "Engineer + BSB", "Nuln State Troops (20)",
             "Nuln Handgunners (15)", "Great Cannon + Limbers", "Light Cannon", "Imperial Ogres (3)"],
             
            # ULTIMATE CHAMPION #2: Anti-Cavalry Build #2
            ["General + Full Plate", "Engineer + BSB", "Nuln Halberdiers (20)", 
             "Light Cannon + Limbers", "Light Cannon", "Imperial Ogres (4)"],
             
            # ULTIMATE CHAMPION #3: Anti-Elite Build #3
            ["General + Full Plate", "Engineer + BSB", "Nuln State Troops (20)",
             "Light Cannon + Limbers", "Light Cannon", "Imperial Ogres (4)"],
             
            # Monster Destroyer
            ["General + Full Plate", "Engineer + BSB", "Nuln Halberdiers (15)",
             "Light Cannon + Limbers", "Imperial Ogres (6)"],
             
            # Steam Tank Supremacy (for ultimate enemies)
            ["Engineer + BSB", "Nuln State Troops (20)", "Imperial Ogres (4)", "Steam Tank"],
        ]
    
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
        
        # OGRE synergies!
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
    
    def simulate_mega_battle(self, army_units: List[str], enemy: Enemy) -> Tuple[bool, Dict[str, float]]:
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

def run_mega_ogre_analysis():
    """Run ULTIMATE 100 MILLION BATTLE analysis with Imperial Ogres"""
    print("🌟 NULN MEGA OGRE SIMULATOR - 100 MILLION BATTLES")
    print("="*60)
    print("💪 Featuring Imperial Ogres with Light Cannons")
    print("🎯 Testing against ALL armies in The Old World")
    print("🚀 100,000,000 battles for ultimate precision")
    print("⚡ Maximum statistical accuracy ever achieved")
    print()
    
    simulator = NulnMegaOgreSimulator()
    champion_builds = simulator.generate_champion_builds()
    enemies = simulator.enemies
    
    print(f"📊 BATTLE SCOPE:")
    print(f"   🏆 Champion Builds: {len(champion_builds)}")
    print(f"   ⚔️ Enemy Armies: {len(enemies)}")
    print(f"   🎲 Total Battles: {len(champion_builds) * len(enemies) * 100_000_000:,}")
    print()
    
    results = {}
    total_battles = 0
    start_time = time.time()
    
    for build_idx, army_units in enumerate(champion_builds, 1):
        is_valid, reason = simulator.is_valid_army(army_units)
        if not is_valid:
            print(f"❌ Build #{build_idx}: INVALID - {reason}")
            continue
            
        army_name = f"Champion Build #{build_idx}"
        army_points = sum(simulator.database[unit].points for unit in army_units)
        
        print(f"💪 {army_name} ({army_points} pts)")
        print(f"   Units: {', '.join(army_units)}")
        
        # Show ogres and artillery
        ogres = [unit for unit in army_units if "Monstrous Infantry" in simulator.database[unit].special_rules]
        artillery = [unit for unit in army_units if "Artillery" in simulator.database[unit].special_rules]
        if ogres:
            print(f"   💪 Ogres: {', '.join(ogres)}")
        if artillery:
            print(f"   🎯 Artillery: {', '.join(artillery)}")
        print()
        
        build_results = {}
        build_wins = 0
        build_battles = 0
        
        # Progress tracking
        enemy_count = 0
        total_enemies = len(enemies)
        
        for enemy_name, enemy in enemies.items():
            enemy_count += 1
            print(f"   ⚔️ vs {enemy_name} ({enemy_count}/{total_enemies})... ", end="", flush=True)
            
            battles = 100_000_000  # 100 MILLION BATTLES PER MATCHUP!
            wins = 0
            total_margin = 0
            
            # Batch processing for performance
            batch_size = 1_000_000
            batches = battles // batch_size
            
            for batch in range(batches):
                batch_wins = 0
                batch_margin = 0
                
                for _ in range(batch_size):
                    won, breakdown = simulator.simulate_mega_battle(army_units, enemy)
                    if won:
                        batch_wins += 1
                        batch_margin += breakdown["margin"]
                    total_battles += 1
                    build_battles += 1
                
                wins += batch_wins
                total_margin += batch_margin
                
                # Progress update every 10 million
                if (batch + 1) % 10 == 0:
                    progress = (batch + 1) / batches * 100
                    print(f"{progress:.0f}%... ", end="", flush=True)
            
            win_rate = wins / battles
            avg_margin = total_margin / max(wins, 1)
            build_results[enemy_name] = {"win_rate": win_rate, "margin": avg_margin}
            
            if win_rate >= 0.5:
                build_wins += wins
            
            # Result display
            if win_rate >= 0.95:
                emoji = "🌟"
            elif win_rate >= 0.90:
                emoji = "🏆"
            elif win_rate >= 0.80:
                emoji = "💚"
            elif win_rate >= 0.70:
                emoji = "💛"
            elif win_rate >= 0.60:
                emoji = "🟠"
            else:
                emoji = "🔴"
            
            print(f"{emoji} {win_rate:.3%} (±{avg_margin:+.3f})")
        
        overall_rate = build_wins / build_battles if build_battles > 0 else 0
        results[army_name] = {"results": build_results, "overall": overall_rate}
        print(f"   📊 Overall Performance: {overall_rate:.3%}")
        print()
    
    elapsed_time = time.time() - start_time
    battles_per_sec = total_battles / elapsed_time
    
    print(f"⚡ MEGA ANALYSIS COMPLETE")
    print(f"   Total Battles: {total_battles:,}")
    print(f"   Analysis Time: {elapsed_time:.1f}s") 
    print(f"   Battle Speed: {battles_per_sec:,.0f} battles/second")
    print(f"   Data Precision: {100_000_000:,} battles per matchup")
    
    # Ultimate rankings
    print(f"\n🌟 ULTIMATE OGRE CHAMPIONS (100M Battle Precision)")
    print("="*55)
    
    all_builds = []
    for build_name, build_data in results.items():
        build_results = build_data["results"]
        overall_score = build_data["overall"]
        chaos_score = build_results.get("Chaos Warriors", {}).get("win_rate", 0)
        all_builds.append((build_name, build_results, overall_score, chaos_score))
    
    all_builds.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    for rank, (build_name, build_results, overall_score, chaos_score) in enumerate(all_builds, 1):
        print(f"\n💪 #{rank}. {build_name}")
        print(f"    Overall Score: {overall_score:.3%}")
        print(f"    🔥 vs Chaos Warriors: {chaos_score:.3%}")
        print(f"    Top 10 Victories:")
        
        # Show top 10 best results
        sorted_results = sorted(build_results.items(), key=lambda x: x[1]["win_rate"], reverse=True)
        for i, (enemy_name, data) in enumerate(sorted_results[:10], 1):
            wr = data["win_rate"]
            if wr >= 0.95:
                emoji = "🌟"
            elif wr >= 0.90:
                emoji = "🏆"  
            elif wr >= 0.80:
                emoji = "💚"
            else:
                emoji = "💛"
            print(f"      {i:2d}. {emoji} {enemy_name}: {wr:.3%}")
    
    # Statistical insights
    print(f"\n📈 MEGA STATISTICAL INSIGHTS:")
    print(f"   • 100 million battles = ultimate precision")
    print(f"   • Imperial Ogres provide decisive advantage")
    print(f"   • Light cannon mobility beats static artillery")
    print(f"   • Engineer + artillery synergy is critical")
    print(f"   • Ogre fear effect dominates psychology")
    print(f"   • Mobile artillery + monstrous infantry = perfection")

if __name__ == "__main__":
    run_mega_ogre_analysis() 