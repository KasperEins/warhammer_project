#!/usr/bin/env python3

import random
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
from enum import Enum
from collections import defaultdict

class TacticalPhase(Enum):
    DEPLOYMENT = "deployment"
    EARLY_GAME = "early_game"
    MID_GAME = "mid_game"
    LATE_GAME = "late_game"
    DECISIVE_MOMENT = "decisive_moment"

@dataclass
class BattlePhaseAnalysis:
    phase: TacticalPhase
    nuln_advantage: float
    key_actions: List[str]
    critical_success_factors: List[str]
    common_mistakes: List[str]

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
    tactical_roles: List[str] = None

@dataclass
class Enemy:
    name: str
    army_type: str
    total_points: int
    primary_threat: str
    faction_bonuses: Dict[str, float]
    weaknesses: List[str]
    tactical_phases: Dict[TacticalPhase, float]
    preferred_deployment: str
    win_conditions: List[str]

@dataclass
class StrategicMatchup:
    army_build: str
    enemy: str
    win_rate: float
    avg_margin: float
    tactical_breakdown: Dict[TacticalPhase, float]
    key_moments: List[str]
    step_by_step_guide: List[str]

class NulnMegaStrategicAnalyzer:
    def __init__(self):
        self.database = self._create_enhanced_database()
        self.enhanced_enemies = self._create_enhanced_enemies()
        self.battle_phase_data = defaultdict(list)
        
    def _create_enhanced_database(self) -> Dict[str, Unit]:
        """Enhanced Nuln database with tactical roles"""
        return {
            # Characters
            "General of the Empire": Unit("General of the Empire", 85, "character", 1, 1, 4.0, 
                                        ["Leadership", "Inspiring Presence"], False, {"cavalry": 0.1, "elite": 0.15},
                                        ["Command", "Inspire", "Combat Support"]),
            "Captain of the Empire": Unit("Captain of the Empire", 45, "character", 1, 1, 2.5,
                                        ["Leadership", "Battle Hardened"], False, {"cavalry": 0.15, "elite": 0.1},
                                        ["Line Command", "Combat Leader"]),
            "Empire Engineer": Unit("Empire Engineer", 45, "character", 1, 1, 2.0,
                                  ["Artillery Support", "Master of Artillery"], False, {"elite": 0.2, "monsters": 0.25},
                                  ["Artillery Coordination", "Mechanical Warfare"]),
            "Battle Standard Bearer": Unit("Battle Standard Bearer", 25, "upgrade", 1, 1, 1.0,
                                         ["Banner", "Rally Point"], False, {"all": 0.1},
                                         ["Morale Anchor", "Formation Control"]),
            "Full Plate Armour": Unit("Full Plate Armour", 8, "upgrade", 1, 1, 0.3,
                                    ["Armor", "Heavy Protection"], False, {"elite": 0.1},
                                    ["Survivability", "Elite Counter"]),
            
            # MANDATORY Core Units
            "Nuln State Troops (20)": Unit("Nuln State Troops (20)", 100, "core", 10, 30, 3.2,
                                          ["Steadfast", "Mandatory", "Formation Fighting"], False, {"cavalry": 0.25},
                                          ["Line Infantry", "Objective Holding", "Anvil"]),
            "Nuln State Troops (25)": Unit("Nuln State Troops (25)", 125, "core", 10, 30, 4.0,
                                          ["Steadfast", "Mandatory", "Formation Fighting"], False, {"cavalry": 0.3},
                                          ["Heavy Line", "Area Control", "Mass Formation"]),
            "Nuln Veteran State Troops (15)": Unit("Nuln Veteran State Troops (15)", 105, "core", 10, 25, 3.0,
                                                  ["Veteran", "Mandatory", "Experienced"], False, {"elite": 0.2, "cavalry": 0.15},
                                                  ["Elite Line", "Reliable Core", "Veteran Tactics"]),
            "Nuln Veteran State Troops (20)": Unit("Nuln Veteran State Troops (20)", 140, "core", 10, 25, 4.0,
                                                  ["Veteran", "Mandatory", "Experienced"], False, {"elite": 0.25, "cavalry": 0.2},
                                                  ["Premium Line", "Elite Counter", "Tactical Flexibility"]),
            
            # MANDATORY Halberdiers
            "Nuln State Halberdiers (15)": Unit("Nuln State Halberdiers (15)", 90, "core", 10, 30, 2.8,
                                               ["Halberd", "Anti-Cavalry", "Mandatory", "Reach"], False, {"cavalry": 0.4, "monsters": 0.2},
                                               ["Cavalry Counter", "Monster Support", "Defensive Line"]),
            "Nuln State Halberdiers (20)": Unit("Nuln State Halberdiers (20)", 120, "core", 10, 30, 3.6,
                                               ["Halberd", "Anti-Cavalry", "Mandatory", "Reach"], False, {"cavalry": 0.45, "monsters": 0.25},
                                               ["Heavy Cavalry Counter", "Anti-Monster", "Defensive Anchor"]),
            
            # Shooting Core
            "Nuln State Handgunners (10)": Unit("Nuln State Handgunners (10)", 60, "core", 5, 20, 2.8,
                                               ["Handgun Drill", "Armor Piercing"], False, {"elite": 0.35, "monsters": 0.25},
                                               ["Elite Hunting", "Skirmisher Screen", "Fire Support"]),
            "Nuln State Handgunners (15)": Unit("Nuln State Handgunners (15)", 90, "core", 5, 20, 4.0,
                                               ["Handgun Drill", "Armor Piercing"], False, {"elite": 0.4, "monsters": 0.3},
                                               ["Heavy Fire Support", "Elite Elimination", "Covering Fire"]),
            "Nuln Veteran Outriders (8)": Unit("Nuln Veteran Outriders (8)", 152, "core", 5, 10, 4.2,
                                              ["Fast Cavalry", "No Ponderous", "Veteran"], True, {"cavalry": 0.4},
                                              ["Mobile Response", "Flank Security", "Hit and Run"]),
            
            # Special Units
            "Empire Greatswords (12)": Unit("Empire Greatswords (12)", 144, "special", 5, 20, 4.8,
                                          ["Elite", "Great Weapons", "Immune to Psychology"], False, {"elite": 0.25, "monsters": 0.35},
                                          ["Elite Counter", "Monster Hunting", "Breakthrough Force"]),
            "Empire Greatswords (15)": Unit("Empire Greatswords (15)", 180, "special", 5, 20, 5.5,
                                          ["Elite", "Great Weapons", "Immune to Psychology"], False, {"elite": 0.3, "monsters": 0.4},
                                          ["Heavy Elite Counter", "Monster Destroyer", "Assault Force"]),
            "Empire Knights (5)": Unit("Empire Knights (5)", 100, "special", 3, 12, 4.5,
                                     ["Heavy Cavalry", "Devastating Charge"], False, {"cavalry": 0.4, "elite": 0.15},
                                     ["Hammer Force", "Flank Attack", "Counter-Charge"]),
            "Empire Knights (8)": Unit("Empire Knights (8)", 160, "special", 3, 12, 6.0,
                                     ["Heavy Cavalry", "Devastating Charge"], False, {"cavalry": 0.5, "elite": 0.2},
                                     ["Heavy Hammer", "Decisive Charge", "Battlefield Control"]),
            "Great Cannon": Unit("Great Cannon", 125, "special", 1, 4, 4.0,
                               ["Artillery", "High Strength", "Long Range"], False, {"elite": 0.3, "monsters": 0.5},
                               ["Monster Hunting", "Elite Sniping", "Long Range Support"]),
            "Great Cannon with Gun Limbers": Unit("Great Cannon with Gun Limbers", 135, "special", 1, 4, 4.5,
                                                ["Artillery", "Vanguard", "High Strength"], False, {"elite": 0.35, "monsters": 0.55},
                                                ["Mobile Artillery", "Rapid Deployment", "Flexible Fire Support"]),
            "Mortar": Unit("Mortar", 90, "special", 1, 2, 3.5,
                         ["Artillery", "Indirect Fire", "Template"], False, {"elite": 0.4, "cavalry": 0.3},
                         ["Area Denial", "Formation Breaking", "Indirect Support"]),
            
            # Rare Units
            "Helblaster Volley Gun": Unit("Helblaster Volley Gun", 120, "rare", 1, 2, 4.2,
                                        ["Multi-shot", "Anti-Infantry"], False, {"cavalry": 0.6, "elite": 0.2},
                                        ["Cavalry Destroyer", "Mass Infantry Counter", "Volume Fire"]),
            "Helblaster with Gun Limbers": Unit("Helblaster with Gun Limbers", 135, "rare", 1, 2, 4.8,
                                              ["Multi-shot", "Vanguard", "Anti-Infantry"], False, {"cavalry": 0.7, "elite": 0.25},
                                              ["Mobile Destruction", "Rapid Response", "Flexible Firepower"]),
            "Steam Tank": Unit("Steam Tank", 285, "rare", 1, 1, 8.0,
                             ["Monster", "Terror", "High Toughness"], True, {"cavalry": 0.8, "elite": 0.4, "monsters": 0.6},
                             ["Terror Weapon", "Breakthrough Unit", "Anvil Breaker"]),
            
            # Mercenaries
            "Imperial Dwarfs (12)": Unit("Imperial Dwarfs (12)", 108, "mercenary", 10, 25, 4.0,
                                       ["Resilient", "Stubborn", "Magic Resistance"], False, {"elite": 0.2, "monsters": 0.25},
                                       ["Defensive Anchor", "Magic Counter", "Reliable Infantry"]),
            "Imperial Dwarfs (15)": Unit("Imperial Dwarfs (15)", 135, "mercenary", 10, 25, 4.8,
                                       ["Resilient", "Stubborn", "Magic Resistance"], False, {"elite": 0.25, "monsters": 0.3},
                                       ["Heavy Defensive Line", "Elite Counter", "Unbreakable Core"]),
        }
    
    def _create_enhanced_enemies(self) -> Dict[str, Enemy]:
        """Enhanced enemy analysis with tactical phases"""
        return {
            "Bretonnian Knights": Enemy(
                "Bretonnian Knights", "cavalry_mobile", 750, "cavalry",
                {"chivalry": 1.20, "cavalry_charge": 1.25, "virtue": 1.15},
                ["anti_cavalry", "defensive_positioning", "volume_shooting", "terrain_advantage"],
                {
                    TacticalPhase.DEPLOYMENT: 0.7,  # Weak at deployment
                    TacticalPhase.EARLY_GAME: 1.2,  # Strong early charge
                    TacticalPhase.MID_GAME: 1.1,    # Good momentum
                    TacticalPhase.LATE_GAME: 0.8,   # Weakens if charge fails
                    TacticalPhase.DECISIVE_MOMENT: 1.3  # Devastating charge
                },
                "aggressive_forward", ["Successful cavalry charge", "Break enemy line", "Avoid prolonged combat"]
            ),
            "Chaos Warriors": Enemy(
                "Chaos Warriors", "elite_small", 760, "elite_infantry",
                {"favor_of_gods": 1.30, "fear": 1.15, "chaos_armor": 1.20},
                ["armor_piercing", "volume_fire", "artillery", "magic_weapons"],
                {
                    TacticalPhase.DEPLOYMENT: 1.0,
                    TacticalPhase.EARLY_GAME: 1.1,
                    TacticalPhase.MID_GAME: 1.3,    # Peak performance
                    TacticalPhase.LATE_GAME: 1.2,   # Still strong
                    TacticalPhase.DECISIVE_MOMENT: 1.4  # Unstoppable in combat
                },
                "methodical_advance", ["Reach combat intact", "Win prolonged melee", "Ignore casualties"]
            ),
            "High Elf Elite": Enemy(
                "High Elf Elite", "elite_small", 730, "elite_infantry",
                {"always_strikes_first": 1.25, "martial_prowess": 1.10, "elven_reflexes": 1.15},
                ["artillery", "volume_shooting", "area_denial", "overwhelm_with_numbers"],
                {
                    TacticalPhase.DEPLOYMENT: 1.1,  # Good positioning
                    TacticalPhase.EARLY_GAME: 1.2,  # Excellent early game
                    TacticalPhase.MID_GAME: 1.3,    # Peak tactical phase
                    TacticalPhase.LATE_GAME: 1.0,   # Weakens with casualties
                    TacticalPhase.DECISIVE_MOMENT: 1.1  # Good but not overwhelming
                },
                "tactical_superiority", ["Maintain formation", "Win key combats", "Exploit enemy mistakes"]
            ),
            "Lizardmen Temple": Enemy(
                "Lizardmen Temple", "monster_mash", 780, "monsters",
                {"cold_blooded": 1.15, "scaly_skin": 1.10, "ancient_power": 1.20, "predatory_fighter": 1.25},
                ["high_strength", "artillery", "armor_piercing", "concentrated_fire"],
                {
                    TacticalPhase.DEPLOYMENT: 0.9,
                    TacticalPhase.EARLY_GAME: 1.0,
                    TacticalPhase.MID_GAME: 1.2,
                    TacticalPhase.LATE_GAME: 1.3,   # Monsters dominate late
                    TacticalPhase.DECISIVE_MOMENT: 1.5  # Overwhelming power
                },
                "monster_rush", ["Get monsters into combat", "Overwhelm key units", "Use raw power"]
            )
        }
    
    def is_valid_army(self, army: List[str]) -> Tuple[bool, str]:
        """Validate army against Nuln Army of Infamy restrictions"""
        total_points = sum(self.database[unit].points for unit in army)
        if total_points > 750:
            return False, f"Exceeds 750 points ({total_points})"
        
        # Check category limits
        character_points = sum(self.database[unit].points for unit in army if self.database[unit].category in ["character", "upgrade"])
        core_points = sum(self.database[unit].points for unit in army if self.database[unit].category == "core")
        special_points = sum(self.database[unit].points for unit in army if self.database[unit].category == "special")
        rare_points = sum(self.database[unit].points for unit in army if self.database[unit].category == "rare")
        
        if character_points > 188: return False, f"Character limit exceeded ({character_points}/188)"
        if core_points > 263: return False, f"Core limit exceeded ({core_points}/263)"
        if special_points > 225: return False, f"Special limit exceeded ({special_points}/225)"
        if rare_points > 188: return False, f"Rare limit exceeded ({rare_points}/188)"
        
        # Check mandatory requirements
        has_mandatory = any("Mandatory" in self.database[unit].special_rules for unit in army)
        if not has_mandatory:
            return False, "Must include mandatory Nuln troops or halberdiers"
        
        # Check per-1000 restrictions
        per_1000_units = [unit for unit in army if self.database[unit].per_1000_restriction]
        if len(per_1000_units) > 2:
            return False, f"Too many 0-X per 1000 units ({len(per_1000_units)}/2)"
        
        return True, "Valid"
    
    def generate_comprehensive_strategies(self) -> Dict[str, List[List[str]]]:
        """Generate comprehensive counter-strategies"""
        strategies = {}
        
        # Anti-Cavalry Specialist Builds
        strategies["Anti-Cavalry Specialists"] = [
            # Halberdier Castle
            ["General of the Empire", "Full Plate Armour", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Halberdiers (20)", "Nuln State Handgunners (15)", "Helblaster Volley Gun"],
            
            # Mobile Anti-Cavalry
            ["Captain of the Empire", "Battle Standard Bearer", "Nuln State Halberdiers (15)",
             "Nuln Veteran Outriders (8)", "Empire Knights (5)", "Great Cannon"],
            
            # Defensive Wall
            ["General of the Empire", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (20)", "Nuln State Halberdiers (20)", "Great Cannon with Gun Limbers"]
        ]
        
        # Anti-Elite Destroyers
        strategies["Anti-Elite Destroyers"] = [
            # Artillery Superiority
            ["General of the Empire", "Full Plate Armour", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (20)", "Nuln State Handgunners (15)", "Great Cannon", "Mortar"],
            
            # Elite Hunter
            ["General of the Empire", "Captain of the Empire", "Battle Standard Bearer",
             "Nuln Veteran State Troops (15)", "Empire Greatswords (15)", "Great Cannon"],
            
            # Mass Firepower
            ["General of the Empire", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (20)", "Nuln State Handgunners (15)", "Helblaster Volley Gun"]
        ]
        
        # Anti-Monster Specialists
        strategies["Anti-Monster Specialists"] = [
            # Double Artillery
            ["General of the Empire", "Full Plate Armour", "Empire Engineer", "Battle Standard Bearer",
             "Nuln State Troops (20)", "Great Cannon", "Great Cannon"],
            
            # Monster Hunter Elite
            ["General of the Empire", "Captain of the Empire", "Battle Standard Bearer",
             "Nuln Veteran State Troops (15)", "Empire Greatswords (15)", "Great Cannon with Gun Limbers"],
            
            # Steam Tank Counter
            ["General of the Empire", "Empire Engineer", "Nuln State Halberdiers (15)", "Steam Tank"]
        ]
        
        return strategies
    
    def simulate_enhanced_battle(self, army_units: List[str], enemy: Enemy) -> Tuple[bool, Dict[str, float]]:
        """Enhanced battle simulation with phase analysis"""
        
        # Calculate base effectiveness
        nuln_base = sum(self.database[unit].effectiveness for unit in army_units)
        enemy_base = (enemy.total_points / 750.0) * 20.0
        
        # Apply faction bonuses
        faction_multiplier = 1.0
        for unit_name in army_units:
            unit = self.database[unit_name]
            if unit.counter_bonuses:
                if enemy.primary_threat in unit.counter_bonuses:
                    faction_multiplier += unit.counter_bonuses[enemy.primary_threat]
        
        # Nuln faction bonuses
        engineers = sum(1 for unit in army_units if "Engineer" in unit)
        artillery = sum(1 for unit in army_units if "Artillery" in self.database[unit].special_rules)
        handgunners = sum(1 for unit in army_units if "Handgun Drill" in self.database[unit].special_rules)
        
        if engineers > 0 and artillery > 0:
            faction_multiplier += min(engineers, artillery) * 0.15
        faction_multiplier += handgunners * 0.08
        if artillery >= 2:
            faction_multiplier += 0.12
        
        # Apply enemy faction bonuses
        enemy_multiplier = 1.0
        for bonus in enemy.faction_bonuses.values():
            enemy_multiplier *= bonus
        
        # Phase-by-phase analysis
        phase_results = {}
        overall_nuln_advantage = 0
        
        for phase, enemy_strength in enemy.tactical_phases.items():
            # Calculate Nuln response to this phase
            nuln_phase_strength = self._calculate_phase_strength(army_units, phase, enemy)
            phase_advantage = nuln_phase_strength / enemy_strength
            phase_results[phase.value] = phase_advantage
            overall_nuln_advantage += phase_advantage * 0.2  # Each phase is 20% weight
        
        # Final calculations
        nuln_final = nuln_base * faction_multiplier * overall_nuln_advantage
        enemy_final = enemy_base * enemy_multiplier
        
        # Battle variance
        tactical_variance = random.uniform(0.85, 1.15)
        battle_luck = random.gauss(1.0, 0.10)
        battle_luck = max(0.7, min(1.3, battle_luck))
        
        nuln_final *= tactical_variance * battle_luck
        enemy_final *= random.uniform(0.95, 1.05)
        
        victory_margin = (nuln_final - enemy_final) / enemy_final if enemy_final > 0 else 0
        
        breakdown = {
            "nuln_base": nuln_base,
            "faction_multiplier": faction_multiplier,
            "phase_advantage": overall_nuln_advantage,
            "victory_margin": victory_margin,
            "nuln_final": nuln_final,
            "enemy_final": enemy_final,
            **phase_results
        }
        
        return nuln_final > enemy_final, breakdown
    
    def _calculate_phase_strength(self, army_units: List[str], phase: TacticalPhase, enemy: Enemy) -> float:
        """Calculate army strength in specific tactical phase"""
        base_strength = 1.0
        
        if phase == TacticalPhase.DEPLOYMENT:
            # Benefits from flexible units
            if any("Vanguard" in self.database[unit].special_rules for unit in army_units):
                base_strength += 0.15
            if any("Fast Cavalry" in self.database[unit].special_rules for unit in army_units):
                base_strength += 0.10
                
        elif phase == TacticalPhase.EARLY_GAME:
            # Benefits from shooting and positioning
            artillery_count = sum(1 for unit in army_units if "Artillery" in self.database[unit].special_rules)
            base_strength += artillery_count * 0.12
            if any("Handgun Drill" in self.database[unit].special_rules for unit in army_units):
                base_strength += 0.08
                
        elif phase == TacticalPhase.MID_GAME:
            # Benefits from combined arms
            has_melee = any("Elite" in self.database[unit].special_rules for unit in army_units)
            has_shooting = any("Artillery" in self.database[unit].special_rules for unit in army_units)
            if has_melee and has_shooting:
                base_strength += 0.15
                
        elif phase == TacticalPhase.LATE_GAME:
            # Benefits from resilient units
            if any("Stubborn" in self.database[unit].special_rules for unit in army_units):
                base_strength += 0.12
            if any("Terror" in self.database[unit].special_rules for unit in army_units):
                base_strength += 0.10
                
        elif phase == TacticalPhase.DECISIVE_MOMENT:
            # Benefits from game-changing units
            if any("Monster" in self.database[unit].special_rules for unit in army_units):
                base_strength += 0.20
            elite_count = sum(1 for unit in army_units if "Elite" in self.database[unit].special_rules)
            base_strength += elite_count * 0.08
        
        return base_strength
    
    def analyze_strategic_matchup(self, army_units: List[str], enemy_name: str, 
                                battle_count: int = 250000) -> StrategicMatchup:
        """Comprehensive strategic analysis of a matchup"""
        enemy = self.enhanced_enemies[enemy_name]
        
        wins = 0
        total_margin = 0
        phase_performance = defaultdict(list)
        key_moments = []
        
        for _ in range(battle_count):
            won, breakdown = self.simulate_enhanced_battle(army_units, enemy)
            if won:
                wins += 1
                total_margin += breakdown["victory_margin"]
            
            # Track phase performance
            for phase in TacticalPhase:
                if phase.value in breakdown:
                    phase_performance[phase].append(breakdown[phase.value])
        
        win_rate = wins / battle_count
        avg_margin = total_margin / max(wins, 1)
        
        # Analyze tactical phases
        tactical_breakdown = {}
        for phase in TacticalPhase:
            if phase in phase_performance:
                tactical_breakdown[phase] = sum(phase_performance[phase]) / len(phase_performance[phase])
        
        # Generate step-by-step guide
        step_by_step = self._generate_tactical_guide(army_units, enemy)
        
        return StrategicMatchup(
            army_build=" + ".join(army_units[:3]) + "...",
            enemy=enemy_name,
            win_rate=win_rate,
            avg_margin=avg_margin,
            tactical_breakdown=tactical_breakdown,
            key_moments=key_moments,
            step_by_step_guide=step_by_step
        )
    
    def _generate_tactical_guide(self, army_units: List[str], enemy: Enemy) -> List[str]:
        """Generate step-by-step tactical guide"""
        guide = []
        
        # Deployment phase
        guide.append("🏁 DEPLOYMENT PHASE:")
        if any("Artillery" in self.database[unit].special_rules for unit in army_units):
            guide.append("• Position artillery on high ground with clear firing lanes")
            guide.append("• Keep artillery 12+ inches from board edge to avoid charges")
        
        if any("Halberd" in self.database[unit].special_rules for unit in army_units):
            guide.append("• Deploy halberdiers to cover artillery flanks")
            guide.append("• Form defensive line to channel enemy charges")
        
        # Early game
        guide.append("\n⚡ EARLY GAME (Turns 1-2):")
        if enemy.primary_threat == "cavalry":
            guide.append("• Focus all shooting on incoming cavalry units")
            guide.append("• Use halberdiers to threaten charge lanes")
            guide.append("• Keep units in supporting distance")
        elif enemy.primary_threat == "elite_infantry":
            guide.append("• Target elite units with artillery first")
            guide.append("• Use handgunners to thin elite ranks")
            guide.append("• Prepare defensive positions")
        elif enemy.primary_threat == "monsters":
            guide.append("• Concentrate all high-strength attacks on monsters")
            guide.append("• Use cannon to target biggest threats first")
            guide.append("• Position to avoid monster charges")
        
        # Mid game
        guide.append("\n⚔️ MID GAME (Turns 3-4):")
        guide.append("• Maintain firing lines while enemy closes")
        if any("Elite" in self.database[unit].special_rules for unit in army_units):
            guide.append("• Position elite units for counter-charges")
        guide.append("• Use engineer to keep artillery functioning")
        guide.append("• Reform units to face main threats")
        
        # Late game
        guide.append("\n🎯 LATE GAME (Turns 5-6):")
        guide.append("• Focus on objective control")
        guide.append("• Use remaining units to contest key areas")
        if any("Terror" in self.database[unit].special_rules for unit in army_units):
            guide.append("• Commit terror-causing units to break enemy morale")
        guide.append("• Preserve scoring units for victory conditions")
        
        return guide

def run_mega_strategic_analysis():
    """Run comprehensive strategic analysis with massive simulations"""
    print("🚀 NULN MEGA STRATEGIC ANALYZER")
    print("="*50)
    print("⚔️ Running 250,000+ battles per matchup")
    print("📊 Comprehensive tactical phase analysis")
    print("📋 Step-by-step strategic guides")
    print("🎯 Tournament-winning strategies")
    print()
    
    analyzer = NulnMegaStrategicAnalyzer()
    strategies = analyzer.generate_comprehensive_strategies()
    enemies = analyzer.enhanced_enemies
    
    results = {}
    total_battles = 0
    start_time = time.time()
    
    print("🏗️ TESTING COMPREHENSIVE STRATEGIES")
    print("="*45)
    
    for strategy_type, army_builds in strategies.items():
        print(f"\n🛡️ {strategy_type.upper()}")
        print("-" * 40)
        
        strategy_results = {}
        
        for i, army_units in enumerate(army_builds, 1):
            is_valid, reason = analyzer.is_valid_army(army_units)
            if not is_valid:
                print(f"❌ Build #{i}: INVALID - {reason}")
                continue
            
            army_name = f"{strategy_type} Build #{i}"
            army_points = sum(analyzer.database[unit].points for unit in army_units)
            
            print(f"\n📋 {army_name} ({army_points} pts)")
            
            # Show tactical roles
            all_roles = set()
            for unit in army_units:
                if analyzer.database[unit].tactical_roles:
                    all_roles.update(analyzer.database[unit].tactical_roles)
            print(f"   Roles: {', '.join(sorted(all_roles))}")
            
            build_results = {}
            build_total_wins = 0
            build_total_battles = 0
            
            for enemy_name in enemies.keys():
                battles = 250000  # Massive simulation count
                print(f"   🔄 Simulating vs {enemy_name} ({battles:,} battles)...")
                
                # Run comprehensive analysis
                matchup = analyzer.analyze_strategic_matchup(army_units, enemy_name, battles)
                build_results[enemy_name] = matchup
                
                if matchup.win_rate >= 0.5:
                    build_total_wins += int(matchup.win_rate * battles)
                build_total_battles += battles
                total_battles += battles
                
                # Display results with tactical breakdown
                if matchup.win_rate >= 0.70:
                    emoji = "💚"
                elif matchup.win_rate >= 0.60:
                    emoji = "💛"
                elif matchup.win_rate >= 0.50:
                    emoji = "🟠"
                else:
                    emoji = "🔴"
                
                print(f"   {emoji} vs {enemy_name:.<20} {matchup.win_rate:.1%} (margin: {matchup.avg_margin:+.2f})")
                
                # Show best tactical phases
                best_phase = max(matchup.tactical_breakdown.keys(), 
                               key=lambda p: matchup.tactical_breakdown[p])
                print(f"       Best Phase: {best_phase.value.title()} ({matchup.tactical_breakdown[best_phase]:.2f}x)")
            
            overall_rate = build_total_wins / build_total_battles if build_total_battles > 0 else 0
            strategy_results[army_name] = build_results
            print(f"\n   📊 Overall Performance: {overall_rate:.1%}")
        
        results[strategy_type] = strategy_results
    
    elapsed_time = time.time() - start_time
    battles_per_sec = total_battles / elapsed_time
    
    print(f"\n⚡ SIMULATION COMPLETE!")
    print(f"   Battles: {total_battles:,}")
    print(f"   Time: {elapsed_time:.1f}s")
    print(f"   Speed: {battles_per_sec:,.0f} battles/second")
    
    # Find best strategies
    print(f"\n🏆 TOP STRATEGIC BUILDS")
    print("="*40)
    
    all_builds = []
    for strategy_type, builds in results.items():
        for build_name, matchups in builds.items():
            overall_score = sum(m.win_rate for m in matchups.values()) / len(matchups)
            all_builds.append((build_name, matchups, overall_score, strategy_type))
    
    all_builds.sort(key=lambda x: x[2], reverse=True)
    
    # Show top 3 with detailed guides
    for rank, (build_name, matchups, score, strategy_type) in enumerate(all_builds[:3], 1):
        print(f"\n🥇 #{rank}. {build_name}")
        print(f"    Strategy: {strategy_type}")
        print(f"    Overall Score: {score:.1%}")
        print(f"    Performance Breakdown:")
        
        for enemy_name, matchup in matchups.items():
            if matchup.win_rate >= 0.60:
                emoji = "💚"
            elif matchup.win_rate >= 0.50:
                emoji = "💛"
            else:
                emoji = "🔴"
            print(f"      {emoji} {enemy_name}: {matchup.win_rate:.1%}")
        
        # Show tactical guide for best matchup
        best_matchup = max(matchups.values(), key=lambda m: m.win_rate)
        print(f"\n    📋 TACTICAL GUIDE vs {best_matchup.enemy}:")
        for step in best_matchup.step_by_step_guide[:8]:  # Show first 8 steps
            print(f"      {step}")
        
        if rank < 3:
            print()

if __name__ == "__main__":
    run_mega_strategic_analysis() 