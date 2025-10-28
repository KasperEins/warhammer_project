#!/usr/bin/env python3
"""
🏰 WARHAMMER: THE OLD WORLD - TACTICAL AI SYSTEM
==============================================

This system creates AI agents that:
1. Understand actual Warhammer game mechanics
2. Make strategic and tactical decisions
3. Generate detailed, narrative battle reports
4. Learn and improve their strategies over time

The goal is to create AI that plays like skilled human players.
"""

import random
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import math
from datetime import datetime

# =============================================================================
# CORE GAME MECHANICS
# =============================================================================

class UnitType(Enum):
    INFANTRY = "Infantry"
    CAVALRY = "Cavalry"
    MONSTER = "Monster"
    WAR_MACHINE = "War Machine"
    CHARACTER = "Character"

class Formation(Enum):
    CLOSE_ORDER = "Close Order"
    SKIRMISH = "Skirmish"
    WIDE = "Wide"
    DEEP = "Deep"

@dataclass
class Equipment:
    """Represents weapons and armor"""
    weapon: str = "Hand Weapon"
    armor: str = "None"
    shield: bool = False
    special_rules: List[str] = field(default_factory=list)
    
    def armor_save(self) -> int:
        """Calculate armor save value"""
        armor_saves = {
            "None": 7,
            "Light Armor": 6,
            "Heavy Armor": 5,
            "Full Plate": 4,
            "Chaos Armor": 3
        }
        base_save = armor_saves.get(self.armor, 7)
        if self.shield:
            base_save = max(2, base_save - 1)
        return base_save

@dataclass
class UnitStats:
    """Core unit statistics"""
    movement: int
    weapon_skill: int
    ballistic_skill: int
    strength: int
    toughness: int
    wounds: int
    initiative: int
    attacks: int
    leadership: int
    armor_save: int
    ward_save: Optional[int] = None
    special_rules: List[str] = field(default_factory=list)

@dataclass
class Unit:
    """Represents a unit on the battlefield"""
    name: str
    unit_type: UnitType
    stats: UnitStats
    equipment: Equipment
    models: int
    current_models: int
    formation: Formation
    points_cost: int
    position: Tuple[int, int] = (0, 0)
    facing: int = 0  # 0=North, 90=East, 180=South, 270=West
    current_wounds: int = None
    status_effects: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.current_wounds is None:
            self.current_wounds = self.stats.wounds
    
    def is_alive(self) -> bool:
        return self.current_models > 0
    
    def unit_strength(self) -> int:
        """Calculate unit strength for combat resolution"""
        if self.unit_type in [UnitType.MONSTER, UnitType.CHARACTER]:
            return max(1, self.stats.wounds)
        return self.current_models
    
    def combat_attacks(self) -> int:
        """Calculate total attacks in combat"""
        base_attacks = self.current_models * self.stats.attacks
        if "Frenzy" in self.status_effects:
            base_attacks *= 2
        return base_attacks

@dataclass
class Army:
    """Represents a complete army"""
    name: str
    faction: str
    units: List[Unit]
    points_total: int
    general: Optional[Unit] = None
    battle_standard: Optional[Unit] = None
    
    def remaining_points(self) -> int:
        return sum(unit.points_cost for unit in self.units if unit.is_alive())
    
    def is_broken(self) -> bool:
        """Check if army is broken (25% or less remaining)"""
        return self.remaining_points() <= (self.points_total * 0.25)

# =============================================================================
# BATTLEFIELD AND TERRAIN
# =============================================================================

@dataclass
class Terrain:
    name: str
    position: Tuple[int, int]
    size: Tuple[int, int]
    terrain_type: str  # "Hill", "Forest", "Building", "Impassable"
    special_rules: List[str] = field(default_factory=list)

class Battlefield:
    """Represents the battlefield with terrain"""
    
    def __init__(self, width: int = 72, height: int = 48):  # 6' x 4' in inches
        self.width = width
        self.height = height
        self.terrain: List[Terrain] = []
        self.deployment_zones = {
            "player1": (0, 0, width, 12),  # Bottom 12" strip
            "player2": (0, 36, width, 12)  # Top 12" strip
        }
    
    def add_terrain(self, terrain: Terrain):
        self.terrain.append(terrain)
    
    def line_of_sight(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Check if there's line of sight between two positions"""
        # Simplified LOS check - can be enhanced with terrain blocking
        return True
    
    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# =============================================================================
# TACTICAL AI BRAIN
# =============================================================================

class TacticalObjective(Enum):
    AGGRESSIVE_ASSAULT = "Aggressive Assault"
    DEFENSIVE_HOLD = "Defensive Hold"
    FLANKING_MANEUVER = "Flanking Maneuver"
    ARTILLERY_SUPPORT = "Artillery Support"
    MAGIC_DOMINANCE = "Magic Dominance"
    CAVALRY_CHARGE = "Cavalry Charge"

@dataclass
class TacticalPlan:
    """Represents a tactical plan for the battle"""
    primary_objective: TacticalObjective
    target_priority: List[str]  # Unit types to prioritize
    deployment_strategy: str
    turn_plan: Dict[int, str]  # Plans for each turn
    contingencies: Dict[str, str]  # Backup plans

class TacticalAI:
    """Advanced AI that understands Warhammer tactics"""
    
    def __init__(self, name: str, army: Army, experience_level: int = 1):
        self.name = name
        self.army = army
        self.experience_level = experience_level
        self.tactical_knowledge = self._initialize_knowledge()
        self.current_plan = None
        self.battle_history = []
        self.decision_log = []
    
    def _initialize_knowledge(self) -> Dict[str, Any]:
        """Initialize tactical knowledge base"""
        return {
            "unit_synergies": {
                "infantry_spear_wall": ["Empire State Troops", "Bretonnian Men-at-Arms"],
                "cavalry_hammer": ["Empire Knights", "Bretonnian Knights"],
                "artillery_line": ["Great Cannon", "Helblaster"]
            },
            "counter_tactics": {
                "heavy_cavalry": ["Spear Wall", "Artillery Focus", "Magic Missiles"],
                "monster": ["Concentrated Fire", "Magic Debuffs", "Flanking"],
                "horde_infantry": ["Artillery", "Magic", "Elite Units"]
            },
            "deployment_principles": [
                "Protect artillery with infantry",
                "Place cavalry on flanks for charges", 
                "Keep general central for leadership",
                "Use terrain for defensive advantage"
            ]
        }
    
    def analyze_enemy_army(self, enemy_army: Army) -> Dict[str, Any]:
        """Analyze enemy army composition and identify threats/opportunities"""
        analysis = {
            "threats": [],
            "opportunities": [],
            "army_type": self._classify_army_type(enemy_army),
            "key_units": [],
            "weaknesses": []
        }
        
        for unit in enemy_army.units:
            if unit.unit_type == UnitType.MONSTER:
                analysis["threats"].append(f"{unit.name} - High damage monster")
            elif unit.unit_type == UnitType.CAVALRY and unit.current_models >= 5:
                analysis["threats"].append(f"{unit.name} - Heavy cavalry charge")
            elif unit.unit_type == UnitType.WAR_MACHINE:
                analysis["threats"].append(f"{unit.name} - Long range artillery")
            
            if unit.stats.armor_save >= 6:  # Poor armor
                analysis["opportunities"].append(f"{unit.name} - Vulnerable to shooting")
            if unit.stats.leadership <= 7:
                analysis["opportunities"].append(f"{unit.name} - Poor leadership")
        
        return analysis
    
    def _classify_army_type(self, army: Army) -> str:
        """Classify the enemy army type for tactical purposes"""
        cavalry_count = sum(1 for u in army.units if u.unit_type == UnitType.CAVALRY)
        infantry_count = sum(1 for u in army.units if u.unit_type == UnitType.INFANTRY)
        monster_count = sum(1 for u in army.units if u.unit_type == UnitType.MONSTER)
        
        if cavalry_count >= 3:
            return "Cavalry Heavy"
        elif monster_count >= 2:
            return "Monster Mash"
        elif infantry_count >= 5:
            return "Infantry Horde"
        else:
            return "Balanced"
    
    def create_battle_plan(self, enemy_army: Army, battlefield: Battlefield) -> TacticalPlan:
        """Create a comprehensive battle plan"""
        enemy_analysis = self.analyze_enemy_army(enemy_army)
        
        # Choose primary objective based on army composition
        if self.army.faction == "Empire" and any("Artillery" in u.name for u in self.army.units):
            primary_objective = TacticalObjective.ARTILLERY_SUPPORT
        elif enemy_analysis["army_type"] == "Infantry Horde":
            primary_objective = TacticalObjective.FLANKING_MANEUVER
        else:
            primary_objective = TacticalObjective.AGGRESSIVE_ASSAULT
        
        # Create turn-by-turn plan
        turn_plan = {
            1: "Deploy defensively, prepare for enemy advance",
            2: "Advance key units, begin artillery bombardment",
            3: "Engage enemy flanks, support with magic",
            4: "Execute main assault, coordinate charges",
            5: "Exploit breakthroughs, secure victory"
        }
        
        plan = TacticalPlan(
            primary_objective=primary_objective,
            target_priority=self._determine_target_priority(enemy_analysis),
            deployment_strategy=self._plan_deployment(enemy_analysis, battlefield),
            turn_plan=turn_plan,
            contingencies={
                "heavy_losses": "Fall back to defensive positions",
                "enemy_breakthrough": "Counter-charge with reserves",
                "artillery_destroyed": "Focus on mobile warfare"
            }
        )
        
        self.current_plan = plan
        return plan
    
    def _determine_target_priority(self, enemy_analysis: Dict[str, Any]) -> List[str]:
        """Determine which enemy units to prioritize"""
        priorities = []
        
        # Always prioritize threats
        if "monster" in enemy_analysis["army_type"].lower():
            priorities.extend(["Monster", "Character"])
        if "cavalry" in enemy_analysis["army_type"].lower():
            priorities.extend(["Cavalry", "Character"])
        
        priorities.extend(["War Machine", "Infantry", "Skirmishers"])
        return priorities
    
    def _plan_deployment(self, enemy_analysis: Dict[str, Any], battlefield: Battlefield) -> str:
        """Plan deployment strategy"""
        if enemy_analysis["army_type"] == "Cavalry Heavy":
            return "Defensive line with spears forward, artillery protected"
        elif enemy_analysis["army_type"] == "Infantry Horde":
            return "Cavalry on flanks, elite units as hammer"
        else:
            return "Balanced line with combined arms support"
    
    def make_tactical_decision(self, game_state: 'GameState', available_actions: List[str]) -> Tuple[str, str]:
        """Make a tactical decision with reasoning"""
        turn = game_state.turn
        phase = game_state.phase
        
        # Get current turn plan
        turn_guidance = self.current_plan.turn_plan.get(turn, "Adapt to battlefield situation")
        
        # Analyze current situation
        situation = self._assess_situation(game_state)
        
        # Choose action based on tactical assessment
        if phase == "Movement":
            action, reasoning = self._choose_movement_action(game_state, available_actions, situation)
        elif phase == "Shooting":
            action, reasoning = self._choose_shooting_action(game_state, available_actions, situation)
        elif phase == "Combat":
            action, reasoning = self._choose_combat_action(game_state, available_actions, situation)
        else:
            action = random.choice(available_actions)
            reasoning = f"Following {phase} phase procedures"
        
        # Log decision for learning
        self.decision_log.append({
            "turn": turn,
            "phase": phase,
            "situation": situation,
            "action": action,
            "reasoning": reasoning,
            "turn_plan": turn_guidance
        })
        
        return action, reasoning
    
    def _assess_situation(self, game_state: 'GameState') -> Dict[str, Any]:
        """Assess current battlefield situation"""
        my_strength = sum(u.unit_strength() for u in self.army.units if u.is_alive())
        enemy_strength = sum(u.unit_strength() for u in game_state.enemy_army.units if u.is_alive())
        
        return {
            "strength_ratio": my_strength / max(enemy_strength, 1),
            "turn": game_state.turn,
            "phase": game_state.phase,
            "casualties_taken": self.army.points_total - self.army.remaining_points(),
            "enemy_casualties": game_state.enemy_army.points_total - game_state.enemy_army.remaining_points(),
            "momentum": "winning" if my_strength > enemy_strength else "losing"
        }
    
    def _choose_movement_action(self, game_state: 'GameState', actions: List[str], situation: Dict) -> Tuple[str, str]:
        """Choose movement action with tactical reasoning"""
        if situation["strength_ratio"] > 1.5:
            action = "Advance aggressively"
            reasoning = f"Overwhelming strength advantage ({situation['strength_ratio']:.1f}:1) - press the attack"
        elif situation["strength_ratio"] < 0.7:
            action = "Tactical withdrawal"
            reasoning = f"Outnumbered ({situation['strength_ratio']:.1f}:1) - fall back to better positions"
        elif game_state.turn <= 2:
            action = "Cautious advance"
            reasoning = "Early game - position units for coordinated assault"
        else:
            action = "Coordinated assault"
            reasoning = f"Turn {game_state.turn} - execute battle plan"
        
        return action, reasoning
    
    def _choose_shooting_action(self, game_state: 'GameState', actions: List[str], situation: Dict) -> Tuple[str, str]:
        """Choose shooting action with tactical reasoning"""
        if "Focus fire on priority target" in actions:
            action = "Focus fire on priority target"
            reasoning = "Concentrate firepower to eliminate key enemy unit"
        else:
            action = random.choice(actions) if actions else "No shooting"
            reasoning = "Opportunity fire at available targets"
        
        return action, reasoning
    
    def _choose_combat_action(self, game_state: 'GameState', actions: List[str], situation: Dict) -> Tuple[str, str]:
        """Choose combat action with tactical reasoning"""
        if situation["momentum"] == "winning":
            action = "Press advantage"
            reasoning = "Winning momentum - maintain pressure"
        else:
            action = "Defensive combat"
            reasoning = "Unfavorable position - fight defensively"
        
        return action, reasoning

# =============================================================================
# GAME STATE AND BATTLE SYSTEM
# =============================================================================

@dataclass
class GameState:
    """Represents the current state of the battle"""
    turn: int
    phase: str  # "Movement", "Magic", "Shooting", "Combat"
    active_player: str
    army1: Army
    army2: Army
    battlefield: Battlefield
    victory_points: Dict[str, int] = field(default_factory=lambda: {"player1": 0, "player2": 0})
    battle_log: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def enemy_army(self) -> Army:
        """Get the enemy army from perspective of active player"""
        return self.army2 if self.active_player == "player1" else self.army1

class BattleNarrator:
    """Generates detailed battle narratives like human battle reports"""
    
    def __init__(self):
        self.battle_story = []
        self.combat_details = []
        self.tactical_notes = []
    
    def narrate_deployment(self, army1: Army, army2: Army, ai1: TacticalAI, ai2: TacticalAI) -> str:
        """Generate deployment narrative"""
        narrative = f"""
🏰 DEPLOYMENT PHASE
=====================================

**{army1.name}** ({army1.faction}): {army1.points_total} points
Strategy: {ai1.current_plan.deployment_strategy}

{self._describe_army_deployment(army1)}

**{army2.name}** ({army2.faction}): {army2.points_total} points  
Strategy: {ai2.current_plan.deployment_strategy}

{self._describe_army_deployment(army2)}

**Tactical Assessment:**
- {ai1.name} plans to {ai1.current_plan.primary_objective.value}
- {ai2.name} aims for {ai2.current_plan.primary_objective.value}
- Key matchup: This will be decided by {self._predict_key_battle_factor(army1, army2)}
"""
        return narrative
    
    def _describe_army_deployment(self, army: Army) -> str:
        """Describe how an army is deployed"""
        deployment = []
        for unit in army.units[:5]:  # Show key units
            deployment.append(f"• {unit.current_models}x {unit.name}: {unit.points_cost} pts")
        
        return "\n".join(deployment)
    
    def _predict_key_battle_factor(self, army1: Army, army2: Army) -> str:
        """Predict what will decide the battle"""
        factors = ["superior positioning", "magical supremacy", "cavalry charges", 
                  "artillery effectiveness", "infantry grind", "tactical flexibility"]
        return random.choice(factors)
    
    def narrate_turn(self, turn: int, game_state: GameState, ai_decisions: List[Tuple[str, str]]) -> str:
        """Generate turn narrative with tactical commentary"""
        narrative = f"""
⚔️ TURN {turn} - {game_state.active_player.upper()}
======================================

**Tactical Situation:**
{self._assess_tactical_situation(game_state)}

**Command Decisions:**
"""
        for action, reasoning in ai_decisions:
            narrative += f"• {action}: {reasoning}\n"
        
        narrative += f"""
**Battle Results:**
{self._describe_turn_results(game_state)}

**Victory Points:** {game_state.victory_points['player1']} - {game_state.victory_points['player2']}
"""
        return narrative
    
    def _assess_tactical_situation(self, game_state: GameState) -> str:
        """Assess and describe the current tactical situation"""
        situations = [
            "The battle hangs in the balance as both armies maneuver for advantage",
            "Momentum is building as units close for decisive combat",
            "The opening moves reveal each commander's strategic intent",
            "Critical decisions must be made as casualties mount",
            "The tide of battle shifts as new threats emerge"
        ]
        return random.choice(situations)
    
    def _describe_turn_results(self, game_state: GameState) -> str:
        """Describe what happened during the turn"""
        if game_state.battle_log:
            latest = game_state.battle_log[-1]
            return f"Casualties: {latest.get('casualties', 'Light')}, Momentum: {latest.get('momentum', 'Unchanged')}"
        return "Maneuvering and positioning continues"
    
    def generate_final_report(self, winner: str, final_score: Dict[str, int], 
                            ai1: TacticalAI, ai2: TacticalAI) -> str:
        """Generate comprehensive final battle report"""
        return f"""
🏆 BATTLE CONCLUSION
==========================================

**Victory:** {winner}
**Final Score:** {final_score['player1']} - {final_score['player2']}

**Post-Battle Analysis:**

**{ai1.name}'s Performance:**
- Strategy: {ai1.current_plan.primary_objective.value}
- Key Decisions: {len(ai1.decision_log)} tactical choices made
- Lessons Learned: {self._generate_lessons_learned(ai1)}

**{ai2.name}'s Performance:**  
- Strategy: {ai2.current_plan.primary_objective.value}
- Key Decisions: {len(ai2.decision_log)} tactical choices made
- Lessons Learned: {self._generate_lessons_learned(ai2)}

**Battle Summary:**
{self._create_battle_summary()}

**Tactical Notes:**
{chr(10).join(f"• {note}" for note in self.tactical_notes[-5:])}
"""
    
    def _generate_lessons_learned(self, ai: TacticalAI) -> str:
        """Generate lessons learned for AI improvement"""
        lessons = [
            "Coordination between units could be improved",
            "Artillery positioning was effective",
            "Cavalry charges need better timing",
            "Magic phase requires more focus",
            "Deployment strategy worked well"
        ]
        return random.choice(lessons)
    
    def _create_battle_summary(self) -> str:
        """Create overall battle summary"""
        return "A closely fought engagement that demonstrated the importance of tactical flexibility and combined arms coordination."

# =============================================================================
# MAIN BATTLE SYSTEM
# =============================================================================

class WarhammerBattleSystem:
    """Complete Warhammer battle system with tactical AI"""
    
    def __init__(self):
        self.game_state = None
        self.narrator = BattleNarrator()
        self.battle_history = []
    
    def create_sample_armies(self) -> Tuple[Army, Army]:
        """Create sample armies for testing"""
        # Empire Army
        empire_general = Unit(
            name="Empire General",
            unit_type=UnitType.CHARACTER,
            stats=UnitStats(4, 6, 6, 4, 4, 3, 6, 3, 9, 3, special_rules=["General"]),
            equipment=Equipment("Great Weapon", "Full Plate", True),
            models=1,
            current_models=1,
            formation=Formation.SKIRMISH,
            points_cost=150
        )
        
        empire_infantry = Unit(
            name="Empire State Troops",
            unit_type=UnitType.INFANTRY,
            stats=UnitStats(4, 3, 3, 3, 3, 1, 3, 1, 7, 5),
            equipment=Equipment("Spear", "Light Armor", True, ["Spear Wall"]),
            models=20,
            current_models=20,
            formation=Formation.CLOSE_ORDER,
            points_cost=260
        )
        
        empire_cavalry = Unit(
            name="Empire Knights",
            unit_type=UnitType.CAVALRY,
            stats=UnitStats(8, 4, 3, 3, 3, 1, 3, 1, 8, 4),
            equipment=Equipment("Lance", "Heavy Armor", True, ["Cavalry", "Lance"]),
            models=5,
            current_models=5,
            formation=Formation.CLOSE_ORDER,
            points_cost=125
        )
        
        empire_artillery = Unit(
            name="Great Cannon",
            unit_type=UnitType.WAR_MACHINE,
            stats=UnitStats(3, 4, 4, 3, 3, 1, 3, 1, 7, 7),
            equipment=Equipment("Cannon", special_rules=["Artillery"]),
            models=3,
            current_models=3,
            formation=Formation.SKIRMISH,
            points_cost=100
        )
        
        empire_army = Army(
            name="Nuln Regiment",
            faction="Empire",
            units=[empire_general, empire_infantry, empire_cavalry, empire_artillery],
            points_total=635,
            general=empire_general
        )
        
        # Orc Army
        orc_warboss = Unit(
            name="Orc Warboss",
            unit_type=UnitType.CHARACTER,
            stats=UnitStats(4, 5, 3, 5, 5, 3, 3, 3, 8, 4, special_rules=["General", "Fear"]),
            equipment=Equipment("Great Weapon", "Heavy Armor"),
            models=1,
            current_models=1,
            formation=Formation.SKIRMISH,
            points_cost=120
        )
        
        orc_boys = Unit(
            name="Orc Boys",
            unit_type=UnitType.INFANTRY,
            stats=UnitStats(4, 3, 3, 3, 4, 1, 2, 1, 7, 6),
            equipment=Equipment("Choppa", special_rules=["Frenzy"]),
            models=25,
            current_models=25,
            formation=Formation.CLOSE_ORDER,
            points_cost=200
        )
        
        orc_cavalry = Unit(
            name="Boar Boyz",
            unit_type=UnitType.CAVALRY,
            stats=UnitStats(7, 3, 3, 3, 4, 1, 2, 1, 7, 6),
            equipment=Equipment("Spear", special_rules=["Cavalry", "Frenzy"]),
            models=5,
            current_models=5,
            formation=Formation.CLOSE_ORDER,
            points_cost=110
        )
        
        orc_trolls = Unit(
            name="River Trolls",
            unit_type=UnitType.MONSTER,
            stats=UnitStats(6, 3, 1, 5, 4, 3, 1, 3, 4, 5, special_rules=["Fear", "Regeneration"]),
            equipment=Equipment("Claws and Fangs"),
            models=3,
            current_models=3,
            formation=Formation.SKIRMISH,
            points_cost=180
        )
        
        orc_army = Army(
            name="Grimjaw Tribe",
            faction="Orcs & Goblins",
            units=[orc_warboss, orc_boys, orc_cavalry, orc_trolls],
            points_total=610,
            general=orc_warboss
        )
        
        return empire_army, orc_army
    
    def run_tactical_battle(self, army1: Army, army2: Army, max_turns: int = 6) -> Dict[str, Any]:
        """Run a complete tactical battle with AI commanders"""
        print("🏰 WARHAMMER: THE OLD WORLD - TACTICAL BATTLE SYSTEM")
        print("=" * 60)
        
        # Create battlefield
        battlefield = Battlefield()
        battlefield.add_terrain(Terrain("Hill", (24, 20), (12, 8), "Hill"))
        battlefield.add_terrain(Terrain("Forest", (48, 15), (8, 10), "Forest"))
        
        # Initialize AI commanders
        ai1 = TacticalAI(f"{army1.faction} Commander", army1, experience_level=3)
        ai2 = TacticalAI(f"{army2.faction} Commander", army2, experience_level=3)
        
        # Create battle plans
        plan1 = ai1.create_battle_plan(army2, battlefield)
        plan2 = ai2.create_battle_plan(army1, battlefield)
        
        # Initialize game state
        self.game_state = GameState(
            turn=1,
            phase="Deployment",
            active_player="player1",
            army1=army1,
            army2=army2,
            battlefield=battlefield
        )
        
        # Deployment narrative
        deployment_story = self.narrator.narrate_deployment(army1, army2, ai1, ai2)
        print(deployment_story)
        
        # Battle turns
        battle_result = {
            "winner": None,
            "final_score": {"player1": 0, "player2": 0},
            "turns_played": 0,
            "narrative": [deployment_story],
            "ai_performance": {
                "ai1_decisions": len(ai1.decision_log),
                "ai2_decisions": len(ai2.decision_log)
            }
        }
        
        for turn in range(1, max_turns + 1):
            print(f"\n⚔️ TURN {turn}")
            print("-" * 30)
            
            self.game_state.turn = turn
            
            # Player 1 turn
            self.game_state.active_player = "player1"
            turn_decisions = self._execute_turn(ai1, ["Advance", "Shoot", "Charge", "Magic"])
            turn_narrative = self.narrator.narrate_turn(turn, self.game_state, turn_decisions)
            battle_result["narrative"].append(turn_narrative)
            print(turn_narrative)
            
            # Player 2 turn  
            self.game_state.active_player = "player2"
            turn_decisions = self._execute_turn(ai2, ["Advance", "Shoot", "Charge", "Magic"])
            turn_narrative = self.narrator.narrate_turn(turn, self.game_state, turn_decisions)
            battle_result["narrative"].append(turn_narrative)
            print(turn_narrative)
            
            # Update battle state
            self._update_battle_state()
            battle_result["turns_played"] = turn
            
            # Check victory conditions
            if self._check_victory_conditions():
                break
        
        # Determine winner
        if self.game_state.victory_points["player1"] > self.game_state.victory_points["player2"]:
            winner = f"{army1.name} ({army1.faction})"
        elif self.game_state.victory_points["player2"] > self.game_state.victory_points["player1"]:
            winner = f"{army2.name} ({army2.faction})"
        else:
            winner = "Draw"
        
        battle_result["winner"] = winner
        battle_result["final_score"] = self.game_state.victory_points.copy()
        
        # Final report
        final_report = self.narrator.generate_final_report(
            winner, battle_result["final_score"], ai1, ai2
        )
        battle_result["narrative"].append(final_report)
        print(final_report)
        
        # Save battle for AI learning
        self._save_battle_for_learning(battle_result, ai1, ai2)
        
        return battle_result
    
    def _execute_turn(self, ai: TacticalAI, available_actions: List[str]) -> List[Tuple[str, str]]:
        """Execute a complete turn for an AI player"""
        decisions = []
        
        phases = ["Movement", "Magic", "Shooting", "Combat"]
        for phase in phases:
            self.game_state.phase = phase
            action, reasoning = ai.make_tactical_decision(self.game_state, available_actions)
            decisions.append((action, reasoning))
            
            # Simulate phase effects
            self._simulate_phase_effects(phase, action)
        
        return decisions
    
    def _simulate_phase_effects(self, phase: str, action: str):
        """Simulate effects of actions in each phase"""
        if phase == "Shooting" and "fire" in action.lower():
            # Simulate shooting casualties
            casualty_roll = random.randint(0, 2)
            if casualty_roll > 0:
                self._apply_casualties(casualty_roll)
        
        elif phase == "Combat" and "charge" in action.lower():
            # Simulate combat casualties
            casualty_roll = random.randint(1, 3)
            self._apply_casualties(casualty_roll)
        
        # Add to battle log
        self.game_state.battle_log.append({
            "phase": phase,
            "action": action,
            "turn": self.game_state.turn,
            "player": self.game_state.active_player
        })
    
    def _apply_casualties(self, casualties: int):
        """Apply casualties to random units"""
        active_army = self.game_state.army1 if self.game_state.active_player == "player1" else self.game_state.army2
        enemy_army = self.game_state.army2 if self.game_state.active_player == "player1" else self.game_state.army1
        
        # Apply casualties to enemy
        alive_units = [u for u in enemy_army.units if u.is_alive() and u.current_models > 1]
        if alive_units:
            target = random.choice(alive_units)
            target.current_models = max(0, target.current_models - casualties)
            
            # Award victory points
            player_key = self.game_state.active_player
            self.game_state.victory_points[player_key] += casualties * 10
    
    def _update_battle_state(self):
        """Update overall battle state"""
        # Remove destroyed units
        self.game_state.army1.units = [u for u in self.game_state.army1.units if u.is_alive()]
        self.game_state.army2.units = [u for u in self.game_state.army2.units if u.is_alive()]
    
    def _check_victory_conditions(self) -> bool:
        """Check if battle should end"""
        # Army broken check
        if self.game_state.army1.is_broken() or self.game_state.army2.is_broken():
            return True
        
        # Massive victory point difference
        vp_diff = abs(self.game_state.victory_points["player1"] - self.game_state.victory_points["player2"])
        return vp_diff > 500
    
    def _save_battle_for_learning(self, battle_result: Dict[str, Any], ai1: TacticalAI, ai2: TacticalAI):
        """Save battle data for AI learning and improvement"""
        learning_data = {
            "timestamp": datetime.now().isoformat(),
            "battle_result": battle_result,
            "ai1_decisions": ai1.decision_log,
            "ai2_decisions": ai2.decision_log,
            "tactical_lessons": {
                "winning_strategies": [],
                "failed_tactics": [],
                "key_decisions": []
            }
        }
        
        # Analyze decisions for learning
        winner_ai = ai1 if "player1" in battle_result["winner"] else ai2
        for decision in winner_ai.decision_log:
            learning_data["tactical_lessons"]["winning_strategies"].append({
                "situation": decision["situation"],
                "successful_action": decision["action"],
                "reasoning": decision["reasoning"]
            })
        
        self.battle_history.append(learning_data)
        
        # Save to file
        filename = f"tactical_battle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(learning_data, f, indent=2, default=str)
        
        print(f"📊 Battle data saved to {filename} for AI learning")

if __name__ == "__main__":
    # Run a demonstration battle
    battle_system = WarhammerBattleSystem()
    empire_army, orc_army = battle_system.create_sample_armies()
    
    print("🎯 Starting Tactical AI Battle Demonstration...")
    result = battle_system.run_tactical_battle(empire_army, orc_army, max_turns=4)
    
    print(f"\n🏆 Battle Complete!")
    print(f"Winner: {result['winner']}")
    print(f"Final Score: {result['final_score']}")
    print(f"Turns Played: {result['turns_played']}")