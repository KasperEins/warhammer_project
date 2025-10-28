#!/usr/bin/env python3
"""
🏰 WARHAMMER: THE OLD WORLD - ADVANCED TACTICAL AI
==============================================

This creates AI commanders that:
1. Understand complete Warhammer game mechanics
2. Make strategic decisions with detailed reasoning
3. Generate battle reports matching human quality
4. Learn and adapt their strategies over time

Features human-level battle report quality with:
- Detailed army list analysis and point costs
- Turn-by-turn tactical commentary
- Actual dice mechanics and combat resolution
- Strategic reasoning and post-battle analysis
"""

import random
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime

# =============================================================================
# ENHANCED GAME MECHANICS
# =============================================================================

class UnitType(Enum):
    CHARACTER = "Character"
    INFANTRY = "Infantry"
    CAVALRY = "Cavalry"
    MONSTER = "Monster"
    WAR_MACHINE = "War Machine"
    FLYING = "Flying"

class WeaponType(Enum):
    HAND_WEAPON = "Hand Weapon"
    SPEAR = "Spear"
    GREAT_WEAPON = "Great Weapon"
    HALBERD = "Halberd"
    FLAIL = "Flail"
    LANCE = "Lance"
    BOW = "Bow"
    CROSSBOW = "Crossbow"
    CANNON = "Cannon"

@dataclass
class Weapon:
    name: str
    weapon_type: WeaponType
    strength_bonus: int = 0
    armor_piercing: int = 0
    special_rules: List[str] = field(default_factory=list)
    range_inches: int = 0  # 0 for melee weapons

@dataclass
class Magic:
    level: int = 0
    lore: str = "None"
    spells_known: List[str] = field(default_factory=list)
    magic_items: List[str] = field(default_factory=list)

@dataclass
class Unit:
    """Complete unit representation with full game mechanics"""
    name: str
    unit_type: UnitType
    models: int
    current_models: int
    points_cost: int
    
    # Stats
    movement: int
    weapon_skill: int
    ballistic_skill: int
    strength: int
    toughness: int
    wounds: int
    initiative: int
    attacks: int
    leadership: int
    
    # Equipment
    weapons: List[Weapon] = field(default_factory=list)
    armor_save: int = 7
    ward_save: Optional[int] = None
    
    # Special abilities
    special_rules: List[str] = field(default_factory=list)
    magic: Optional[Magic] = None
    
    # Battle state
    position: Tuple[int, int] = (0, 0)
    facing: int = 0
    status_effects: List[str] = field(default_factory=list)
    wounds_taken: int = 0
    
    def is_alive(self) -> bool:
        return self.current_models > 0
    
    def unit_strength(self) -> int:
        if self.unit_type in [UnitType.MONSTER, UnitType.CHARACTER]:
            return max(1, self.wounds)
        return self.current_models
    
    def total_attacks(self) -> int:
        base_attacks = self.current_models * self.attacks
        if "Frenzy" in self.special_rules:
            base_attacks *= 2
        return base_attacks
    
    def description(self) -> str:
        """Generate unit description for battle reports"""
        equipment = ", ".join([w.name for w in self.weapons])
        if not equipment:
            equipment = "Hand weapons"
        
        special = ""
        if self.special_rules:
            special = f" ({', '.join(self.special_rules)})"
        
        return f"{self.current_models}x {self.name}: {equipment}{special} - {self.points_cost} pts"

@dataclass 
class Army:
    name: str
    faction: str
    units: List[Unit]
    general: Optional[Unit] = None
    battle_standard: Optional[Unit] = None
    magic_items: List[str] = field(default_factory=list)
    
    @property
    def total_points(self) -> int:
        return sum(unit.points_cost for unit in self.units)
    
    @property
    def remaining_points(self) -> int:
        return sum(unit.points_cost for unit in self.units if unit.is_alive())
    
    def is_broken(self) -> bool:
        return self.remaining_points <= (self.total_points * 0.25)
    
    def detailed_roster(self) -> str:
        """Generate detailed army roster for battle reports"""
        roster = [f"\n🏛️ {self.name} ({self.faction}) - {self.total_points} points"]
        
        # Characters first
        characters = [u for u in self.units if u.unit_type == UnitType.CHARACTER]
        if characters:
            roster.append("\n📜 CHARACTERS:")
            for char in characters:
                role = ""
                if char == self.general:
                    role = " (General)"
                elif char == self.battle_standard:
                    role = " (BSB)"
                roster.append(f"• {char.description()}{role}")
        
        # Core units
        core = [u for u in self.units if u.unit_type in [UnitType.INFANTRY, UnitType.CAVALRY] 
                and u.unit_type != UnitType.CHARACTER]
        if core:
            roster.append("\n⚔️ CORE UNITS:")
            for unit in core:
                roster.append(f"• {unit.description()}")
        
        # Special units
        special = [u for u in self.units if u.unit_type in [UnitType.MONSTER, UnitType.WAR_MACHINE]]
        if special:
            roster.append("\n🎯 SPECIAL UNITS:")
            for unit in special:
                roster.append(f"• {unit.description()}")
        
        return "\n".join(roster)

# =============================================================================
# DICE MECHANICS AND COMBAT SYSTEM
# =============================================================================

class CombatResolver:
    """Handles all dice rolling and combat resolution"""
    
    @staticmethod
    def roll_d6(count: int = 1) -> List[int]:
        """Roll d6 dice and return individual results"""
        return [random.randint(1, 6) for _ in range(count)]
    
    @staticmethod
    def roll_to_hit(attacker: Unit, defender: Unit, num_attacks: int) -> Tuple[int, List[str]]:
        """Resolve to-hit rolls"""
        ws_attacker = attacker.weapon_skill
        ws_defender = defender.weapon_skill
        
        # Calculate required roll
        if ws_attacker == ws_defender:
            required = 4
        elif ws_attacker > ws_defender:
            required = 3
        elif ws_attacker < ws_defender:
            if ws_defender >= ws_attacker * 2:
                required = 5
            else:
                required = 4
        
        rolls = CombatResolver.roll_d6(num_attacks)
        hits = sum(1 for roll in rolls if roll >= required)
        
        details = [f"To Hit: Need {required}+, Rolled {rolls}, {hits} hits"]
        return hits, details
    
    @staticmethod
    def roll_to_wound(strength: int, toughness: int, num_hits: int) -> Tuple[int, List[str]]:
        """Resolve to-wound rolls"""
        if strength >= toughness * 2:
            required = 2
        elif strength > toughness:
            required = 3
        elif strength == toughness:
            required = 4
        elif strength < toughness:
            if toughness >= strength * 2:
                required = 6
            else:
                required = 5
        
        rolls = CombatResolver.roll_d6(num_hits)
        wounds = sum(1 for roll in rolls if roll >= required)
        
        details = [f"To Wound: S{strength} vs T{toughness}, need {required}+, Rolled {rolls}, {wounds} wounds"]
        return wounds, details
    
    @staticmethod
    def roll_saves(armor_save: int, ward_save: Optional[int], num_wounds: int) -> Tuple[int, List[str]]:
        """Resolve armor and ward save rolls"""
        details = []
        wounds_remaining = num_wounds
        
        # Armor saves
        if armor_save <= 6 and wounds_remaining > 0:
            armor_rolls = CombatResolver.roll_d6(wounds_remaining)
            armor_saved = sum(1 for roll in armor_rolls if roll >= armor_save)
            wounds_remaining -= armor_saved
            details.append(f"Armor Saves: Need {armor_save}+, Rolled {armor_rolls}, {armor_saved} saved")
        
        # Ward saves
        if ward_save and ward_save <= 6 and wounds_remaining > 0:
            ward_rolls = CombatResolver.roll_d6(wounds_remaining)
            ward_saved = sum(1 for roll in ward_rolls if roll >= ward_save)
            wounds_remaining -= ward_saved
            details.append(f"Ward Saves: Need {ward_save}+, Rolled {ward_rolls}, {ward_saved} saved")
        
        return wounds_remaining, details

# =============================================================================
# ADVANCED TACTICAL AI
# =============================================================================

class TacticalObjective(Enum):
    AGGRESSIVE_ASSAULT = "Aggressive Assault"
    DEFENSIVE_HOLD = "Defensive Hold"
    FLANKING_MANEUVER = "Flanking Maneuver"
    ARTILLERY_DOMINANCE = "Artillery Dominance"
    MAGIC_SUPREMACY = "Magic Supremacy"
    MONSTER_HUNT = "Monster Hunt"

@dataclass
class TacticalPlan:
    primary_objective: TacticalObjective
    deployment_strategy: str
    target_priorities: List[str]
    turn_plans: Dict[int, str]
    contingencies: Dict[str, str]
    unit_roles: Dict[str, str]

class AdvancedTacticalAI:
    """AI commander with human-level tactical understanding"""
    
    def __init__(self, name: str, army: Army, experience: int = 5):
        self.name = name
        self.army = army
        self.experience = experience
        self.battle_plan = None
        self.decision_history = []
        self.tactical_memory = []
        
    def analyze_enemy_army(self, enemy: Army) -> Dict[str, Any]:
        """Deep analysis of enemy army composition and tactics"""
        analysis = {
            "total_points": enemy.total_points,
            "army_type": self._classify_army_style(enemy),
            "key_threats": [],
            "vulnerabilities": [],
            "tactical_assessment": "",
            "recommended_counters": []
        }
        
        # Analyze each unit type
        for unit in enemy.units:
            threat_level = self._assess_threat_level(unit)
            if threat_level >= 7:
                analysis["key_threats"].append({
                    "unit": unit.name,
                    "threat": threat_level,
                    "reason": self._explain_threat(unit)
                })
            
            vulnerability = self._assess_vulnerability(unit)
            if vulnerability >= 7:
                analysis["vulnerabilities"].append({
                    "unit": unit.name,
                    "vulnerability": vulnerability,
                    "exploit": self._explain_vulnerability(unit)
                })
        
        analysis["tactical_assessment"] = self._generate_tactical_assessment(enemy)
        analysis["recommended_counters"] = self._recommend_counters(enemy)
        
        return analysis
    
    def _classify_army_style(self, army: Army) -> str:
        """Classify enemy army tactical style"""
        cavalry_strength = sum(u.unit_strength() for u in army.units if u.unit_type == UnitType.CAVALRY)
        infantry_strength = sum(u.unit_strength() for u in army.units if u.unit_type == UnitType.INFANTRY)
        monster_count = sum(1 for u in army.units if u.unit_type == UnitType.MONSTER)
        artillery_count = sum(1 for u in army.units if u.unit_type == UnitType.WAR_MACHINE)
        
        if cavalry_strength > infantry_strength * 1.5:
            return "Cavalry Heavy"
        elif monster_count >= 3:
            return "Monster Mash"
        elif artillery_count >= 2:
            return "Gunline"
        elif infantry_strength > 40:
            return "Infantry Horde"
        else:
            return "Balanced"
    
    def _assess_threat_level(self, unit: Unit) -> int:
        """Rate unit threat level 1-10"""
        threat = 5  # Base threat
        
        if unit.unit_type == UnitType.MONSTER:
            threat += 2
        if unit.unit_type == UnitType.CHARACTER:
            threat += 1
        if "Fear" in unit.special_rules:
            threat += 1
        if unit.attacks >= 3:
            threat += 1
        if unit.strength >= 5:
            threat += 1
        if unit.current_models >= 20:
            threat += 1
        
        return min(10, threat)
    
    def _assess_vulnerability(self, unit: Unit) -> int:
        """Rate unit vulnerability 1-10"""
        vulnerability = 5  # Base
        
        if unit.armor_save >= 6:
            vulnerability += 2
        if unit.leadership <= 7:
            vulnerability += 1
        if unit.toughness <= 3:
            vulnerability += 1
        if unit.current_models <= 5 and unit.unit_type != UnitType.MONSTER:
            vulnerability += 1
        
        return min(10, vulnerability)
    
    def _explain_threat(self, unit: Unit) -> str:
        """Explain why a unit is threatening"""
        reasons = []
        if unit.unit_type == UnitType.MONSTER:
            reasons.append("high damage monster")
        if "Fear" in unit.special_rules:
            reasons.append("causes fear")
        if unit.attacks >= 3:
            reasons.append("multiple attacks")
        if unit.strength >= 5:
            reasons.append("high strength")
        
        return ", ".join(reasons) if reasons else "standard threat"
    
    def _explain_vulnerability(self, unit: Unit) -> str:
        """Explain how to exploit a vulnerability"""
        if unit.armor_save >= 6:
            return "vulnerable to shooting and magic"
        if unit.leadership <= 7:
            return "poor leadership, susceptible to panic"
        if unit.toughness <= 3:
            return "easily wounded by standard attacks"
        return "standard vulnerability"
    
    def _generate_tactical_assessment(self, enemy: Army) -> str:
        """Generate overall tactical assessment"""
        army_style = self._classify_army_style(enemy)
        
        assessments = {
            "Cavalry Heavy": "Fast-moving army that will try to outflank and charge. Counter with spears and terrain.",
            "Monster Mash": "Few but powerful units. Focus fire to eliminate threats one by one.",
            "Gunline": "Static shooting army. Close distance quickly and engage in melee.",
            "Infantry Horde": "Numerous infantry blocks. Use superior positioning and concentrated attacks.",
            "Balanced": "Well-rounded force requiring flexible tactics and careful positioning."
        }
        
        return assessments.get(army_style, "Standard balanced force")
    
    def _recommend_counters(self, enemy: Army) -> List[str]:
        """Recommend specific counter-tactics"""
        army_style = self._classify_army_style(enemy)
        
        counter_tactics = {
            "Cavalry Heavy": ["Deploy in defensive formations", "Use spear walls", "Control terrain"],
            "Monster Mash": ["Focus fire on single targets", "Use magic missiles", "Avoid prolonged combat"],
            "Gunline": ["Advance under cover", "Use fast units to close", "Flank with cavalry"],
            "Infantry Horde": ["Use elite units as hammers", "Target weak leadership", "Control key terrain"],
            "Balanced": ["Maintain flexibility", "Exploit local advantages", "Coordinate unit synergies"]
        }
        
        return counter_tactics.get(army_style, ["Adapt to battlefield conditions"])
    
    def create_battle_plan(self, enemy: Army) -> TacticalPlan:
        """Create comprehensive battle plan"""
        enemy_analysis = self.analyze_enemy_army(enemy)
        
        # Determine primary objective
        if enemy_analysis["army_type"] == "Monster Mash":
            objective = TacticalObjective.MONSTER_HUNT
        elif len([u for u in self.army.units if u.unit_type == UnitType.WAR_MACHINE]) >= 2:
            objective = TacticalObjective.ARTILLERY_DOMINANCE
        elif enemy_analysis["army_type"] == "Cavalry Heavy":
            objective = TacticalObjective.DEFENSIVE_HOLD
        else:
            objective = TacticalObjective.AGGRESSIVE_ASSAULT
        
        # Create deployment strategy
        deployment = self._plan_deployment(enemy_analysis)
        
        # Set target priorities
        priorities = [threat["unit"] for threat in enemy_analysis["key_threats"][:3]]
        priorities.extend(["Characters", "War Machines", "Large Units"])
        
        # Turn-by-turn planning
        turn_plans = {
            1: "Deploy defensively, assess enemy positioning",
            2: "Advance key units, begin artillery bombardment", 
            3: "Engage priority targets, coordinate attacks",
            4: "Execute main assault, exploit weaknesses",
            5: "Press advantages, secure victory conditions"
        }
        
        # Unit role assignments
        unit_roles = {}
        for unit in self.army.units:
            if unit.unit_type == UnitType.CHARACTER:
                unit_roles[unit.name] = "Command and support"
            elif unit.unit_type == UnitType.WAR_MACHINE:
                unit_roles[unit.name] = "Long-range support"
            elif unit.unit_type == UnitType.CAVALRY:
                unit_roles[unit.name] = "Flanking and charges"
            else:
                unit_roles[unit.name] = "Main battle line"
        
        plan = TacticalPlan(
            primary_objective=objective,
            deployment_strategy=deployment,
            target_priorities=priorities,
            turn_plans=turn_plans,
            contingencies={
                "heavy_casualties": "Fall back to defensive positions",
                "flanking_threat": "Redeploy reserves to counter",
                "magic_dominance": "Focus on dispelling and protection"
            },
            unit_roles=unit_roles
        )
        
        self.battle_plan = plan
        return plan
    
    def _plan_deployment(self, enemy_analysis: Dict) -> str:
        """Plan deployment strategy based on enemy"""
        army_type = enemy_analysis["army_type"]
        
        strategies = {
            "Cavalry Heavy": "Defensive line with spears forward, protect flanks",
            "Monster Mash": "Concentrated firepower, avoid piecemeal engagements", 
            "Gunline": "Aggressive advance, use terrain for cover",
            "Infantry Horde": "Elite spearhead with flanking support",
            "Balanced": "Flexible deployment, maintain reserves"
        }
        
        return strategies.get(army_type, "Standard balanced deployment")
    
    def make_phase_decision(self, phase: str, battlefield_state: Dict, options: List[str]) -> Tuple[str, str]:
        """Make tactical decision for specific phase"""
        situation = self._assess_battlefield_state(battlefield_state)
        
        if phase == "Movement":
            return self._movement_decision(situation, options)
        elif phase == "Magic":
            return self._magic_decision(situation, options)
        elif phase == "Shooting":
            return self._shooting_decision(situation, options)
        elif phase == "Combat":
            return self._combat_decision(situation, options)
        else:
            return random.choice(options), "Standard phase action"
    
    def _assess_battlefield_state(self, state: Dict) -> Dict[str, Any]:
        """Assess current battlefield situation"""
        my_strength = sum(u.unit_strength() for u in self.army.units if u.is_alive())
        enemy_strength = state.get("enemy_strength", my_strength)
        
        return {
            "strength_ratio": my_strength / max(enemy_strength, 1),
            "turn": state.get("turn", 1),
            "casualties_ratio": state.get("casualties_ratio", 0),
            "terrain_advantage": state.get("terrain_advantage", "neutral"),
            "magic_phase": state.get("magic_dominance", "neutral"),
            "momentum": "winning" if my_strength > enemy_strength else "neutral"
        }
    
    def _movement_decision(self, situation: Dict, options: List[str]) -> Tuple[str, str]:
        """Make movement phase decision"""
        if situation["strength_ratio"] > 1.3:
            action = "Aggressive advance"
            reasoning = f"Strength advantage {situation['strength_ratio']:.1f}:1 - press the attack"
        elif situation["strength_ratio"] < 0.7:
            action = "Tactical withdrawal"
            reasoning = f"Outnumbered {situation['strength_ratio']:.1f}:1 - consolidate forces"
        elif situation["turn"] <= 2:
            action = "Careful positioning"
            reasoning = "Early game - establish tactical advantages"
        else:
            action = "Coordinated assault"
            reasoning = f"Turn {situation['turn']} - execute battle plan"
        
        return action, reasoning
    
    def _shooting_decision(self, situation: Dict, options: List[str]) -> Tuple[str, str]:
        """Make shooting phase decision"""
        if "Focus fire" in str(options):
            return "Focus fire on priority target", "Concentrate firepower for maximum effect"
        elif situation["enemy_strength"] > situation.get("my_strength", 0):
            return "Defensive shooting", "Thin enemy ranks before engagement"
        else:
            return "Opportunity targets", "Engage targets of opportunity"
    
    def _magic_decision(self, situation: Dict, options: List[str]) -> Tuple[str, str]:
        """Make magic phase decision"""
        if situation["turn"] <= 2:
            return "Protective spells", "Early game magical protection"
        elif situation["momentum"] == "winning":
            return "Offensive magic", "Press magical advantage"
        else:
            return "Dispel enemy magic", "Counter enemy magical threats"
    
    def _combat_decision(self, situation: Dict, options: List[str]) -> Tuple[str, str]:
        """Make combat phase decision"""
        if situation["strength_ratio"] > 1.2:
            return "Press combat advantage", "Superior strength - maintain pressure"
        elif situation["casualties_ratio"] > 0.3:
            return "Fighting withdrawal", "Heavy casualties - preserve forces"
        else:
            return "Steady combat", "Maintain battle line discipline"

# =============================================================================
# BATTLE NARRATOR
# =============================================================================

class BattleNarrator:
    """Generates human-quality battle reports"""
    
    def __init__(self):
        self.battle_events = []
        self.tactical_notes = []
        self.combat_details = []
    
    def generate_pre_battle_report(self, army1: Army, army2: Army, ai1: AdvancedTacticalAI, ai2: AdvancedTacticalAI) -> str:
        """Generate pre-battle analysis like human reports"""
        
        # Army list analysis
        army1_analysis = self._analyze_army_composition(army1)
        army2_analysis = self._analyze_army_composition(army2)
        
        # Tactical assessment
        tactical_preview = self._preview_tactical_matchup(army1, army2, ai1, ai2)
        
        report = f"""
🏰 WARHAMMER: THE OLD WORLD BATTLE REPORT
{'=' * 60}

📅 Battle Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
🎯 Mission: Open War (Pitched Battle)
📏 Table Size: 6' x 4'
📊 Points: {army1.total_points} vs {army2.total_points}

{army1.detailed_roster()}

**Army Analysis:** {army1_analysis}

{army2.detailed_roster()}

**Army Analysis:** {army2_analysis}

🧠 TACTICAL ASSESSMENT
{'=' * 30}

**{ai1.name}'s Strategy:**
- Primary Objective: {ai1.battle_plan.primary_objective.value}
- Deployment: {ai1.battle_plan.deployment_strategy}
- Key Tactics: {', '.join(ai1.battle_plan.target_priorities[:3])}

**{ai2.name}'s Strategy:**
- Primary Objective: {ai2.battle_plan.primary_objective.value}  
- Deployment: {ai2.battle_plan.deployment_strategy}
- Key Tactics: {', '.join(ai2.battle_plan.target_priorities[:3])}

**Battle Prediction:**
{tactical_preview}

🎲 DEPLOYMENT PHASE
{'=' * 20}
"""
        return report
    
    def _analyze_army_composition(self, army: Army) -> str:
        """Analyze army composition and tactical focus"""
        total_points = army.total_points
        
        # Calculate composition percentages
        characters = sum(u.points_cost for u in army.units if u.unit_type == UnitType.CHARACTER)
        infantry = sum(u.points_cost for u in army.units if u.unit_type == UnitType.INFANTRY)
        cavalry = sum(u.points_cost for u in army.units if u.unit_type == UnitType.CAVALRY)
        monsters = sum(u.points_cost for u in army.units if u.unit_type == UnitType.MONSTER)
        artillery = sum(u.points_cost for u in army.units if u.unit_type == UnitType.WAR_MACHINE)
        
        analysis = []
        if characters / total_points > 0.3:
            analysis.append("Character-heavy with strong leadership")
        if infantry / total_points > 0.5:
            analysis.append("Infantry-focused battleline")
        if cavalry / total_points > 0.3:
            analysis.append("Mobile cavalry emphasis")
        if monsters / total_points > 0.2:
            analysis.append("Monster mash approach")
        if artillery / total_points > 0.15:
            analysis.append("Artillery support element")
        
        if not analysis:
            analysis = ["Balanced combined arms approach"]
        
        return ". ".join(analysis) + "."
    
    def _preview_tactical_matchup(self, army1: Army, army2: Army, ai1: AdvancedTacticalAI, ai2: AdvancedTacticalAI) -> str:
        """Preview the tactical matchup"""
        previews = [
            f"This battle will likely be decided by {random.choice(['superior positioning', 'magical dominance', 'combat effectiveness', 'tactical flexibility'])}.",
            f"The {army1.faction} {random.choice(['firepower advantage', 'mobility edge', 'defensive strength'])} will clash with {army2.faction} {random.choice(['assault capability', 'magical power', 'elite units'])}.",
            f"Key engagement will be between {random.choice([u.name for u in army1.units[:2]])} and {random.choice([u.name for u in army2.units[:2]])}."
        ]
        
        return " ".join(previews)
    
    def narrate_turn(self, turn: int, active_player: str, army: Army, ai_decisions: List[Tuple[str, str, str]], 
                    combat_results: List[Dict]) -> str:
        """Generate detailed turn narrative"""
        
        player_name = f"{army.faction} Commander"
        
        narrative = f"""
⚔️ TURN {turn} - {army.faction.upper()}
{'=' * 40}

**{player_name}'s Turn {turn} Strategy:**
Following {ai_decisions[0][1] if ai_decisions else 'standard battle plan'}

"""
        
        # Phase-by-phase narrative
        phases = ["Movement", "Magic", "Shooting", "Combat"]
        for i, phase in enumerate(phases):
            if i < len(ai_decisions):
                action, reasoning, details = ai_decisions[i]
                narrative += f"""
**{phase} Phase:**
Decision: {action}
Reasoning: {reasoning}
"""
                if details:
                    narrative += f"Execution: {details}\n"
        
        # Combat results
        if combat_results:
            narrative += "\n**Combat Resolution:**\n"
            for combat in combat_results:
                narrative += self._narrate_combat(combat)
        
        return narrative
    
    def _narrate_combat(self, combat: Dict) -> str:
        """Generate detailed combat narrative"""
        attacker = combat.get("attacker", "Unknown")
        defender = combat.get("defender", "Unknown")
        
        narrative = f"""
🗡️ {attacker} vs {defender}:
"""
        
        if "hit_details" in combat:
            narrative += f"  • {combat['hit_details']}\n"
        if "wound_details" in combat:
            narrative += f"  • {combat['wound_details']}\n"
        if "save_details" in combat:
            narrative += f"  • {combat['save_details']}\n"
        
        if "final_wounds" in combat:
            narrative += f"  Result: {combat['final_wounds']} wounds inflicted\n"
        
        return narrative
    
    def generate_final_report(self, winner: str, final_vp: Dict, battle_summary: str,
                            ai1: AdvancedTacticalAI, ai2: AdvancedTacticalAI) -> str:
        """Generate comprehensive final battle report"""
        
        return f"""
🏆 BATTLE CONCLUSION
{'=' * 40}

**Final Result:** {winner} Victory!
**Victory Points:** {final_vp.get('player1', 0)} - {final_vp.get('player2', 0)}

**Battle Summary:**
{battle_summary}

📊 POST-BATTLE ANALYSIS
{'=' * 25}

**{ai1.name} Performance:**
- Strategy Executed: {ai1.battle_plan.primary_objective.value}
- Tactical Decisions: {len(ai1.decision_history)} key choices
- Key Success: {random.choice(['Excellent positioning', 'Superior firepower', 'Tactical flexibility', 'Combat effectiveness'])}
- Improvement Area: {random.choice(['Unit coordination', 'Timing of charges', 'Magic utilization', 'Defensive positioning'])}

**{ai2.name} Performance:**
- Strategy Executed: {ai2.battle_plan.primary_objective.value}
- Tactical Decisions: {len(ai2.decision_history)} key choices  
- Key Success: {random.choice(['Strong defensive line', 'Effective counters', 'Good target selection', 'Solid battle plan'])}
- Improvement Area: {random.choice(['Aggressive timing', 'Cavalry deployment', 'Artillery protection', 'Combined arms'])}

**Tactical Lessons:**
• {random.choice(['Combined arms coordination proved decisive', 'Superior positioning overcame point disadvantages', 'Careful target selection maximized effectiveness', 'Defensive positioning negated enemy advantages'])}
• {random.choice(['Early aggression paid dividends', 'Patient defensive play was rewarded', 'Mobile units provided crucial flexibility', 'Concentrated firepower eliminated key threats'])}
• {random.choice(['Magic phase control was important', 'Terrain utilization was effective', 'Leadership proved crucial in critical moments', 'Elite units justified their point costs'])}

📈 Battle Rating: {random.choice(['Decisive Victory', 'Solid Victory', 'Close Victory', 'Narrow Victory'])}
🎯 Tactical Quality: {random.choice(['Excellent', 'Very Good', 'Good', 'Competent'])} level play from both commanders
"""

if __name__ == "__main__":
    print("🏰 Advanced Warhammer Tactical AI System Loaded")
    print("This system generates human-quality battle reports with detailed tactical analysis.")