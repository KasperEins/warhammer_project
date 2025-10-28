#!/usr/bin/env python3
"""
🏰 EPIC WARHAMMER TACTICAL BATTLE SYSTEM
=======================================

This system generates battle reports that match the quality and detail
of human Warhammer battle reports, including:

- Detailed army composition analysis
- Pre-battle strategic assessment  
- Turn-by-turn tactical decision making
- Actual dice mechanics and combat resolution
- Post-battle analysis and lessons learned

The AI commanders understand real Warhammer tactics and generate
narrative battle reports like experienced human players.
"""

from advanced_warhammer_ai import *
import random
from datetime import datetime

class EpicBattleSystem:
    """Complete battle system with human-quality AI commanders"""
    
    def __init__(self):
        self.narrator = BattleNarrator()
        self.combat_resolver = CombatResolver()
        
    def create_empire_army(self) -> Army:
        """Create a detailed Empire army"""
        
        # Empire General on Griffon
        general = Unit(
            name="General Karl von Steiner",
            unit_type=UnitType.CHARACTER,
            models=1, current_models=1, points_cost=280,
            movement=8, weapon_skill=6, ballistic_skill=5, strength=4,
            toughness=4, wounds=3, initiative=6, attacks=3, leadership=9,
            weapons=[Weapon("Runefang", WeaponType.GREAT_WEAPON, strength_bonus=2, special_rules=["Magic Weapon"])],
            armor_save=3, ward_save=5,
            special_rules=["General", "Flying", "Terror", "Large Target"]
        )
        
        # Empire Battle Standard Bearer
        bsb = Unit(
            name="Captain Heinrich Zimmer",
            unit_type=UnitType.CHARACTER,
            models=1, current_models=1, points_cost=120,
            movement=4, weapon_skill=5, ballistic_skill=4, strength=4,
            toughness=4, wounds=2, initiative=5, attacks=2, leadership=8,
            weapons=[Weapon("Great Weapon", WeaponType.GREAT_WEAPON, strength_bonus=2)],
            armor_save=4, ward_save=6,
            special_rules=["Battle Standard Bearer"]
        )
        
        # Empire State Troops with Spears
        state_troops = Unit(
            name="Nuln State Troops",
            unit_type=UnitType.INFANTRY,
            models=25, current_models=25, points_cost=300,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=7,
            weapons=[Weapon("Spear", WeaponType.SPEAR, special_rules=["Fight in Extra Rank"])],
            armor_save=5,
            special_rules=["Spear Wall"]
        )
        
        # Empire Knights
        knights = Unit(
            name="Knights of the Blazing Sun",
            unit_type=UnitType.CAVALRY,
            models=8, current_models=8, points_cost=240,
            movement=8, weapon_skill=4, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=8,
            weapons=[Weapon("Lance", WeaponType.LANCE, strength_bonus=2, special_rules=["Cavalry"])],
            armor_save=4,
            special_rules=["Fast Cavalry", "Impact Hits"]
        )
        
        # Handgunners
        handgunners = Unit(
            name="Nuln Handgunners",
            unit_type=UnitType.INFANTRY,
            models=15, current_models=15, points_cost=180,
            movement=4, weapon_skill=3, ballistic_skill=4, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=7,
            weapons=[Weapon("Handgun", WeaponType.CROSSBOW, range_inches=24, special_rules=["Armor Piercing"])],
            armor_save=6,
            special_rules=["Stand and Shoot"]
        )
        
        # Great Cannon
        cannon = Unit(
            name="Great Cannon of Nuln",
            unit_type=UnitType.WAR_MACHINE,
            models=4, current_models=4, points_cost=120,
            movement=3, weapon_skill=4, ballistic_skill=4, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=7,
            weapons=[Weapon("Great Cannon", WeaponType.CANNON, range_inches=48, special_rules=["Artillery"])],
            armor_save=7,
            special_rules=["War Machine"]
        )
        
        # Helblaster Volley Gun
        helblaster = Unit(
            name="Helblaster Volley Gun",
            unit_type=UnitType.WAR_MACHINE,
            models=3, current_models=3, points_cost=110,
            movement=3, weapon_skill=4, ballistic_skill=4, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=7,
            weapons=[Weapon("Volley Gun", WeaponType.CROSSBOW, range_inches=24, special_rules=["Multiple Shots"])],
            armor_save=7,
            special_rules=["War Machine", "Multiple Barrels"]
        )
        
        empire_army = Army(
            name="Grand Army of Nuln",
            faction="Empire",
            units=[general, bsb, state_troops, knights, handgunners, cannon, helblaster],
            general=general,
            battle_standard=bsb
        )
        
        return empire_army
    
    def create_orc_army(self) -> Army:
        """Create a detailed Orc & Goblin army"""
        
        # Orc Warboss on Wyvern
        warboss = Unit(
            name="Warboss Grimjaw Smasher",
            unit_type=UnitType.CHARACTER,
            models=1, current_models=1, points_cost=320,
            movement=9, weapon_skill=6, ballistic_skill=3, strength=5,
            toughness=5, wounds=3, initiative=3, attacks=4, leadership=8,
            weapons=[Weapon("Choppa of Doom", WeaponType.GREAT_WEAPON, strength_bonus=2)],
            armor_save=4, ward_save=6,
            special_rules=["General", "Flying", "Terror", "Frenzy"]
        )
        
        # Orc Big Boss
        big_boss = Unit(
            name="Big Boss Skarfang",
            unit_type=UnitType.CHARACTER,
            models=1, current_models=1, points_cost=100,
            movement=4, weapon_skill=5, ballistic_skill=3, strength=4,
            toughness=5, wounds=2, initiative=3, attacks=3, leadership=8,
            weapons=[Weapon("Great Weapon", WeaponType.GREAT_WEAPON, strength_bonus=2)],
            armor_save=5,
            special_rules=["Battle Standard Bearer", "Frenzy"]
        )
        
        # Large Orc Boys Unit
        orc_boys = Unit(
            name="Grimjaw's Boyz",
            unit_type=UnitType.INFANTRY,
            models=30, current_models=30, points_cost=240,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=4, wounds=1, initiative=2, attacks=1, leadership=7,
            weapons=[Weapon("Choppa", WeaponType.HAND_WEAPON, special_rules=["Extra Attack"])],
            armor_save=6,
            special_rules=["Frenzy", "Mob Rule"]
        )
        
        # Orc Boar Riders
        boar_riders = Unit(
            name="Tusker Cavalry",
            unit_type=UnitType.CAVALRY,
            models=6, current_models=6, points_cost=180,
            movement=7, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=4, wounds=1, initiative=2, attacks=1, leadership=7,
            weapons=[Weapon("Spear", WeaponType.SPEAR)],
            armor_save=6,
            special_rules=["Fast Cavalry", "Frenzy"]
        )
        
        # River Trolls
        trolls1 = Unit(
            name="River Trolls",
            unit_type=UnitType.MONSTER,
            models=3, current_models=3, points_cost=150,
            movement=6, weapon_skill=3, ballistic_skill=1, strength=5,
            toughness=4, wounds=3, initiative=1, attacks=3, leadership=4,
            weapons=[Weapon("Claws and Fangs", WeaponType.HAND_WEAPON)],
            armor_save=5, ward_save=5,
            special_rules=["Fear", "Regeneration", "Stupidity"]
        )
        
        # Stone Trolls
        trolls2 = Unit(
            name="Stone Trolls",
            unit_type=UnitType.MONSTER,
            models=3, current_models=3, points_cost=180,
            movement=6, weapon_skill=3, ballistic_skill=1, strength=5,
            toughness=5, wounds=3, initiative=1, attacks=3, leadership=4,
            weapons=[Weapon("Stone Fists", WeaponType.HAND_WEAPON)],
            armor_save=4, ward_save=5,
            special_rules=["Fear", "Regeneration", "Stupidity"]
        )
        
        # Goblin Wolf Riders (Fast Harassment)
        wolf_riders = Unit(
            name="Goblin Wolf Riders",
            unit_type=UnitType.CAVALRY,
            models=10, current_models=10, points_cost=120,
            movement=9, weapon_skill=2, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, initiative=2, attacks=1, leadership=6,
            weapons=[Weapon("Spear", WeaponType.SPEAR), Weapon("Bow", WeaponType.BOW, range_inches=24)],
            armor_save=6,
            special_rules=["Fast Cavalry", "Hit and Run"]
        )
        
        orc_army = Army(
            name="Grimjaw's Warband",
            faction="Orcs & Goblins",
            units=[warboss, big_boss, orc_boys, boar_riders, trolls1, trolls2, wolf_riders],
            general=warboss,
            battle_standard=big_boss
        )
        
        return orc_army
    
    def run_epic_battle(self, army1: Army, army2: Army, max_turns: int = 6) -> Dict[str, Any]:
        """Run complete epic battle with human-quality reporting"""
        
        print("🏰 WARHAMMER: THE OLD WORLD - EPIC TACTICAL BATTLE")
        print("=" * 60)
        print("⚡ Initializing Advanced AI Commanders...")
        
        # Create AI commanders
        ai1 = AdvancedTacticalAI(f"{army1.faction} Marshal", army1, experience=8)
        ai2 = AdvancedTacticalAI(f"{army2.faction} Warlord", army2, experience=8)
        
        # Analyze enemies and create battle plans
        print("🧠 Analyzing enemy forces...")
        ai1.create_battle_plan(army2)
        ai2.create_battle_plan(army1)
        
        # Generate pre-battle report
        print("📋 Generating pre-battle analysis...")
        pre_battle_report = self.narrator.generate_pre_battle_report(army1, army2, ai1, ai2)
        print(pre_battle_report)
        
        # Initialize battle state
        battle_state = {
            "turn": 1,
            "victory_points": {"player1": 0, "player2": 0},
            "battle_events": [],
            "combat_log": [],
            "tactical_notes": []
        }
        
        battle_narrative = [pre_battle_report]
        
        # Battle turns
        for turn in range(1, max_turns + 1):
            print(f"\n⚔️ TURN {turn}")
            print("=" * 30)
            
            battle_state["turn"] = turn
            
            # Player 1 Turn
            print(f"🔵 {army1.faction} Turn {turn}")
            turn_result1 = self._execute_army_turn(army1, army2, ai1, battle_state)
            turn_narrative1 = self.narrator.narrate_turn(turn, "player1", army1, 
                                                       turn_result1["decisions"], 
                                                       turn_result1["combat_results"])
            battle_narrative.append(turn_narrative1)
            print(turn_narrative1)
            
            # Update battle state
            battle_state["victory_points"]["player1"] += turn_result1["victory_points"]
            
            # Check for victory
            if self._check_battle_end(army1, army2, battle_state):
                break
            
            # Player 2 Turn  
            print(f"🟢 {army2.faction} Turn {turn}")
            turn_result2 = self._execute_army_turn(army2, army1, ai2, battle_state)
            turn_narrative2 = self.narrator.narrate_turn(turn, "player2", army2,
                                                       turn_result2["decisions"],
                                                       turn_result2["combat_results"])
            battle_narrative.append(turn_narrative2)
            print(turn_narrative2)
            
            # Update battle state
            battle_state["victory_points"]["player2"] += turn_result2["victory_points"]
            
            # Check for victory
            if self._check_battle_end(army1, army2, battle_state):
                break
        
        # Determine winner and generate final report
        winner, battle_summary = self._determine_winner(army1, army2, battle_state)
        final_report = self.narrator.generate_final_report(
            winner, battle_state["victory_points"], battle_summary, ai1, ai2
        )
        
        battle_narrative.append(final_report)
        print(final_report)
        
        # Save complete battle report
        complete_report = "\n".join(battle_narrative)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epic_battle_report_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(complete_report)
        
        print(f"\n📄 Complete battle report saved to: {filename}")
        
        return {
            "winner": winner,
            "final_vp": battle_state["victory_points"],
            "turns_played": battle_state["turn"],
            "narrative": battle_narrative,
            "filename": filename,
            "battle_summary": battle_summary,
            "ai_performance": {
                "ai1_decisions": len(ai1.decision_history),
                "ai2_decisions": len(ai2.decision_history)
            }
        }
    
    def _execute_army_turn(self, active_army: Army, enemy_army: Army, 
                          ai: AdvancedTacticalAI, battle_state: Dict) -> Dict[str, Any]:
        """Execute a complete army turn with all phases"""
        
        turn_decisions = []
        combat_results = []
        victory_points = 0
        
        # Current battlefield assessment
        battlefield_state = {
            "turn": battle_state["turn"],
            "my_strength": sum(u.unit_strength() for u in active_army.units if u.is_alive()),
            "enemy_strength": sum(u.unit_strength() for u in enemy_army.units if u.is_alive()),
            "casualties_ratio": self._calculate_casualty_ratio(active_army),
            "terrain_advantage": "neutral",
            "magic_dominance": "neutral"
        }
        
        # Execute each phase
        phases = ["Movement", "Magic", "Shooting", "Combat"]
        
        for phase in phases:
            # AI makes tactical decision
            decision, reasoning = ai.make_phase_decision(
                phase, battlefield_state, 
                self._get_phase_options(phase, active_army, enemy_army)
            )
            
            # Execute phase and get detailed results
            phase_details, phase_combat = self._execute_phase(
                phase, decision, active_army, enemy_army, battlefield_state
            )
            
            # Record decision and results
            turn_decisions.append((decision, reasoning, phase_details))
            if phase_combat:
                combat_results.extend(phase_combat)
            
            # Update AI decision history
            ai.decision_history.append({
                "turn": battle_state["turn"],
                "phase": phase,
                "decision": decision,
                "reasoning": reasoning,
                "battlefield_state": battlefield_state.copy()
            })
        
        # Calculate victory points earned this turn
        victory_points = self._calculate_turn_victory_points(active_army, enemy_army, combat_results)
        
        return {
            "decisions": turn_decisions,
            "combat_results": combat_results,
            "victory_points": victory_points
        }
    
    def _get_phase_options(self, phase: str, army: Army, enemy: Army) -> List[str]:
        """Get available options for each phase"""
        if phase == "Movement":
            return ["Aggressive advance", "Careful positioning", "Tactical withdrawal", "Flanking maneuver"]
        elif phase == "Magic":
            return ["Offensive magic", "Protective spells", "Dispel enemy magic", "Magic missiles"]
        elif phase == "Shooting":
            return ["Focus fire on priority target", "Defensive shooting", "Opportunity targets", "Artillery bombardment"]
        elif phase == "Combat":
            return ["Press combat advantage", "Steady combat", "Fighting withdrawal", "Coordinated charge"]
        return ["Standard action"]
    
    def _execute_phase(self, phase: str, decision: str, army: Army, enemy: Army, 
                      battlefield_state: Dict) -> Tuple[str, List[Dict]]:
        """Execute phase with detailed mechanics"""
        
        details = f"Executing {decision.lower()}"
        combat_results = []
        
        if phase == "Shooting" and "fire" in decision.lower():
            # Simulate shooting combat
            shooter = random.choice([u for u in army.units if u.is_alive() and 
                                   any(w.range_inches > 0 for w in u.weapons)])
            target = random.choice([u for u in enemy.units if u.is_alive()])
            
            if shooter and target:
                combat_result = self._resolve_shooting_combat(shooter, target)
                combat_results.append(combat_result)
                details += f" - {shooter.name} targets {target.name}"
        
        elif phase == "Combat" and "charge" in decision.lower():
            # Simulate melee combat
            attacker = random.choice([u for u in army.units if u.is_alive() and 
                                    u.unit_type in [UnitType.INFANTRY, UnitType.CAVALRY]])
            defender = random.choice([u for u in enemy.units if u.is_alive()])
            
            if attacker and defender:
                combat_result = self._resolve_melee_combat(attacker, defender)
                combat_results.append(combat_result)
                details += f" - {attacker.name} charges {defender.name}"
        
        elif phase == "Magic":
            # Simulate magic effects
            if "offensive" in decision.lower():
                caster = next((u for u in army.units if u.magic and u.is_alive()), None)
                target = random.choice([u for u in enemy.units if u.is_alive()])
                
                if caster and target:
                    magic_result = self._resolve_magic_attack(caster, target)
                    combat_results.append(magic_result)
                    details += f" - {caster.name} casts spell at {target.name}"
        
        return details, combat_results
    
    def _resolve_shooting_combat(self, shooter: Unit, target: Unit) -> Dict[str, Any]:
        """Resolve detailed shooting combat with dice"""
        
        # Determine number of shots
        num_shots = shooter.current_models
        if "Multiple Shots" in shooter.special_rules:
            num_shots *= 3
        
        # To hit rolls
        hits, hit_details = self.combat_resolver.roll_to_hit(shooter, target, num_shots)
        
        # To wound rolls
        strength = shooter.strength
        wounds, wound_details = self.combat_resolver.roll_to_wound(strength, target.toughness, hits)
        
        # Save rolls
        final_wounds, save_details = self.combat_resolver.roll_saves(
            target.armor_save, target.ward_save, wounds
        )
        
        # Apply casualties
        models_killed = min(final_wounds, target.current_models)
        target.current_models -= models_killed
        
        return {
            "type": "shooting",
            "attacker": shooter.name,
            "defender": target.name,
            "hit_details": hit_details[0] if hit_details else "",
            "wound_details": wound_details[0] if wound_details else "",
            "save_details": "; ".join(save_details),
            "final_wounds": final_wounds,
            "models_killed": models_killed
        }
    
    def _resolve_melee_combat(self, attacker: Unit, defender: Unit) -> Dict[str, Any]:
        """Resolve detailed melee combat with dice"""
        
        # Calculate attacks
        num_attacks = attacker.total_attacks()
        
        # To hit rolls
        hits, hit_details = self.combat_resolver.roll_to_hit(attacker, defender, num_attacks)
        
        # To wound rolls  
        strength = attacker.strength
        if any(w.weapon_type == WeaponType.GREAT_WEAPON for w in attacker.weapons):
            strength += 2
        
        wounds, wound_details = self.combat_resolver.roll_to_wound(strength, defender.toughness, hits)
        
        # Save rolls
        final_wounds, save_details = self.combat_resolver.roll_saves(
            defender.armor_save, defender.ward_save, wounds
        )
        
        # Apply casualties
        models_killed = min(final_wounds, defender.current_models)
        defender.current_models -= models_killed
        
        return {
            "type": "melee",
            "attacker": attacker.name,
            "defender": defender.name,
            "hit_details": hit_details[0] if hit_details else "",
            "wound_details": wound_details[0] if wound_details else "",
            "save_details": "; ".join(save_details),
            "final_wounds": final_wounds,
            "models_killed": models_killed
        }
    
    def _resolve_magic_attack(self, caster: Unit, target: Unit) -> Dict[str, Any]:
        """Resolve magic attack"""
        
        # Simplified magic - direct damage
        magic_strength = 4 + (caster.magic.level if caster.magic else 0)
        magic_hits = random.randint(1, 3)
        
        wounds, wound_details = self.combat_resolver.roll_to_wound(
            magic_strength, target.toughness, magic_hits
        )
        
        # No armor saves vs magic, only ward saves
        final_wounds, save_details = self.combat_resolver.roll_saves(
            7, target.ward_save, wounds
        )
        
        models_killed = min(final_wounds, target.current_models)
        target.current_models -= models_killed
        
        return {
            "type": "magic",
            "attacker": caster.name,
            "defender": target.name,
            "hit_details": f"Magic hits: {magic_hits}",
            "wound_details": wound_details[0] if wound_details else "",
            "save_details": "; ".join(save_details),
            "final_wounds": final_wounds,
            "models_killed": models_killed
        }
    
    def _calculate_casualty_ratio(self, army: Army) -> float:
        """Calculate casualty ratio for army"""
        original_models = sum(u.models for u in army.units)
        current_models = sum(u.current_models for u in army.units)
        return (original_models - current_models) / original_models if original_models > 0 else 0
    
    def _calculate_turn_victory_points(self, army: Army, enemy: Army, 
                                     combat_results: List[Dict]) -> int:
        """Calculate victory points earned this turn"""
        vp = 0
        for combat in combat_results:
            if "models_killed" in combat:
                vp += combat["models_killed"] * 10  # 10 VP per model killed
        return vp
    
    def _check_battle_end(self, army1: Army, army2: Army, battle_state: Dict) -> bool:
        """Check if battle should end"""
        # Army broken (25% remaining)
        if army1.is_broken() or army2.is_broken():
            return True
        
        # Massive VP difference
        vp_diff = abs(battle_state["victory_points"]["player1"] - 
                     battle_state["victory_points"]["player2"])
        if vp_diff > 800:
            return True
        
        return False
    
    def _determine_winner(self, army1: Army, army2: Army, battle_state: Dict) -> Tuple[str, str]:
        """Determine battle winner and generate summary"""
        
        vp1 = battle_state["victory_points"]["player1"]
        vp2 = battle_state["victory_points"]["player2"]
        
        if vp1 > vp2 * 1.5:
            victory_level = "Crushing Victory"
        elif vp1 > vp2 * 1.2:
            victory_level = "Solid Victory"
        elif vp1 > vp2:
            victory_level = "Minor Victory"
        elif vp2 > vp1 * 1.5:
            victory_level = "Crushing Defeat"
        elif vp2 > vp1 * 1.2:
            victory_level = "Solid Defeat"
        elif vp2 > vp1:
            victory_level = "Minor Defeat"
        else:
            victory_level = "Draw"
        
        if vp1 > vp2:
            winner = f"{army1.name} ({army1.faction}) - {victory_level}"
            summary = f"The {army1.faction} forces proved superior through {random.choice(['tactical superiority', 'superior firepower', 'better positioning', 'coordinated assault'])}."
        elif vp2 > vp1:
            winner = f"{army2.name} ({army2.faction}) - {victory_level}"
            summary = f"The {army2.faction} warband dominated through {random.choice(['aggressive tactics', 'overwhelming assault', 'strategic flexibility', 'combat prowess'])}."
        else:
            winner = "Honorable Draw"
            summary = "Both armies fought with distinction, neither able to gain decisive advantage."
        
        return winner, summary

if __name__ == "__main__":
    print("🎯 Initializing Epic Tactical Battle System...")
    
    battle_system = EpicBattleSystem()
    
    print("🏗️ Creating detailed armies...")
    empire_army = battle_system.create_empire_army()
    orc_army = battle_system.create_orc_army()
    
    print(f"✅ Empire Army: {empire_army.total_points} points")
    print(f"✅ Orc Army: {orc_army.total_points} points")
    
    print("\n🎲 Beginning Epic Battle...")
    result = battle_system.run_epic_battle(empire_army, orc_army, max_turns=5)
    
    print(f"\n🏆 BATTLE COMPLETED!")
    print(f"📊 Winner: {result['winner']}")
    print(f"📈 Final VP: {result['final_vp']['player1']} - {result['final_vp']['player2']}")
    print(f"⏰ Turns: {result['turns_played']}")
    print(f"📄 Report saved: {result['filename']}")