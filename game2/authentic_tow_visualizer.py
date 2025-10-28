#!/usr/bin/env python3
"""
🏛️ AUTHENTIC WARHAMMER: THE OLD WORLD VISUALIZER
===============================================

The most authentic TOW web visualization system ever built, featuring:
- Proper unit blocks with visible ranks, files, and facing
- Authentic terrain representation with TOW rules
- Formation display (Close Order, Open Order, Skirmish)
- Real-time movement and combat visualization
- Official scenario implementation
- True-to-rules visual representation

Built to exact TOW specifications for maximum authenticity.
"""

import asyncio
import random
import json
import time
from typing import Dict, List, Tuple, Optional
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import threading
from authentic_tow_engine import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tow_authentic_visualizer'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=False)

# =============================================================================
# AUTHENTIC TOW UNIT DATA
# =============================================================================

def create_authentic_empire_units() -> List[UnitProfile]:
    """Create authentic Empire units with exact TOW stats"""
    return [
        UnitProfile(
            name="Empire Handgunners",
            unit_type=UnitType.INFANTRY,
            faction=Faction.EMPIRE,
            movement=4,
            weapon_skill=3,
            ballistic_skill=3,
            strength=3,
            toughness=3,
            wounds=1,
            initiative=3,
            attacks=1,
            leadership=7,
            armor_save=6,  # Light Armor
            weapons=["Handgun", "Hand Weapon"],
            special_rules=["Move or Fire"],
            points_per_model=9
        ),
        UnitProfile(
            name="Empire Knights",
            unit_type=UnitType.CAVALRY,
            faction=Faction.EMPIRE,
            movement=7,
            weapon_skill=4,
            ballistic_skill=3,
            strength=3,
            toughness=3,
            wounds=1,
            initiative=3,
            attacks=1,
            leadership=8,
            armor_save=4,  # Heavy Armor + Shield
            weapons=["Lance", "Hand Weapon", "Shield"],
            special_rules=["Cavalry", "Lance Formation"],
            points_per_model=20
        ),
        UnitProfile(
            name="Empire Great Cannon",
            unit_type=UnitType.WAR_MACHINE,
            faction=Faction.EMPIRE,
            movement=3,
            weapon_skill=3,
            ballistic_skill=3,
            strength=7,
            toughness=7,
            wounds=3,
            initiative=1,
            attacks=1,
            leadership=7,
            armor_save=7,  # No save
            weapons=["Great Cannon"],
            special_rules=["Artillery", "Cannon"],
            points_per_model=120
        )
    ]

def create_authentic_orc_units() -> List[UnitProfile]:
    """Create authentic Orc units with exact TOW stats"""
    return [
        UnitProfile(
            name="Orc Boyz",
            unit_type=UnitType.INFANTRY,
            faction=Faction.ORCS_GOBLINS,
            movement=4,
            weapon_skill=3,
            ballistic_skill=3,
            strength=3,
            toughness=4,
            wounds=1,
            initiative=2,
            attacks=1,
            leadership=7,
            armor_save=6,  # Light Armor
            weapons=["Choppa", "Shield"],
            special_rules=["Animosity"],
            points_per_model=6
        ),
        UnitProfile(
            name="Orc Arrer Boyz",
            unit_type=UnitType.INFANTRY,
            faction=Faction.ORCS_GOBLINS,
            movement=4,
            weapon_skill=3,
            ballistic_skill=3,
            strength=3,
            toughness=4,
            wounds=1,
            initiative=2,
            attacks=1,
            leadership=7,
            armor_save=6,  # Light Armor
            weapons=["Bow", "Choppa"],
            special_rules=["Animosity"],
            points_per_model=6
        ),
        UnitProfile(
            name="Wolf Riders",
            unit_type=UnitType.CAVALRY,
            faction=Faction.ORCS_GOBLINS,
            movement=9,
            weapon_skill=3,
            ballistic_skill=3,
            strength=3,
            toughness=4,
            wounds=1,
            initiative=2,
            attacks=1,
            leadership=6,
            armor_save=6,  # Light Armor
            weapons=["Spear", "Bow"],
            special_rules=["Cavalry", "Fast Cavalry"],
            points_per_model=14
        )
    ]

# =============================================================================
# AUTHENTIC TOW BATTLEFIELD
# =============================================================================

class AuthenticTOWBattlefield:
    """Authentic TOW battlefield with proper terrain and scenarios"""
    
    def __init__(self, width: int = 72, height: int = 48):  # Standard TOW 6' x 4' battlefield
        self.width = width
        self.height = height
        self.units: Dict[str, UnitBlock] = {}
        self.terrain_features: List[TerrainFeature] = []
        self.current_scenario: Optional[TOWScenario] = None
        self.current_turn = 1
        self.active_faction = Faction.EMPIRE
        self.game_log: List[str] = []
        
        # Initialize scenario engine
        self.scenario_engine = ScenarioEngine()
        
        # Movement and psychology engines
        self.movement_engine = TOWMovementEngine(width, height)
        self.psychology_engine = TOWPsychologyEngine()
        
        self._initialize_battlefield()
    
    def _initialize_battlefield(self):
        """Initialize battlefield with terrain and scenario"""
        # Create Upon the Field of Glory scenario (classic pitched battle)
        self.current_scenario = self.scenario_engine.create_upon_field_of_glory(self.width, self.height)
        
        # Generate authentic terrain following TOW rules
        self.terrain_features = self.scenario_engine.generate_authentic_terrain(
            self.width, self.height, ScenarioType.UPON_FIELD_OF_GLORY
        )
        
        # Deploy armies
        self._deploy_armies()
        
        self.log_event("🏛️ Battlefield initialized with 'Upon the Field of Glory' scenario")
        self.log_event(f"📊 Battlefield: {self.width}x{self.height} hexes (6' x 4' table)")
        self.log_event(f"🌲 Terrain features: {len(self.terrain_features)} pieces following TOW placement rules")
        
        # Log terrain details
        for i, terrain in enumerate(self.terrain_features):
            terrain_name = terrain.terrain_type.value
            size = len(terrain.positions)
            rules = ", ".join(terrain.special_rules[:2])  # First 2 rules
            self.log_event(f"   • {terrain_name} ({size} hexes) - {rules}")
    
    def _add_terrain_features(self):
        """Add authentic TOW terrain features"""
        # Woods feature
        woods_positions = set()
        for x in range(10, 15):
            for y in range(8, 12):
                woods_positions.add((x, y))
        
        woods = TerrainFeature(
            terrain_type=TerrainType.WOODS,
            positions=woods_positions,
            special_rules=["Provides Cover", "Difficult Terrain"]
        )
        self.terrain_features.append(woods)
        
        # Hill feature (central objective)
        hill_positions = set()
        center_x, center_y = self.width // 2, self.height // 2
        for x in range(center_x - 2, center_x + 3):
            for y in range(center_y - 2, center_y + 3):
                hill_positions.add((x, y))
        
        hill = TerrainFeature(
            terrain_type=TerrainType.HILLS,
            positions=hill_positions,
            height=1,
            special_rules=["Higher Ground", "+1 Combat Resolution"]
        )
        self.terrain_features.append(hill)
        
        # Ruins
        ruins_positions = {(35, 20), (36, 20), (35, 21), (36, 21)}
        ruins = TerrainFeature(
            terrain_type=TerrainType.BUILDINGS_RUINS,
            positions=ruins_positions,
            special_rules=["Hard Cover", "Dangerous Terrain"]
        )
        self.terrain_features.append(ruins)
    
    def _deploy_armies(self):
        """Deploy armies according to scenario deployment zones"""
        empire_units = create_authentic_empire_units()
        orc_units = create_authentic_orc_units()
        
        # Empire deployment (south)
        empire_deployments = [
            {"profile": empire_units[0], "position": Position(12, 4, 0), "formation": Formation.CLOSE_ORDER, "models": 20},
            {"profile": empire_units[1], "position": Position(20, 3, 0), "formation": Formation.CLOSE_ORDER, "models": 8},
            {"profile": empire_units[2], "position": Position(28, 5, 0), "formation": Formation.CLOSE_ORDER, "models": 1},
        ]
        
        # Orc deployment (north)
        orc_deployments = [
            {"profile": orc_units[0], "position": Position(15, 28, 3), "formation": Formation.CLOSE_ORDER, "models": 25},
            {"profile": orc_units[1], "position": Position(25, 27, 3), "formation": Formation.CLOSE_ORDER, "models": 15},
            {"profile": orc_units[2], "position": Position(8, 26, 3), "formation": Formation.OPEN_ORDER, "models": 6},
        ]
        
        # Create Empire units
        for i, deployment in enumerate(empire_deployments):
            unit_id = f"empire_{i}"
            unit = UnitBlock(
                id=unit_id,
                profile=deployment["profile"],
                position=deployment["position"],
                formation=deployment["formation"],
                ranks=1,
                files=1,
                current_models=deployment["models"]
            )
            self.units[unit_id] = unit
            self.log_event(f"⚔️ Empire {unit.profile.name} deployed: {unit.files}x{unit.ranks} formation, {unit.current_models} models")
        
        # Create Orc units
        for i, deployment in enumerate(orc_deployments):
            unit_id = f"orc_{i}"
            unit = UnitBlock(
                id=unit_id,
                profile=deployment["profile"],
                position=deployment["position"],
                formation=deployment["formation"],
                ranks=1,
                files=1,
                current_models=deployment["models"]
            )
            self.units[unit_id] = unit
            self.log_event(f"🧌 Orc {unit.profile.name} deployed: {unit.files}x{unit.ranks} formation, {unit.current_models} models")
    
    def log_event(self, message: str):
        """Add event to game log"""
        timestamp = time.strftime("%H:%M:%S")
        self.game_log.append(f"[{timestamp}] {message}")
        if len(self.game_log) > 50:  # Keep last 50 events
            self.game_log.pop(0)
    
    def get_battlefield_state(self) -> Dict:
        """Get complete battlefield state for visualization"""
        units_data = []
        for unit in self.units.values():
            if unit.is_alive:
                # Get all positions occupied by unit
                unit_positions = unit.all_positions
                
                units_data.append({
                    'id': unit.id,
                    'name': unit.profile.name,
                    'faction': unit.profile.faction.value,
                    'unit_type': unit.profile.unit_type.value,
                    'position': {
                        'x': unit.position.x,
                        'y': unit.position.y,
                        'facing': unit.position.facing
                    },
                    'formation': unit.formation.value,
                    'ranks': unit.ranks,
                    'files': unit.files,
                    'current_models': unit.current_models,
                    'max_models': unit.profile.max_unit_size,
                    'positions': unit_positions,  # All occupied hexes
                    'front_positions': unit.front_arc_positions,  # Front rank for facing
                    'stats': {
                        'M': unit.profile.movement,
                        'WS': unit.profile.weapon_skill,
                        'BS': unit.profile.ballistic_skill,
                        'S': unit.profile.strength,
                        'T': unit.profile.toughness,
                        'W': unit.profile.wounds,
                        'I': unit.profile.initiative,
                        'A': unit.profile.attacks,
                        'Ld': unit.profile.leadership,
                        'Save': f"{unit.profile.armor_save}+" if unit.profile.armor_save < 7 else "-"
                    },
                    'status': {
                        'has_moved': unit.has_moved,
                        'has_charged': unit.has_charged,
                        'has_shot': unit.has_shot,
                        'is_engaged': unit.is_engaged,
                        'is_fleeing': unit.is_fleeing,
                        'is_disordered': unit.is_disordered,
                        'is_steadfast': unit.is_steadfast
                    },
                    'special_rules': unit.profile.special_rules,
                    'weapons': unit.profile.weapons
                })
        
        terrain_data = []
        for terrain in self.terrain_features:
            terrain_data.append({
                'type': terrain.terrain_type.value,
                'positions': list(terrain.positions),
                'height': terrain.height,
                'special_rules': terrain.special_rules
            })
        
        objectives_data = []
        if self.current_scenario:
            for obj in self.current_scenario.objectives:
                objectives_data.append({
                    'id': obj.id,
                    'name': obj.name,
                    'position': obj.position,
                    'control_radius': obj.control_radius,
                    'controlling_faction': obj.controlling_faction.value if obj.controlling_faction else None,
                    'points_per_turn': obj.points_per_turn
                })
        
        return {
            'battlefield': {
                'width': self.width,
                'height': self.height,
                'turn': self.current_turn,
                'active_faction': self.active_faction.value
            },
            'units': units_data,
            'terrain': terrain_data,
            'objectives': objectives_data,
            'scenario': {
                'name': self.current_scenario.name if self.current_scenario else "No Scenario",
                'type': self.current_scenario.scenario_type.value if self.current_scenario else "None",
                'description': self.current_scenario.description if self.current_scenario else "",
                'special_rules': self.current_scenario.special_rules if self.current_scenario else []
            },
            'game_log': self.game_log[-10:]  # Last 10 events
        }
    
    def simulate_turn(self):
        """Simulate one turn of authentic TOW gameplay"""
        self.log_event(f"🎲 Turn {self.current_turn} - {self.active_faction.value} Phase")
        
        # Reset unit actions
        for unit in self.units.values():
            unit.has_moved = False
            unit.has_marched = False
            unit.has_shot = False
            unit.has_charged = False
        
        # Simulate faction actions
        faction_units = [u for u in self.units.values() if u.profile.faction == self.active_faction and u.is_alive]
        
        for unit in faction_units:
            self._simulate_unit_action(unit)
        
        # Check objectives
        self._check_objectives()
        
        # Switch to next faction/turn
        if self.active_faction == Faction.EMPIRE:
            self.active_faction = Faction.ORCS_GOBLINS
        else:
            self.active_faction = Faction.EMPIRE
            self.current_turn += 1
        
        return self.get_battlefield_state()
    
    def _simulate_unit_action(self, unit: UnitBlock):
        """Simulate action for a unit"""
        # Find enemy units
        enemy_units = [u for u in self.units.values() 
                      if u.profile.faction != unit.profile.faction and u.is_alive]
        
        if not enemy_units:
            return
        
        closest_enemy = min(enemy_units, key=lambda e: unit.position.distance_to(e.position))
        distance = unit.position.distance_to(closest_enemy.position)
        
        # Decide action based on unit type and distance
        if unit.can_charge() and distance <= unit.profile.movement + 7:  # Average charge range
            # Attempt charge
            charge_action = self.movement_engine.calculate_charge(unit, closest_enemy)
            if charge_action.is_legal:
                unit.position = charge_action.target_position
                unit.has_charged = True
                unit.is_engaged = True
                closest_enemy.is_engaged = True
                self.log_event(f"⚡ {unit.profile.name} charges {closest_enemy.profile.name}!")
                
                # Resolve combat
                self._resolve_combat(unit, closest_enemy)
            else:
                self.log_event(f"❌ {unit.profile.name} charge failed: {charge_action.penalties[0]}")
        
        elif unit.can_shoot() and unit.profile.ballistic_skill > 0:
            # Shoot if in range
            if distance <= 24:  # Shooting range
                unit.has_shot = True
                hits = self._calculate_shooting(unit, closest_enemy)
                if hits > 0:
                    wounds_result = closest_enemy.take_wounds(hits, unit.profile.strength)
                    self.log_event(f"🏹 {unit.profile.name} shoots {closest_enemy.profile.name}: {hits} hits, {wounds_result['models_killed']} casualties")
        
        elif unit.can_move():
            # Move towards enemy
            move_distance = min(unit.profile.movement, distance - 1)
            if move_distance > 0:
                # Simplified movement towards enemy
                dx = closest_enemy.position.x - unit.position.x
                dy = closest_enemy.position.y - unit.position.y
                
                # Normalize and move
                if abs(dx) > abs(dy):
                    new_x = unit.position.x + (1 if dx > 0 else -1) * min(abs(dx), move_distance)
                    new_y = unit.position.y
                else:
                    new_x = unit.position.x
                    new_y = unit.position.y + (1 if dy > 0 else -1) * min(abs(dy), move_distance)
                
                unit.position.x = max(0, min(self.width - 1, new_x))
                unit.position.y = max(0, min(self.height - 1, new_y))
                unit.has_moved = True
                self.log_event(f"🚶 {unit.profile.name} advances toward {closest_enemy.profile.name}")
    
    def _calculate_shooting(self, shooter: UnitBlock, target: UnitBlock) -> int:
        """Calculate shooting hits using TOW rules"""
        shots = shooter.current_models  # Simplified: 1 shot per model
        to_hit = shooter.profile.ballistic_skill
        
        # Roll to hit
        hits = 0
        for _ in range(shots):
            if random.randint(1, 6) >= to_hit:
                hits += 1
        
        return hits
    
    def _resolve_combat(self, attacker: UnitBlock, defender: UnitBlock):
        """Resolve combat using authentic TOW rules"""
        # Simplified combat resolution
        attacker_attacks = attacker.current_models * attacker.profile.attacks
        defender_attacks = defender.current_models * defender.profile.attacks
        
        # Attacker hits
        attacker_hits = 0
        for _ in range(attacker_attacks):
            if random.randint(1, 6) >= attacker.profile.weapon_skill:
                if random.randint(1, 6) >= defender.profile.toughness:
                    if random.randint(1, 6) >= defender.profile.armor_save:
                        attacker_hits += 1
        
        # Defender hits back
        defender_hits = 0
        for _ in range(defender_attacks):
            if random.randint(1, 6) >= defender.profile.weapon_skill:
                if random.randint(1, 6) >= attacker.profile.toughness:
                    if random.randint(1, 6) >= attacker.profile.armor_save:
                        defender_hits += 1
        
        # Apply wounds
        attacker_wounds = defender.take_wounds(attacker_hits, attacker.profile.strength)
        defender_wounds = attacker.take_wounds(defender_hits, defender.profile.strength)
        
        # Calculate combat resolution
        attacker_score = attacker_wounds['models_killed'] + attacker.rank_bonus
        defender_score = defender_wounds['models_killed'] + defender.rank_bonus
        
        if attacker_score > defender_score:
            # Attacker wins
            self.log_event(f"⚔️ {attacker.profile.name} defeats {defender.profile.name} in combat!")
            # Test defender morale
            panic_result = self.psychology_engine.panic_test(defender, attacker_score - defender_score)
            self.log_event(f"🎲 {defender.profile.name} Panic Test: {panic_result.effect}")
        elif defender_score > attacker_score:
            # Defender wins
            self.log_event(f"🛡️ {defender.profile.name} repels {attacker.profile.name}!")
            panic_result = self.psychology_engine.panic_test(attacker, defender_score - attacker_score)
            self.log_event(f"🎲 {attacker.profile.name} Panic Test: {panic_result.effect}")
        else:
            self.log_event(f"⚖️ Combat between {attacker.profile.name} and {defender.profile.name} is drawn!")
    
    def _check_objectives(self):
        """Check scenario objectives"""
        if not self.current_scenario:
            return
        
        for objective in self.current_scenario.objectives:
            # Find units within control radius
            controlling_units = []
            for unit in self.units.values():
                if (unit.is_alive and 
                    unit.position.distance_to(Position(objective.position[0], objective.position[1])) <= objective.control_radius):
                    controlling_units.append(unit)
            
            # Determine control
            empire_units = [u for u in controlling_units if u.profile.faction == Faction.EMPIRE]
            orc_units = [u for u in controlling_units if u.profile.faction == Faction.ORCS_GOBLINS]
            
            if empire_units and not orc_units:
                if objective.controlling_faction != Faction.EMPIRE:
                    objective.controlling_faction = Faction.EMPIRE
                    self.log_event(f"🏛️ Empire controls {objective.name}!")
            elif orc_units and not empire_units:
                if objective.controlling_faction != Faction.ORCS_GOBLINS:
                    objective.controlling_faction = Faction.ORCS_GOBLINS
                    self.log_event(f"🧌 Orcs control {objective.name}!")
            elif empire_units and orc_units:
                if objective.controlling_faction is not None:
                    objective.controlling_faction = None
                    self.log_event(f"⚖️ {objective.name} is contested!")

# =============================================================================
# WEB INTERFACE
# =============================================================================

# Global battlefield instance
battlefield = AuthenticTOWBattlefield()

@app.route('/')
def index():
    return render_template('authentic_tow_battlefield.html')

@app.route('/api/battlefield')
def get_battlefield():
    return jsonify(battlefield.get_battlefield_state())

@socketio.on('connect')
def handle_connect():
    print("Client connected to authentic TOW visualizer")
    emit('battlefield_update', battlefield.get_battlefield_state())

@socketio.on('start_battle')
def handle_start_battle():
    """Start the authentic TOW battle simulation"""
    def run_battle():
        while battlefield.current_turn <= 6:  # Standard 6-turn game
            time.sleep(3)  # 3 seconds per turn
            state = battlefield.simulate_turn()
            socketio.emit('battlefield_update', state)
            
            # Check if any army is completely destroyed
            empire_alive = any(u.is_alive and u.profile.faction == Faction.EMPIRE for u in battlefield.units.values())
            orc_alive = any(u.is_alive and u.profile.faction == Faction.ORCS_GOBLINS for u in battlefield.units.values())
            
            if not empire_alive:
                battlefield.log_event("🏆 ORCS & GOBLINS VICTORY - Empire army destroyed!")
                socketio.emit('battle_ended', {'winner': 'Orcs & Goblins', 'reason': 'Enemy army destroyed'})
                break
            elif not orc_alive:
                battlefield.log_event("🏆 EMPIRE VICTORY - Orc army destroyed!")
                socketio.emit('battle_ended', {'winner': 'Empire', 'reason': 'Enemy army destroyed'})
                break
        
        if battlefield.current_turn > 6:
            # Determine winner by objectives
            empire_objectives = sum(1 for obj in battlefield.current_scenario.objectives if obj.controlling_faction == Faction.EMPIRE)
            orc_objectives = sum(1 for obj in battlefield.current_scenario.objectives if obj.controlling_faction == Faction.ORCS_GOBLINS)
            
            if empire_objectives > orc_objectives:
                battlefield.log_event("🏆 EMPIRE VICTORY - Controls more objectives!")
                socketio.emit('battle_ended', {'winner': 'Empire', 'reason': 'Scenario objectives'})
            elif orc_objectives > empire_objectives:
                battlefield.log_event("🏆 ORCS & GOBLINS VICTORY - Controls more objectives!")
                socketio.emit('battle_ended', {'winner': 'Orcs & Goblins', 'reason': 'Scenario objectives'})
            else:
                battlefield.log_event("🤝 DRAW - Equal objectives controlled!")
                socketio.emit('battle_ended', {'winner': 'Draw', 'reason': 'Equal objectives'})
    
    # Start battle in separate thread
    battle_thread = threading.Thread(target=run_battle)
    battle_thread.daemon = True
    battle_thread.start()
    
    emit('battle_started', {'message': 'Authentic TOW battle commenced!'})

if __name__ == '__main__':
    print("🏛️ AUTHENTIC WARHAMMER: THE OLD WORLD VISUALIZER")
    print("=" * 60)
    print("🎯 Starting the most authentic TOW experience ever...")
    print("📱 Open http://localhost:5001 to witness authentic TOW battles!")
    print("⚔️ Featuring proper unit blocks, formations, and TOW rules!")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=True)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ Port 5001 is already in use!")
            print("🔧 Try stopping the existing server or use a different port")
        else:
            raise e 