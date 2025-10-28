#!/usr/bin/env python3
"""
🎨 WARHAMMER WEB VISUALIZER
===========================

Real-time web interface for watching AI-commanded Warhammer battles.
Features live battlefield updates, unit tracking, and AI decision visualization.
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import threading
import time
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import random

# Import our battle engine
from warhammer_battle_core import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'warhammer_ai_battle_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global battle state
current_battle = None
battle_thread = None
battle_running = False

# =============================================================================
# AI LOADING SYSTEM
# =============================================================================

class WarhammerAI(nn.Module):
    """Neural network for Warhammer AI decisions"""
    def __init__(self, input_size=50, hidden_size=256, output_size=15):
        super(WarhammerAI, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AICommander:
    """AI commander that makes tactical decisions"""
    def __init__(self, faction: Faction, model_path: str = None):
        self.faction = faction
        self.ai_network = WarhammerAI()
        self.epsilon = 0.0  # Pure exploitation for visualization
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if 'q_network_state_dict' in checkpoint:
                    self.ai_network.load_state_dict(checkpoint['q_network_state_dict'])
                    print(f"✅ Loaded {faction.value} AI from {model_path}")
                else:
                    print(f"⚠️ No q_network_state_dict in {model_path}, using untrained AI")
            except Exception as e:
                print(f"⚠️ Failed to load {faction.value} AI: {e}")
        else:
            print(f"⚠️ Using untrained {faction.value} AI")
    
    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Get AI action and Q-value for the current state"""
        if random.random() < self.epsilon:
            action = random.randint(0, 14)
            q_value = 0.0
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.ai_network(state_tensor)
                action = q_values.argmax().item()
                q_value = q_values[0][action].item()
        
        return action, q_value

# =============================================================================
# BATTLE SIMULATION SYSTEM
# =============================================================================

class VisualBattleSimulator:
    """Manages the visual battle simulation"""
    
    def __init__(self):
        self.battlefield = BattleField(width=24, height=16)
        self.empire_ai = AICommander(Faction.EMPIRE, "empire_ai_300k.pth")
        self.orc_ai = AICommander(Faction.ORCS, "orc_ai_300k.pth")
        self.translator = AIDecisionTranslator(self.battlefield)
        self.turn_number = 0
        self.battle_log = []
        self.setup_initial_battle()
    
    def setup_initial_battle(self):
        """Setup a standard Warhammer battle scenario"""
        # Empire army setup (left side)
        empire_units = [
            WarhammerUnit(
                id="empire_handgunners_1",
                profile=UNIT_PROFILES["Empire Handgunners"],
                position=Position(2, 8),
                current_models=20
            ),
            WarhammerUnit(
                id="empire_knights_1", 
                profile=UNIT_PROFILES["Empire Knights"],
                position=Position(1, 6),
                current_models=8
            ),
            WarhammerUnit(
                id="empire_cannon_1",
                profile=UNIT_PROFILES["Empire Cannon"],
                position=Position(0, 8),
                current_models=1
            )
        ]
        
        # Orc army setup (right side)
        orc_units = [
            WarhammerUnit(
                id="orc_boyz_1",
                profile=UNIT_PROFILES["Orc Boyz"],
                position=Position(21, 8),
                current_models=25
            ),
            WarhammerUnit(
                id="orc_arrer_boyz_1",
                profile=UNIT_PROFILES["Orc Arrer Boyz"],
                position=Position(22, 6),
                current_models=20
            ),
            WarhammerUnit(
                id="orc_wolf_riders_1",
                profile=UNIT_PROFILES["Orc Wolf Riders"],
                position=Position(23, 10),
                current_models=10
            )
        ]
        
        # Add units to battlefield
        for unit in empire_units + orc_units:
            self.battlefield.add_unit(unit)
    
    def get_battle_state_json(self):
        """Get current battle state as JSON for frontend"""
        empire_units_data = []
        for unit in self.battlefield.empire_units:
            if unit.is_alive:
                empire_units_data.append({
                    'id': unit.id,
                    'name': unit.profile.name,
                    'type': unit.profile.unit_type.value,
                    'position': {'x': unit.position.x, 'y': unit.position.y, 'facing': unit.position.facing},
                    'models': unit.current_models,
                    'max_models': unit.profile.max_unit_size,
                    'health_percent': (unit.total_wounds_remaining / (unit.current_models * unit.profile.wounds)) if unit.current_models > 0 else 0,
                    'can_move': unit.can_move(),
                    'can_shoot': unit.can_shoot(),
                    'is_engaged': unit.is_engaged
                })
        
        orc_units_data = []
        for unit in self.battlefield.orc_units:
            if unit.is_alive:
                orc_units_data.append({
                    'id': unit.id,
                    'name': unit.profile.name,
                    'type': unit.profile.unit_type.value,
                    'position': {'x': unit.position.x, 'y': unit.position.y, 'facing': unit.position.facing},
                    'models': unit.current_models,
                    'max_models': unit.profile.max_unit_size,
                    'health_percent': (unit.total_wounds_remaining / (unit.current_models * unit.profile.wounds)) if unit.current_models > 0 else 0,
                    'can_move': unit.can_move(),
                    'can_shoot': unit.can_shoot(),
                    'is_engaged': unit.is_engaged
                })
        
        return {
            'turn': self.turn_number,
            'empire_units': empire_units_data,
            'orc_units': orc_units_data,
            'battlefield': {
                'width': self.battlefield.width,
                'height': self.battlefield.height
            }
        }
    
    def execute_turn(self):
        """Execute one turn of the battle"""
        self.turn_number += 1
        
        # Reset movement flags
        for unit in self.battlefield.empire_units + self.battlefield.orc_units:
            unit.has_moved = False
            unit.has_shot = False
            unit.has_charged = False
        
        # Get current state
        state = self.battlefield.get_battle_state_vector()
        
        # Empire AI decision
        empire_action, empire_q_value = self.empire_ai.get_action(state)
        empire_commands = self.translator.translate_ai_decision(empire_action, empire_q_value, Faction.EMPIRE)
        
        # Orc AI decision  
        orc_action, orc_q_value = self.orc_ai.get_action(state)
        orc_commands = self.translator.translate_ai_decision(orc_action, orc_q_value, Faction.ORCS)
        
        # Execute commands
        turn_log = {
            'turn': self.turn_number,
            'empire_decision': {
                'action': empire_action,
                'action_name': self.translator.action_names[empire_action],
                'q_value': empire_q_value,
                'commands': len(empire_commands)
            },
            'orc_decision': {
                'action': orc_action,
                'action_name': self.translator.action_names[orc_action], 
                'q_value': orc_q_value,
                'commands': len(orc_commands)
            },
            'events': []
        }
        
        # Execute Empire commands
        for cmd in empire_commands:
            self.execute_command(cmd, turn_log)
        
        # Execute Orc commands
        for cmd in orc_commands:
            self.execute_command(cmd, turn_log)
        
        self.battle_log.append(turn_log)
        return turn_log
    
    def execute_command(self, command: WarhammerCommand, turn_log: dict):
        """Execute a specific command and log results"""
        unit = self.get_unit_by_id(command.unit_id)
        if not unit or not unit.is_alive:
            return
        
        if isinstance(command, MoveCommand):
            if unit.can_move() and self.battlefield.is_valid_position(command.target_position):
                if not self.battlefield.get_unit_at(command.target_position):
                    old_pos = (unit.position.x, unit.position.y)
                    unit.position = command.target_position
                    unit.has_moved = True
                    turn_log['events'].append({
                        'type': 'MOVE',
                        'unit': unit.profile.name,
                        'from': old_pos,
                        'to': (command.target_position.x, command.target_position.y)
                    })
        
        elif isinstance(command, ShootCommand):
            target = self.get_unit_by_id(command.target_unit_id)
            if unit.can_shoot() and target and target.is_alive:
                # Simplified shooting mechanics
                hit_roll = random.randint(1, 6)
                if hit_roll >= 7 - unit.profile.ballistic_skill:
                    wound_roll = random.randint(1, 6)
                    wound_needed = max(2, 7 - unit.profile.strength + target.profile.toughness)
                    
                    if wound_roll >= wound_needed:
                        save_roll = random.randint(1, 6)
                        if save_roll < target.profile.armor_save:
                            models_killed = target.take_wounds(1)
                            unit.has_shot = True
                            turn_log['events'].append({
                                'type': 'SHOOTING_HIT',
                                'shooter': unit.profile.name,
                                'target': target.profile.name,
                                'killed': models_killed
                            })
                        else:
                            turn_log['events'].append({
                                'type': 'SHOOTING_SAVED',
                                'shooter': unit.profile.name,
                                'target': target.profile.name
                            })
                    else:
                        turn_log['events'].append({
                            'type': 'SHOOTING_NO_WOUND',
                            'shooter': unit.profile.name,
                            'target': target.profile.name
                        })
                else:
                    turn_log['events'].append({
                        'type': 'SHOOTING_MISS',
                        'shooter': unit.profile.name,
                        'target': target.profile.name
                    })
        
        elif isinstance(command, ChargeCommand):
            target = self.get_unit_by_id(command.target_unit_id)
            if unit.can_move() and target and target.is_alive:
                distance = unit.position.distance_to(target.position)
                charge_distance = unit.profile.movement + random.randint(1, 6) + random.randint(1, 6)
                
                if distance <= charge_distance:
                    # Move unit adjacent to target
                    new_pos = Position(target.position.x - 1, target.position.y, unit.position.facing)
                    if self.battlefield.is_valid_position(new_pos) and not self.battlefield.get_unit_at(new_pos):
                        unit.position = new_pos
                        unit.has_moved = True
                        unit.has_charged = True
                        unit.is_engaged = True
                        target.is_engaged = True
                        
                        turn_log['events'].append({
                            'type': 'CHARGE_SUCCESS',
                            'charger': unit.profile.name,
                            'target': target.profile.name,
                            'distance': distance
                        })
                else:
                    turn_log['events'].append({
                        'type': 'CHARGE_FAILED',
                        'charger': unit.profile.name,
                        'target': target.profile.name,
                        'distance': distance,
                        'needed': charge_distance
                    })
    
    def get_unit_by_id(self, unit_id: str) -> Optional[WarhammerUnit]:
        """Find unit by ID"""
        for unit in self.battlefield.empire_units + self.battlefield.orc_units:
            if unit.id == unit_id and unit.is_alive:
                return unit
        return None
    
    def is_battle_over(self) -> bool:
        """Check if battle is finished"""
        empire_alive = any(unit.is_alive for unit in self.battlefield.empire_units)
        orcs_alive = any(unit.is_alive for unit in self.battlefield.orc_units)
        return not (empire_alive and orcs_alive) or self.turn_number >= 50

# =============================================================================
# WEB ROUTES & SOCKETIO
# =============================================================================

@app.route('/')
def index():
    """Main battle visualization page"""
    return render_template('battlefield.html')

@app.route('/api/battle/status')
def battle_status():
    """Get current battle status"""
    global current_battle
    if current_battle:
        return jsonify({
            'running': battle_running,
            'state': current_battle.get_battle_state_json(),
            'log': current_battle.battle_log[-5:] if current_battle.battle_log else []
        })
    return jsonify({'running': False, 'state': None, 'log': []})

@socketio.on('start_battle')
def handle_start_battle():
    """Start a new AI battle"""
    global current_battle, battle_thread, battle_running
    
    if not battle_running:
        current_battle = VisualBattleSimulator()
        battle_running = True
        
        def run_battle():
            global battle_running
            while battle_running and not current_battle.is_battle_over():
                turn_log = current_battle.execute_turn()
                battle_state = current_battle.get_battle_state_json()
                
                # Emit updates to all connected clients
                socketio.emit('battle_update', {
                    'state': battle_state,
                    'turn_log': turn_log
                })
                
                time.sleep(2)  # 2 second delay between turns
            
            battle_running = False
            socketio.emit('battle_ended', {
                'final_state': current_battle.get_battle_state_json(),
                'total_turns': current_battle.turn_number
            })
        
        battle_thread = threading.Thread(target=run_battle)
        battle_thread.start()
        
        emit('battle_started', {'message': 'AI Battle simulation started!'})

@socketio.on('stop_battle')
def handle_stop_battle():
    """Stop the current battle"""
    global battle_running
    battle_running = False
    emit('battle_stopped', {'message': 'Battle simulation stopped'})

@socketio.on('reset_battle')
def handle_reset_battle():
    """Reset to a new battle"""
    global current_battle, battle_running
    battle_running = False
    current_battle = VisualBattleSimulator()
    emit('battle_reset', {
        'state': current_battle.get_battle_state_json(),
        'message': 'New battle setup complete'
    })

if __name__ == '__main__':
    import os
    print("🎨 WARHAMMER WEB VISUALIZER")
    print("=" * 40)
    print("🌐 Starting web server...")
    print("📱 Open http://localhost:5001 to watch AI battles!")
    print("⚔️ Empire vs Orcs - 300k trained AIs in action!")
    
    socketio.run(app, host='0.0.0.0', port=5001, debug=True) 