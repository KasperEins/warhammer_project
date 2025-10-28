#!/usr/bin/env python3
"""
Warhammer: The Old World - Web Backend
Flask app with WebSocket support for real-time battle updates
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import time
import threading
from dataclasses import asdict
from typing import List, Dict, Any

# Import our existing battle system
from old_world_battle import OldWorldBattle, OldWorldUnit, UnitType, FormationType

app = Flask(__name__)
app.config['SECRET_KEY'] = 'warhammer_old_world_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global battle instance
current_battle = None
battle_thread = None
battle_running = False

class WebBattle(OldWorldBattle):
    """Extended battle class for web interface"""
    
    def __init__(self):
        # Initialize parent class but skip matplotlib setup
        # Use 68x48 inches battlefield as requested
        self.width = 68
        self.height = 48
        self.units = []
        self.turn = 1
        self.phase = "Start"
        self.current_player = 1
        self.battle_events = []
        
    def setup_battlefield(self):
        """Override to skip matplotlib setup"""
        pass
    
    def setup_terrain(self):
        """Setup terrain features for frontend"""
        return [
            {
                'type': 'forest',
                'name': 'Darkwood Forest',
                'x': 15, 'y': 8, 'width': 18, 'height': 12,
                'color': '#0d3300'
            },
            {
                'type': 'hill',
                'name': 'Valiant Hill',
                'x': 55, 'y': 30, 'width': 16, 'height': 10,
                'color': '#8B4513'
            },
            {
                'type': 'road',
                'name': 'The Great Road',
                'x': 0, 'y': 22, 'width': self.width, 'height': 4,
                'color': '#8B7355'
            },
            {
                'type': 'ruins',
                'name': 'Ancient Ruins',
                'x': 8, 'y': 35, 'width': 4, 'height': 6,
                'color': '#696969'
            }
        ]
    
    def update_display(self):
        """Override to skip matplotlib display"""
        pass
    
    def add_log(self, message: str):
        """Append a log entry and emit over WebSocket."""
        timestamped = message
        self.battle_events.append(timestamped)
        if len(self.battle_events) > 200:
            self.battle_events = self.battle_events[-200:]
        socketio.emit('log_event', {'message': timestamped})
    
    def _clamp_to_deployment_zone(self, unit: OldWorldUnit, y_min: float, y_max: float):
        """Clamp unit center so its whole formation stays within [y_min, y_max]."""
        half_depth = unit.depth / 2.0
        # Keep entire rectangle within zone
        lower_bound = y_min + half_depth
        upper_bound = y_max - half_depth
        # Safety margin to avoid being exactly on the line
        lower_bound += 0.1
        upper_bound -= 0.1
        unit.y = max(lower_bound, min(upper_bound, unit.y))
    
    def create_armies(self):
        """Deploy armies on opposite Y edges (top vs bottom)."""
        armies = []
        mid_x = self.width // 2
        # Official pitched battle deployment: 12" from long edge
        deploy_band = 12
        bottom_y_min, bottom_y_max = 0, deploy_band
        top_y_min, top_y_max = self.height - deploy_band, self.height

        # Empire (Player 1) near bottom, facing north (0°)
        halberdiers = OldWorldUnit(
            name="Empire Halberdiers", x=mid_x - 10, y=bottom_y_min + 8, facing=0,
            models=20, max_models=20, unit_type=UnitType.INFANTRY,
            formation=FormationType.DEEP, width=5, depth=4,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, attacks=1, leadership=7,
            armor_save=5, player=1, color='blue'
        )
        halberdiers.has_standard = True
        halberdiers.has_musician = True

        handgunners = OldWorldUnit(
            name="Handgunners", x=mid_x - 18, y=bottom_y_min + 6, facing=0,
            models=15, max_models=15, unit_type=UnitType.INFANTRY,
            formation=FormationType.WIDE, width=10, depth=2,
            movement=4, weapon_skill=3, ballistic_skill=4, strength=4,
            toughness=3, wounds=1, attacks=1, leadership=7,
            armor_save=5, player=1, color='lightblue', weapon_range=24
        )
        handgunners.armor_piercing = True

        cannon = OldWorldUnit(
            name="Great Cannon", x=mid_x + 18, y=bottom_y_min + 4, facing=0,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=0, ballistic_skill=4, strength=10,
            toughness=7, wounds=3, attacks=0, leadership=7,
            armor_save=6, player=1, color='navy', weapon_range=48
        )
        cannon.armor_piercing = True

        knights = OldWorldUnit(
            name="Empire Knights", x=mid_x + 6, y=bottom_y_min + 10, facing=0,
            models=6, max_models=6, unit_type=UnitType.CAVALRY,
            formation=FormationType.WIDE, width=3, depth=2,
            movement=8, weapon_skill=4, ballistic_skill=3, strength=4,
            toughness=3, wounds=1, attacks=1, leadership=8,
            armor_save=3, player=1, color='purple'
        )
        knights.has_standard = True
        knights.has_musician = True
        knights.lance_formation = True

        # Clamp Empire to bottom 12" zone
        for u in (halberdiers, handgunners, cannon, knights):
            self._clamp_to_deployment_zone(u, bottom_y_min, bottom_y_max)

        armies.extend([halberdiers, handgunners, cannon, knights])
        self.add_log(f"Deploy Empire Halberdiers at ({halberdiers.x:.1f}, {halberdiers.y:.1f}) facing {halberdiers.facing}°")
        self.add_log(f"Deploy Handgunners at ({handgunners.x:.1f}, {handgunners.y:.1f})")
        self.add_log(f"Deploy Great Cannon at ({cannon.x:.1f}, {cannon.y:.1f})")
        self.add_log(f"Deploy Empire Knights at ({knights.x:.1f}, {knights.y:.1f})")

        # Orcs (Player 2) near top, facing south (180°)
        orc_boyz = OldWorldUnit(
            name="Orc Boyz", x=mid_x - 10, y=top_y_min + 6, facing=180,
            models=25, max_models=25, unit_type=UnitType.INFANTRY,
            formation=FormationType.DEEP, width=5, depth=5,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=4, wounds=1, attacks=1, leadership=7,
            armor_save=6, player=2, color='red'
        )
        orc_boyz.has_standard = True
        orc_boyz.has_musician = True
        orc_boyz.frenzy = True

        orc_archers = OldWorldUnit(
            name="Orc Archers", x=mid_x + 8, y=top_y_min + 10, facing=180,
            models=15, max_models=15, unit_type=UnitType.INFANTRY,
            formation=FormationType.WIDE, width=8, depth=2,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=4, wounds=1, attacks=1, leadership=6,
            armor_save=7, player=2, color='darkred', weapon_range=18
        )

        wolf_riders = OldWorldUnit(
            name="Wolf Riders", x=mid_x - 20, y=top_y_min + 4, facing=180,
            models=6, max_models=6, unit_type=UnitType.CAVALRY,
            formation=FormationType.WIDE, width=3, depth=2,
            movement=9, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, attacks=1, leadership=6,
            armor_save=6, player=2, color='orange'
        )
        wolf_riders.fast_cavalry = True

        # Clamp Orcs to top 12" zone
        for u in (orc_boyz, orc_archers, wolf_riders):
            self._clamp_to_deployment_zone(u, top_y_min, top_y_max)

        armies.extend([orc_boyz, orc_archers, wolf_riders])
        self.add_log(f"Deploy Orc Boyz at ({orc_boyz.x:.1f}, {orc_boyz.y:.1f}) facing {orc_boyz.facing}°")
        self.add_log(f"Deploy Orc Archers at ({orc_archers.x:.1f}, {orc_archers.y:.1f})")
        self.add_log(f"Deploy Wolf Riders at ({wolf_riders.x:.1f}, {wolf_riders.y:.1f})")

        return armies

    def execute_movement(self):
        """Movement along Y axis with simple AI heuristics and marching."""
        active_units = [u for u in self.units if u.player == self.current_player and u.is_alive()]

        for unit in active_units:
            start_y = unit.y
            enemies = [u for u in self.units if u.player != self.current_player and u.is_alive()]
            target = min(enemies, key=lambda e: self.distance(unit, e)) if enemies else None

            # March if no enemies within 8"
            march_allowed = True
            if enemies:
                nearest_dist = min(self.distance(unit, e) for e in enemies)
                march_allowed = nearest_dist > 8

            if unit.weapon_range > 0 and target:
                # Ranged: try to maintain effective range
                dist = self.distance(unit, target)
                if dist > unit.weapon_range - 4:
                    step = min(unit.movement, dist - (unit.weapon_range - 4))
                    if self.current_player == 1:
                        unit.y = min(self.height - 1, unit.y + max(0, step))
                    else:
                        unit.y = max(1, unit.y - max(0, step))
                    self.add_log(f"{unit.name} advances to shooting range")
                else:
                    self.add_log(f"{unit.name} holds position for shooting")
            else:
                # Infantry/Cavalry/Monsters: advance or march
                delta = unit.movement * 2 if march_allowed else unit.movement
                if self.current_player == 1:
                    unit.y = min(self.height - 1, unit.y + delta)
                else:
                    unit.y = max(1, unit.y - delta)
                self.add_log(f"{unit.name} {'marches' if delta>unit.movement else 'advances'} {delta}\"")

            unit.update_formation()
            if abs(unit.y - start_y) >= 0.1:
                self.add_log(f"{unit.name} moves from y={start_y:.1f} to y={unit.y:.1f}")
    
    def execute_shooting(self):
        """Handle shooting phase with authentic ballistics"""
        shooters = [u for u in self.units if u.player == self.current_player 
                   and u.is_alive() and u.weapon_range > 0]
        
        print(f"  Found {len(shooters)} shooting units for Player {self.current_player}")
        
        for shooter in shooters:
            enemies = [u for u in self.units if u.player != self.current_player and u.is_alive()]
            if not enemies:
                continue
                
            in_range = [e for e in enemies if self.distance(shooter, e) <= shooter.weapon_range]
            print(f"  {shooter.name} (range {shooter.weapon_range}): {len(in_range)} targets in range")
            
            if not in_range:
                # Print distances for debugging
                closest_enemy = min(enemies, key=lambda e: self.distance(shooter, e))
                dist = self.distance(shooter, closest_enemy)
                print(f"    Closest target {closest_enemy.name} at {dist:.1f}\" (out of range)")
                continue
                
            target = min(in_range, key=lambda e: self.distance(shooter, e))
            
            # Handle artillery differently
            if shooter.unit_type.value == 'Artillery':
                # Use warmachine scatter rules
                final_target, result = self.warmachine_scatter(shooter, target)
                wounds = 0
                
                if final_target and result in ["DIRECT", "SCATTER"]:
                    # Artillery hit - multiple wounds
                    for _ in range(self.roll_dice(6)):  # D6 wounds
                        if self.roll_dice(6) >= self.get_to_wound_score(shooter.strength, final_target.toughness):
                            wounds += 1
                    
                    # Apply wounds
                    if wounds > 0:
                        final_target.models = max(0, final_target.models - wounds)
                        final_target.update_formation()
                        
                        result_text = "💥 DIRECT HIT!" if result == "DIRECT" else "💥 Scatter Hit!"
                        event = f"🎯 {shooter.name} → {final_target.name}: {wounds} wounds ({result_text})"
                        self.battle_events.append(event)
                        print(f"  {event}")
                        
                        # Create explosion animation
                        self.emit_battle_update('explosion', {
                            'x': final_target.x, 'y': final_target.y,
                            'type': 'cannon', 'intensity': wounds
                        })
            else:
                # Normal shooting
                shots = min(shooter.models, shooter.width)
                wounds = 0
                
                # Determine hit and wound scores
                to_hit = 7 - shooter.ballistic_skill  # BS3 = 4+, BS4 = 3+, etc.
                to_wound = self.get_to_wound_score(shooter.strength, target.toughness)
                
                for _ in range(shots):
                    if self.roll_dice(6) >= to_hit:  # Hit
                        if self.roll_dice(6) >= to_wound:  # Wound
                            # Armor save (unless armor piercing)
                            armor_save = target.armor_save
                            if hasattr(shooter, 'armor_piercing') and shooter.armor_piercing:
                                armor_save = 7  # No armor save against AP weapons
                            
                            # Save succeeds on roll >= armor_save; fail removes model
                            if self.roll_dice(6) < armor_save:
                                wounds += 1
            
            if wounds > 0:
                target.models = max(0, target.models - wounds)
                target.update_formation()
                
                ap_text = " (AP)" if hasattr(shooter, 'armor_piercing') and shooter.armor_piercing else ""
                event = f"🏹 {shooter.name} → {target.name}: {wounds} wounds{ap_text}"
                self.add_log(event)
    
    def execute_charges(self):
        """Handle charge phase (M + 2D6 range) with logs; contact along Y."""
        chargers = [u for u in self.units if u.player == self.current_player 
                   and u.is_alive() and u.unit_type.value != 'Artillery']
        
        for charger in chargers:
            enemies = [u for u in self.units if u.player != self.current_player and u.is_alive()]
            if not enemies:
                continue
            
            # Charge range per TOW/8e style: M + 2D6
            roll = self.roll_dice(6) + self.roll_dice(6)
            charge_range = charger.movement + roll
            target = min(enemies, key=lambda e: self.distance(charger, e))
            distance = self.distance(charger, target)
            
            if distance <= charge_range:
                # Successful charge
                charger.x = target.x
                charger.y = target.y + (-3 if charger.player == 1 else 3)
                charger.has_charged = True
                self.add_log(f"{charger.name} charges {target.name} (roll {roll}, range {charge_range}")
            else:
                # Failed charge (no move for simplicity)
                self.add_log(f"{charger.name} fails charge on {target.name} (distance {distance:.1f} > {charge_range})")
    
    def roll_dice(self, sides):
        """Simple dice rolling"""
        import random
        return random.randint(1, sides)
    
    def unit_to_dict(self, unit: OldWorldUnit) -> Dict[str, Any]:
        """Convert unit to dictionary for JSON serialization"""
        return {
            'id': unit._id,
            'name': unit.name,
            'x': unit.x,
            'y': unit.y,
            'facing': unit.facing,
            'models': unit.models,
            'max_models': unit.max_models,
            'unit_type': unit.unit_type.value,
            'formation': unit.formation.value,
            'width': unit.width,
            'depth': unit.depth,
            'movement': unit.movement,
            'weapon_skill': unit.weapon_skill,
            'ballistic_skill': unit.ballistic_skill,
            'strength': unit.strength,
            'toughness': unit.toughness,
            'wounds': unit.wounds,
            'attacks': unit.attacks,
            'leadership': unit.leadership,
            'armor_save': unit.armor_save,
            'player': unit.player,
            'color': unit.color,
            'weapon_range': unit.weapon_range,
            'has_charged': unit.has_charged,
            'is_fleeing': unit.is_fleeing,
            'is_alive': unit.is_alive(),
            'formation_points': unit.get_formation_points(),
            # New properties
            'has_standard': getattr(unit, 'has_standard', False),
            'has_musician': getattr(unit, 'has_musician', False),
            'armor_piercing': getattr(unit, 'armor_piercing', False),
            'lance_formation': getattr(unit, 'lance_formation', False),
            'frenzy': getattr(unit, 'frenzy', False),
            'fast_cavalry': getattr(unit, 'fast_cavalry', False),
            'fear': getattr(unit, 'fear', False),
            'terror': getattr(unit, 'terror', False),
            'immune_to_fear': getattr(unit, 'immune_to_fear', False),
            'stubborn': getattr(unit, 'stubborn', False),
            'regeneration': getattr(unit, 'regeneration', False)
        }
    
    def get_battle_state(self) -> Dict[str, Any]:
        """Get complete battle state for frontend"""
        return {
            'turn': self.turn,
            'phase': self.phase,
            'current_player': self.current_player,
            'units': [self.unit_to_dict(unit) for unit in self.units],
            'terrain': self.setup_terrain(),
            'battle_events': self.battle_events[-50:],  # Last 50 events for UI
            'battlefield': {
                'width': self.width,
                'height': self.height
            }
        }
    
    def emit_battle_update(self, event_type: str, data: Dict[str, Any] = None):
        """Emit battle update to all connected clients"""
        socketio.emit('battle_update', {
            'event_type': event_type,
            'battle_state': self.get_battle_state(),
            'data': data or {}
        })
    
    def run_turn_async(self):
        """Run battle turn with WebSocket updates"""
        global battle_running
        
        if not battle_running:
            return False
            
        print(f"\nTURN {self.turn} - PLAYER {self.current_player}")
        self.add_log(f"TURN {self.turn} - Player {self.current_player}")
        
        # Rally Phase (start of turn)
        self.phase = "Rally"
        self.add_log("🏳️ Rally Phase")
        self.emit_battle_update('phase_change', {'phase': 'Rally'})
        socketio.sleep(0.5)
        
        self.rally_phase()
        self.emit_battle_update('rally_complete')
        socketio.sleep(0.5)
        
        # Charges (declared/resolved before movement)
        self.phase = "Charges"
        self.add_log("🐎 Charge Phase")
        self.emit_battle_update('phase_change', {'phase': 'Charges'})
        socketio.sleep(0.5)
        
        self.execute_charges()
        self.emit_battle_update('charge_complete')
        socketio.sleep(0.8)
        
        # Movement Phase
        self.phase = "Movement"
        self.add_log("🏃 Movement Phase")
        self.emit_battle_update('phase_change', {'phase': 'Movement'})
        socketio.sleep(0.5)
        
        self.execute_movement()
        self.emit_battle_update('movement_complete')
        socketio.sleep(0.8)
        
        # Shooting Phase
        self.phase = "Shooting"
        self.add_log("🏹 Shooting Phase")
        self.emit_battle_update('phase_change', {'phase': 'Shooting'})
        socketio.sleep(0.5)
        
        self.execute_shooting()
        self.emit_battle_update('shooting_complete')
        socketio.sleep(0.8)
        
        # Combat Phase
        self.phase = "Combat"
        self.add_log("⚔️ Combat Phase")
        self.emit_battle_update('phase_change', {'phase': 'Combat'})
        socketio.sleep(0.5)
        
        self.execute_combat()
        self.emit_battle_update('combat_complete')
        socketio.sleep(1.0)
        
        # Next player/turn
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1
            self.turn += 1
        
        # Check victory
        victory = self.check_victory()
        if not victory:
            battle_running = False
            self.emit_battle_update('battle_end')
            return False  # Battle ended
        else:
            self.emit_battle_update('turn_complete')
            return True  # Battle continues

    def check_victory(self) -> bool:
        """Check victory conditions using authentic Old World rules"""
        p1_alive = any(u.player == 1 and u.is_alive() for u in self.units)
        p2_alive = any(u.player == 2 and u.is_alive() for u in self.units)
        
        # Immediate victory - army eliminated
        if not p1_alive:
            print("\n🏆 ORC FORCES VICTORY! (Army Eliminated)")
            self.add_log("🏆 ORC FORCES VICTORY! (Army Eliminated)")
            return False
        elif not p2_alive:
            print("\n🏆 EMPIRE FORCES VICTORY! (Army Eliminated)")
            self.add_log("🏆 EMPIRE FORCES VICTORY! (Army Eliminated)")
            return False
        
        # Game ends after 6 turns - count victory points
        if self.turn > 6:
            vp_p1, vp_p2 = self.calculate_victory_points()
            print(f"\n⏰ BATTLE ENDS AFTER 6 TURNS!")
            print(f"📊 VICTORY POINTS:")
            print(f"   Empire: {vp_p1} VP")
            print(f"   Orcs: {vp_p2} VP")
            self.add_log(f"⏰ BATTLE ENDS AFTER 6 TURNS!")
            self.add_log(f"Empire: {vp_p1} VP | Orcs: {vp_p2} VP")
            
            if vp_p1 > vp_p2:
                print(f"\n🏆 EMPIRE VICTORY! (+{vp_p1 - vp_p2} VP)")
                self.add_log(f"🏆 EMPIRE VICTORY!")
            elif vp_p2 > vp_p1:
                print(f"\n🏆 ORC VICTORY! (+{vp_p2 - vp_p1} VP)")
                self.add_log(f"🏆 ORC VICTORY!")
            else:
                print(f"\n🤝 HONORABLE DRAW! (Tied at {vp_p1} VP each)")
                self.add_log(f"🤝 HONORABLE DRAW!")
            return False
        
        return True
    
    def calculate_victory_points(self) -> tuple:
        """Calculate victory points earned by each player"""
        vp_p1 = 0  # Points earned by Player 1 (Empire)
        vp_p2 = 0  # Points earned by Player 2 (Orcs)
        
        for unit in self.units:
            if unit.points_cost == 0:
                continue  # Skip units without point values
                
            if unit.player == 1:  # Empire unit
                # Player 2 (Orcs) gets VP for casualties inflicted
                casualties = unit.starting_models - unit.models
                if casualties > 0:
                    casualty_percentage = casualties / unit.starting_models
                    if casualty_percentage >= 1.0:  # Completely destroyed
                        vp_p2 += unit.points_cost
                    elif casualty_percentage >= 0.5:  # Half or more destroyed
                        vp_p2 += unit.points_cost // 2
                        
            elif unit.player == 2:  # Orc unit
                # Player 1 (Empire) gets VP for casualties inflicted
                casualties = unit.starting_models - unit.models
                if casualties > 0:
                    casualty_percentage = casualties / unit.starting_models
                    if casualty_percentage >= 1.0:  # Completely destroyed
                        vp_p1 += unit.points_cost
                    elif casualty_percentage >= 0.5:  # Half or more destroyed
                        vp_p1 += unit.points_cost // 2
        
        return vp_p1, vp_p2

    def get_to_hit_score(self, attacker_ws: int, defender_ws: int) -> int:
        """Authentic WS vs WS to-hit chart from Warhammer: The Old World"""
        if attacker_ws >= defender_ws * 2:
            return 2  # Need 2+ to hit
        elif attacker_ws > defender_ws:
            return 3  # Need 3+ to hit  
        elif attacker_ws == defender_ws:
            return 4  # Need 4+ to hit
        elif attacker_ws * 2 <= defender_ws:
            return 5  # Need 5+ to hit
        else:
            return 4  # Default 4+ to hit
    
    def get_to_wound_score(self, strength: int, toughness: int) -> int:
        """Authentic S vs T wound chart from Warhammer: The Old World"""
        if strength >= toughness * 2:
            return 2  # Need 2+ to wound
        elif strength > toughness:
            return 3  # Need 3+ to wound
        elif strength == toughness:
            return 4  # Need 4+ to wound
        elif strength * 2 <= toughness:
            return 6  # Need 6+ to wound
        else:
            return 5  # Need 5+ to wound
    
    def calculate_combat_resolution(self, unit1, unit2, wounds1: int, wounds2: int) -> tuple:
        """Calculate combat resolution with rank bonuses, standards, musicians"""
        # Base wounds caused
        resolution1 = wounds1
        resolution2 = wounds2
        
        # Rank bonus (extra ranks past the first)
        if unit1.depth > 1:
            resolution1 += min(unit1.depth - 1, 3)  # Max +3 for ranks
        if unit2.depth > 1:
            resolution2 += min(unit2.depth - 1, 3)
        
        # Standard bearer bonus (+1 combat resolution)
        if hasattr(unit1, 'has_standard') and unit1.has_standard:
            resolution1 += 1
        if hasattr(unit2, 'has_standard') and unit2.has_standard:
            resolution2 += 1
        
        # Charge bonus (+1 combat resolution)
        if unit1.has_charged:
            resolution1 += 1
        if unit2.has_charged:
            resolution2 += 1
        
        # Higher ground bonus (+1 if on hill)
        # Could be implemented based on terrain position
        
        return resolution1, resolution2
    
    def break_test(self, unit, combat_resolution_lost: int) -> bool:
        """Perform break test - return True if unit breaks"""
        # Leadership test with negative modifier
        break_roll = self.roll_dice(6) + self.roll_dice(6)  # 2D6
        modified_leadership = unit.leadership - combat_resolution_lost
        
        if break_roll > modified_leadership:
            unit.is_fleeing = True
            print(f"  {unit.name} breaks and flees! (Rolled {break_roll} vs Ld{modified_leadership})")
            return True
        else:
            print(f"  {unit.name} holds firm! (Rolled {break_roll} vs Ld{modified_leadership})")
            return False
    
    def execute_combat(self):
        """Handle combat phase with authentic TOW mechanics"""
        combat_pairs = []
        processed_ids = set()
        
        try:
            for unit in self.units:
                if not unit.is_alive() or unit._id in processed_ids:
                    continue
                    
                enemies = [u for u in self.units if u.player != unit.player 
                          and u.is_alive() and self.distance(unit, u) <= 3]
                
                for enemy in enemies:
                    if enemy._id not in processed_ids:
                        combat_pairs.append((unit, enemy))
                        processed_ids.add(unit._id)
                        processed_ids.add(enemy._id)
                        break
        except Exception as e:
            print(f"Combat error: {e}")
            return
        
        for unit1, unit2 in combat_pairs:
            self.add_log(f"⚔️ {unit1.name} engages {unit2.name}")
            
            # Check for Fear/Terror effects before combat
            unit1_fear_penalty = 0
            unit2_fear_penalty = 0
            
            if hasattr(unit2, 'fear') and unit2.fear:
                if not self.fear_test(unit1, unit2):
                    unit1_fear_penalty = -1  # -1 to hit if failed Fear test
            
            if hasattr(unit1, 'fear') and unit1.fear:
                if not self.fear_test(unit2, unit1):
                    unit2_fear_penalty = -1
            
            # Terror tests (if first time encountering)
            if hasattr(unit2, 'terror') and unit2.terror:
                if not self.terror_test(unit1, unit2):
                    continue  # Unit1 fled from Terror, no combat
                    
            if hasattr(unit1, 'terror') and unit1.terror:
                if not self.terror_test(unit2, unit1):
                    continue  # Unit2 fled from Terror, no combat
            
            # Calculate attacks (front rank fights, plus supporting attacks)
            attacks1 = min(unit1.models, unit1.width) * unit1.attacks
            if unit1.depth > 1:  # Supporting attacks from second rank
                attacks1 += min(max(0, unit1.models - unit1.width), unit1.width) // 2
            
            # Add bonus attacks from psychology
            attacks1 += self.calculate_charge_bonus(unit1, unit2)
            
            attacks2 = min(unit2.models, unit2.width) * unit2.attacks
            if unit2.depth > 1:
                attacks2 += min(max(0, unit2.models - unit2.width), unit2.width) // 2
                
            attacks2 += self.calculate_charge_bonus(unit2, unit1)
            
            wounds1 = 0
            wounds2 = 0
            
            # Unit1 attacks Unit2
            to_hit1 = self.get_to_hit_score(unit1.weapon_skill, unit2.weapon_skill) - unit1_fear_penalty
            to_wound1 = self.get_to_wound_score(unit1.strength, unit2.toughness)
            
            for _ in range(attacks1):
                hit_roll = self.roll_dice(6)
                if hit_roll >= to_hit1:  # Hit
                    if self.roll_dice(6) >= to_wound1:  # Wound
                        if self.roll_dice(6) < unit2.armor_save:  # Failed save
                            wounds1 += 1
            
            # Unit2 attacks Unit1  
            to_hit2 = self.get_to_hit_score(unit2.weapon_skill, unit1.weapon_skill) - unit2_fear_penalty
            to_wound2 = self.get_to_wound_score(unit2.strength, unit1.toughness)
            
            for _ in range(attacks2):
                hit_roll = self.roll_dice(6)
                if hit_roll >= to_hit2:  # Hit
                    if self.roll_dice(6) >= to_wound2:  # Wound
                        if self.roll_dice(6) < unit1.armor_save:  # Failed save
                            wounds2 += 1
            
            # Remove casualties
            unit1.models = max(0, unit1.models - wounds2)
            unit2.models = max(0, unit2.models - wounds1)
            unit1.update_formation()
            unit2.update_formation()
            
            # Combat Resolution
            resolution1, resolution2 = self.calculate_combat_resolution(unit1, unit2, wounds1, wounds2)
            
            self.add_log(f"    {unit1.name} deals {wounds1} wounds (CR: {resolution1})")
            self.add_log(f"    {unit2.name} deals {wounds2} wounds (CR: {resolution2})")
            
            # Determine winner and break tests
            if resolution1 > resolution2:
                diff = resolution1 - resolution2
                self.add_log(f"    {unit1.name} wins combat by {diff}!")
                if unit2.is_alive():
                    if self.break_test(unit2, diff):
                        # Check for panic tests when unit breaks
                        self.check_panic_tests(unit2)
            elif resolution2 > resolution1:
                diff = resolution2 - resolution1
                self.add_log(f"    {unit2.name} wins combat by {diff}!")
                if unit1.is_alive():
                    if self.break_test(unit1, diff):
                        # Check for panic tests when unit breaks
                        self.check_panic_tests(unit1)
            else:
                self.add_log("    Combat is a draw!")
            
            # Reset charge bonuses
            unit1.has_charged = False
            unit2.has_charged = False
            
            if wounds1 > 0 or wounds2 > 0:
                event = f"⚔️ {unit1.name} vs {unit2.name}: {wounds1}/{wounds2} wounds"
                self.add_log(event)

    def panic_test(self, unit, modifier: int = 0) -> bool:
        """Perform panic test - units panic when seeing friends flee"""
        panic_roll = self.roll_dice(6) + self.roll_dice(6)  # 2D6
        modified_leadership = unit.leadership + modifier
        
        if panic_roll > modified_leadership:
            unit.is_fleeing = True
            print(f"  {unit.name} panics and flees! (Rolled {panic_roll} vs Ld{modified_leadership})")
            return True
        else:
            print(f"  {unit.name} passes panic test! (Rolled {panic_roll} vs Ld{modified_leadership})")
            return False
    
    def check_panic_tests(self, fleeing_unit):
        """Check for panic when a unit breaks"""
        for unit in self.units:
            if (unit.player == fleeing_unit.player and 
                unit.is_alive() and 
                not unit.is_fleeing and 
                unit._id != fleeing_unit._id):
                
                distance = self.distance(unit, fleeing_unit)
                if distance <= 6:  # Panic test if within 6"
                    modifier = -1 if distance <= 3 else 0  # -1 if very close
                    if self.panic_test(unit, modifier):
                        # Chain reaction - this unit fleeing might cause more panic
                        self.check_panic_tests(unit)
    
    def rally_phase(self):
        """Handle fleeing units trying to rally"""
        print("Rally Phase")
        for unit in self.units:
            if unit.is_fleeing and unit.is_alive():
                # Musicians give +1 to rally
                rally_bonus = 1 if hasattr(unit, 'has_musician') and unit.has_musician else 0
                rally_roll = self.roll_dice(6) + self.roll_dice(6)
                
                if rally_roll <= unit.leadership + rally_bonus:
                    unit.is_fleeing = False
                    print(f"  {unit.name} rallies! (Rolled {rally_roll} vs Ld{unit.leadership + rally_bonus})")
                else:
                    print(f"  {unit.name} continues to flee (Rolled {rally_roll} vs Ld{unit.leadership + rally_bonus})")
                    # Fleeing units move away from enemy
                    if unit.player == 1:
                        unit.x = max(0, unit.x - 6)  # Move towards own table edge
                    else:
                        unit.x = min(self.width, unit.x + 6)
    
    def calculate_charge_bonus(self, unit, target):
        """Calculate charge bonuses including lance formation"""
        bonus_attacks = 0
        
        # Frenzy gives +1 attack when charging
        if hasattr(unit, 'frenzy') and unit.frenzy and unit.has_charged:
            bonus_attacks += 1
        
        # Lance formation for cavalry
        if (hasattr(unit, 'lance_formation') and unit.lance_formation and 
            unit.has_charged and unit.unit_type == UnitType.CAVALRY):
            # Lance formation gives +1 Strength on charge (handled elsewhere)
            pass
        
        return bonus_attacks

    def fear_test(self, unit, fear_causer) -> bool:
        """Perform Fear test - units must test when charging/charged by Fear-causing enemies"""
        if hasattr(unit, 'immune_to_fear') and unit.immune_to_fear:
            return True  # Passes automatically
        
        fear_roll = self.roll_dice(6) + self.roll_dice(6)  # 2D6
        
        if fear_roll <= unit.leadership:
            print(f"  {unit.name} passes Fear test! (Rolled {fear_roll} vs Ld{unit.leadership})")
            return True
        else:
            print(f"  {unit.name} fails Fear test! (Rolled {fear_roll} vs Ld{unit.leadership})")
            return False
    
    def terror_test(self, unit, terror_causer) -> bool:
        """Perform Terror test - automatically causes Fear, plus forces break test"""
        if hasattr(unit, 'immune_to_fear') and unit.immune_to_fear:
            return True  # Immune to Terror as well
        
        terror_roll = self.roll_dice(6) + self.roll_dice(6)  # 2D6
        
        if terror_roll <= unit.leadership:
            print(f"  {unit.name} passes Terror test! (Rolled {terror_roll} vs Ld{unit.leadership})")
            return True
        else:
            print(f"  {unit.name} fails Terror test and flees! (Rolled {terror_roll} vs Ld{unit.leadership})")
            unit.is_fleeing = True
            self.check_panic_tests(unit)  # Terror failure causes panic
            return False
    
    def warmachine_scatter(self, cannon, target):
        """Handle cannon scatter and misfires"""
        # Artillery scatter mechanics
        misfire_roll = self.roll_dice(6)
        
        if misfire_roll == 1:
            # Misfire!
            misfire_type = self.roll_dice(6)
            if misfire_type <= 2:
                print(f"  {cannon.name} DESTROYED in catastrophic explosion!")
                cannon.models = 0
                cannon.update_formation()
                return None, "DESTROYED"
            elif misfire_type <= 4:
                print(f"  {cannon.name} cannon damaged - cannot shoot this turn!")
                return None, "DAMAGED"
            else:
                print(f"  {cannon.name} misfires but crew quickly reloads!")
                return None, "MISFIRE"
        
        # Normal shot - check for scatter
        artillery_roll = self.roll_dice(6)
        if artillery_roll >= 4:  # Direct hit
            print(f"  {cannon.name} scores a DIRECT HIT on {target.name}!")
            return target, "DIRECT"
        else:
            # Scatter
            scatter_distance = self.roll_dice(6)
            scatter_direction = self.roll_dice(8)  # 8 directions
            
            # Find new target location (simplified)
            scatter_targets = [u for u in self.units if u.player != cannon.player and u.is_alive()]
            if scatter_targets and scatter_distance <= 3:
                # Scatter might hit another unit
                scatter_target = self.roll_dice(len(scatter_targets))
                new_target = scatter_targets[scatter_target - 1]
                print(f"  {cannon.name} scatters {scatter_distance}\" and hits {new_target.name}!")
                return new_target, "SCATTER"
            else:
                print(f"  {cannon.name} scatters {scatter_distance}\" and misses completely!")
                return None, "MISS"
    
    def calculate_terrain_effects(self, unit, target_terrain=None):
        """Calculate movement and combat penalties from terrain"""
        terrain_penalty = 0
        
        # Check if unit is in difficult terrain
        if hasattr(unit, 'in_terrain'):
            if unit.in_terrain == "forest":
                terrain_penalty = -1  # -1 to hit in forest
                print(f"    {unit.name} suffers -1 to hit (fighting in forest)")
            elif unit.in_terrain == "hill":
                if target_terrain != "hill":
                    terrain_penalty = 1  # +1 advantage from high ground
                    print(f"    {unit.name} gains +1 to hit (high ground advantage)")
        
        return terrain_penalty
    
    def check_unit_psychology(self, unit, enemy_unit=None):
        """Check for various psychology effects"""
        effects = []
        
        # Frenzy effects
        if hasattr(unit, 'frenzy') and unit.frenzy and not unit.is_fleeing:
            if enemy_unit and self.distance(unit, enemy_unit) <= 12:
                effects.append("FRENZIED")
                # Frenzied units must charge if able
                
        # Hatred effects (could be added to specific unit matchups)
        if hasattr(unit, 'hatred_vs') and enemy_unit:
            if enemy_unit.name in unit.hatred_vs:
                effects.append("HATRED")
                # Re-roll failed to-hit rolls in first round of combat
        
        # Stubborn
        if hasattr(unit, 'stubborn') and unit.stubborn:
            effects.append("STUBBORN")
            # Uses unmodified Leadership for break tests
        
        return effects

@app.route('/')
def index():
    """Serve main battle interface"""
    return render_template('battle.html')

@app.route('/api/new_battle', methods=['POST'])
def new_battle():
    """Start a new battle"""
    global current_battle, battle_running
    
    current_battle = WebBattle()
    armies = current_battle.create_armies()
    for unit in armies:
        current_battle.add_unit(unit)
    
    battle_running = False
    
    return jsonify({
        'success': True,
        'battle_state': current_battle.get_battle_state()
    })

@app.route('/api/battle_state')
def get_battle_state():
    """Get current battle state"""
    if current_battle:
        return jsonify(current_battle.get_battle_state())
    return jsonify({'error': 'No active battle'})

@app.route('/api/start_battle', methods=['POST'])
def api_start_battle():
    """HTTP endpoint to start the running battle loop"""
    global battle_thread, battle_running, current_battle
    # Auto-create a new battle if one doesn't exist
    if not current_battle:
        current_battle = WebBattle()
        armies = current_battle.create_armies()
        for unit in armies:
            current_battle.add_unit(unit)
        # Emit initial state so the UI sees deployments
        socketio.emit('battle_update', {
            'event_type': 'initialized',
            'battle_state': current_battle.get_battle_state(),
            'data': {}
        })
    if battle_running:
        return jsonify({'success': False})
    battle_running = True

    def run_battle():
        try:
            while battle_running and current_battle:
                battle_continues = current_battle.run_turn_async()
                if not battle_continues:
                    break
                socketio.sleep(1.5)
        finally:
            pass

    battle_thread = threading.Thread(target=run_battle)
    battle_thread.daemon = True
    battle_thread.start()
    return jsonify({'success': True})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    if current_battle:
        emit('battle_update', {
            'event_type': 'connected',
            'battle_state': current_battle.get_battle_state(),
            'data': {}
        })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_battle')
def handle_start_battle():
    """Start battle execution"""
    global battle_thread, battle_running
    
    if not current_battle:
        return
    
    if battle_running:
        return  # Battle already running
    
    battle_running = True
    
    def run_battle():
        try:
            while battle_running and current_battle:
                battle_continues = current_battle.run_turn_async()
                if not battle_continues:
                    battle_running = False
                    break
                if battle_running:
                    socketio.sleep(1.5)  # Quick pause between turns
        except Exception as e:
            print(f"Battle error: {e}")
            battle_running = False
    
    battle_thread = threading.Thread(target=run_battle)
    battle_thread.daemon = True
    battle_thread.start()

@socketio.on('pause_battle')
def handle_pause_battle():
    """Pause battle execution"""
    global battle_running
    battle_running = False
    emit('battle_update', {'event_type': 'paused', 'battle_state': current_battle.get_battle_state(), 'data': {}})

@socketio.on('next_turn')
def handle_next_turn():
    """Execute next turn manually"""
    if current_battle and not battle_running:
        current_battle.run_turn_async()

if __name__ == '__main__':
    print("🚀 Starting Warhammer: The Old World Web Server")
    print("📱 Open your browser to http://localhost:5001")
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)