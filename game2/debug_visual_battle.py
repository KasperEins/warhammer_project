#!/usr/bin/env python3
"""
DEBUG VISUAL AI BATTLE
======================
Debugging version to fix unit display issues
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from visual_ai_battle import VisualAIBattle

class DebugVisualBattle(VisualAIBattle):
    
    def setup_battlefield(self):
        """Simplified battlefield setup"""
        self.ax.clear()
        self.ax.set_xlim(0, 80)
        self.ax.set_ylim(0, 60)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#228B22')  # Green background
        
        # Simple terrain markers (no emojis)
        forest = patches.Rectangle((2, 45), 15, 12, color='#006400', alpha=0.7)
        self.ax.add_patch(forest)
        self.ax.text(9.5, 51, 'FOREST', ha='center', va='center', 
                    fontsize=12, color='white', weight='bold')
        
        # River
        river_points = [(0, 30), (25, 28), (55, 32), (80, 30)]
        river_x = [p[0] for p in river_points]
        river_y = [p[1] for p in river_points]
        
        for i in range(len(river_x)-1):
            self.ax.plot([river_x[i], river_x[i+1]], 
                       [river_y[i], river_y[i+1]], 
                       color='#4169E1', linewidth=8, alpha=0.7)
        
        # Hills
        hill = patches.Circle((65, 15), 8, color='#8B7355', alpha=0.6)
        self.ax.add_patch(hill)
        self.ax.text(65, 15, 'HILL', ha='center', va='center', 
                    fontsize=10, color='white', weight='bold')
        
        self.ax.set_title('AI vs AI: WARHAMMER BATTLE (DEBUG)', 
                         fontsize=20, fontweight='bold', color='gold', pad=20)
    
    def draw_units(self):
        """Simplified unit drawing for debugging"""
        print(f"Drawing {len(self.units)} units...")
        
        for i, unit in enumerate(self.units):
            print(f"Drawing unit {i+1}: {unit.name} (Player {unit.player})")
            print(f"  Position: ({unit.x}, {unit.y})")
            print(f"  Alive: {unit.is_alive()}, Models: {unit.models}, Health: {unit.health}")
            
            if unit.is_alive():
                # Draw a simple circle for each unit
                circle = patches.Circle((unit.x, unit.y), 3, 
                                      facecolor=unit.color, alpha=0.8,
                                      edgecolor='white', linewidth=2)
                self.ax.add_patch(circle)
                
                # Unit label
                unit_info = f"{unit.name}\nP{unit.player}\n{unit.models} models\n{int(unit.health)}% HP"
                
                bbox_color = 'lightblue' if unit.player == 1 else 'lightcoral'
                self.ax.text(unit.x, unit.y + 5, unit_info, ha='center', va='bottom',
                           fontsize=8, fontweight='bold', color='black',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.9))
                
                print(f"  Drew circle at ({unit.x}, {unit.y}) with color {unit.color}")
            else:
                print(f"  Unit {unit.name} is not alive, skipping")
    
    def update_display(self):
        """Update display with more debugging"""
        print("\n=== UPDATING DISPLAY ===")
        self.setup_battlefield()
        self.draw_units()
        
        # Army status
        nuln_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        troll_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        print(f"Alive units: {len(nuln_units)} Nuln, {len(troll_units)} Troll")
        
        status = f"""TURN {self.turn} - {self.phase}
AI vs AI BATTLE (DEBUG)

NULN ARMY: {len(nuln_units)} units alive
TROLL HORDE: {len(troll_units)} units alive"""
        
        self.ax.text(0.02, 0.98, status, transform=self.ax.transAxes,
                    fontsize=12, va='top', fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='navy', alpha=0.9))
        
        # Battle log
        if self.battle_log:
            log_text = "BATTLE LOG:\n" + "\n".join(self.battle_log[-3:])
            self.ax.text(0.98, 0.02, log_text, transform=self.ax.transAxes,
                        fontsize=10, va='bottom', ha='right', color='white',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='darkgreen', alpha=0.9))
        
        plt.draw()
        plt.pause(0.5)  # Longer pause for debugging
        print("Display updated\n")
    
    def run_battle(self):
        """Run battle with more debugging and slower pace"""
        print("🏛️ DEBUG VISUAL AI vs AI BATTLE COMMENCING!")
        print("=" * 60)
        
        self.phase = "Battle Begins"
        self.update_display()
        
        input("Press Enter to start the battle...")
        
        while self.battle_active and self.turn <= self.max_turns:
            print(f"\n⚔️ TURN {self.turn} - AI COMMAND PHASE")
            
            self.phase = f"Turn {self.turn} - AI Commands"
            
            # Execute AI actions
            if self.nuln_ai and self.troll_ai:
                print("Getting AI actions...")
                nuln_action, troll_action = self.execute_ai_actions()
                print(f"Nuln AI chose action {nuln_action}, Troll AI chose action {troll_action}")
            else:
                print("AI agents not loaded, using random actions")
                nuln_action, troll_action = 0, 0
                self.battle_log.append("Random actions taken (AI not loaded)")
            
            self.update_display()
            
            # Check for victory
            victory_result = self.check_victory()
            if victory_result:
                self.phase = victory_result
                self.update_display()
                
                print(f"\n🏆 {victory_result}")
                print("=" * 60)
                break
            
            self.turn += 1
            input(f"Press Enter to continue to turn {self.turn}...")
        
        print("\n📺 Battle complete. Close the window to exit.")
        plt.ioff()
        plt.show()

def main():
    print("🔧 DEBUG VISUAL AI vs AI BATTLE SYSTEM")
    print("=" * 40)
    print("This debug version runs slower and shows more information")
    print()
    
    battle = DebugVisualBattle()
    battle.run_battle()

if __name__ == "__main__":
    main() 