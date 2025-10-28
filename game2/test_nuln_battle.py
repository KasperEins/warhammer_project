#!/usr/bin/env python3
"""
NULN ARMY vs TROLL HORDE - Authentic Old World Battle
Test version with armies starting closer for immediate combat
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from old_world_battle import OldWorldBattle, OldWorldUnit, UnitType, FormationType
import random

class NulnVsTrollBattle(OldWorldBattle):
    def create_armies(self):
        """Create authentic Nuln Army vs Troll Horde armies"""
        armies = []
        
        # ===============================
        # ARMY OF NULN (EMPIRE) - Player 1
        # ===============================
        
        # CHARACTERS
        # General Hans von Löwenhacke [190 pts]
        general_hans = OldWorldUnit(
            name="General Hans von Löwenhacke", x=15, y=24, facing=90,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=4, weapon_skill=6, ballistic_skill=5, strength=4,
            toughness=4, wounds=3, attacks=4, leadership=9,
            armor_save=2, player=1, color='gold'  # Full plate + Griffon Helm
        )
        general_hans.has_standard = True  # General
        general_hans.immune_to_fear = True
        general_hans.inspiring_presence = True
        
        # Empire Engineer with Hochland Long Rifle [55 pts]
        engineer_rifle = OldWorldUnit(
            name="Engineer (Hochland Rifle)", x=12, y=20, facing=90,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=4, weapon_skill=4, ballistic_skill=5, strength=4,
            toughness=3, wounds=2, attacks=2, leadership=8,
            armor_save=6, player=1, color='lightblue', weapon_range=36
        )
        engineer_rifle.armor_piercing = True  # Hochland rifle
        
        # Master Mage Level 2 [90 pts]
        master_mage = OldWorldUnit(
            name="Master Mage", x=18, y=22, facing=90,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=2, attacks=1, leadership=8,
            armor_save=7, player=1, color='purple'
        )
        master_mage.wizard_level = 2
        
        # CORE UNITS
        # 20 Nuln Veteran State Troops [375 pts]
        nuln_veterans = OldWorldUnit(
            name="Nuln Veteran Halberdiers", x=20, y=25, facing=90,
            models=20, max_models=20, unit_type=UnitType.INFANTRY,
            formation=FormationType.DEEP, width=5, depth=4,
            movement=4, weapon_skill=4, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, attacks=1, leadership=8,
            armor_save=5, player=1, color='blue'  # Light armour + drilled
        )
        nuln_veterans.has_standard = True
        nuln_veterans.has_musician = True
        nuln_veterans.veteran = True  # +1 WS
        nuln_veterans.drilled = True  # Special rule
        
        # 10 State Missile Troops (Handgunners) [Part of 375 pts]
        handgunners_1 = OldWorldUnit(
            name="Handgunners I", x=12, y=15, facing=90,
            models=10, max_models=10, unit_type=UnitType.INFANTRY,
            formation=FormationType.WIDE, width=8, depth=2,
            movement=4, weapon_skill=3, ballistic_skill=4, strength=4,
            toughness=3, wounds=1, attacks=1, leadership=7,
            armor_save=6, player=1, color='lightblue', weapon_range=24
        )
        handgunners_1.armor_piercing = True
        handgunners_1.drilled = True
        
        # 10 State Missile Troops (Handgunners) 
        handgunners_2 = OldWorldUnit(
            name="Handgunners II", x=25, y=15, facing=90,
            models=10, max_models=10, unit_type=UnitType.INFANTRY,
            formation=FormationType.WIDE, width=8, depth=2,
            movement=4, weapon_skill=3, ballistic_skill=4, strength=4,
            toughness=3, wounds=1, attacks=1, leadership=7,
            armor_save=6, player=1, color='lightblue', weapon_range=24
        )
        handgunners_2.armor_piercing = True
        handgunners_2.drilled = True
        
        # 5 Outriders [95 pts]
        outriders_1 = OldWorldUnit(
            name="Outriders I", x=8, y=18, facing=90,
            models=5, max_models=5, unit_type=UnitType.CAVALRY,
            formation=FormationType.WIDE, width=5, depth=1,
            movement=8, weapon_skill=4, ballistic_skill=4, strength=3,
            toughness=3, wounds=1, attacks=1, leadership=7,
            armor_save=4, player=1, color='cyan', weapon_range=18
        )
        outriders_1.fast_cavalry = True
        outriders_1.repeater_weapons = True  # Multiple shots
        
        # 5 Outriders [95 pts] 
        outriders_2 = OldWorldUnit(
            name="Outriders II", x=30, y=18, facing=90,
            models=5, max_models=5, unit_type=UnitType.CAVALRY,
            formation=FormationType.WIDE, width=5, depth=1,
            movement=8, weapon_skill=4, ballistic_skill=4, strength=3,
            toughness=3, wounds=1, attacks=1, leadership=7,
            armor_save=4, player=1, color='cyan', weapon_range=18
        )
        outriders_2.fast_cavalry = True
        outriders_2.repeater_weapons = True
        
        # SPECIAL UNITS
        # Great Cannon [130 pts]
        great_cannon_1 = OldWorldUnit(
            name="Great Cannon I", x=5, y=12, facing=90,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=0, ballistic_skill=4, strength=10,
            toughness=7, wounds=3, attacks=0, leadership=7,
            armor_save=6, player=1, color='navy', weapon_range=48
        )
        great_cannon_1.armor_piercing = True
        great_cannon_1.veteran = True
        great_cannon_1.vanguard = True
        
        # Great Cannon [130 pts]
        great_cannon_2 = OldWorldUnit(
            name="Great Cannon II", x=35, y=12, facing=90,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=0, ballistic_skill=4, strength=10,
            toughness=7, wounds=3, attacks=0, leadership=7,
            armor_save=6, player=1, color='navy', weapon_range=48
        )
        great_cannon_2.armor_piercing = True
        great_cannon_2.veteran = True
        great_cannon_2.vanguard = True
        
        # RARE UNITS
        # Helblaster Volley Gun [135 pts]
        helblaster_1 = OldWorldUnit(
            name="Helblaster I", x=10, y=8, facing=90,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=0, ballistic_skill=4, strength=5,
            toughness=7, wounds=3, attacks=0, leadership=7,
            armor_save=6, player=1, color='darkblue', weapon_range=24
        )
        helblaster_1.multiple_shots = 9  # D6+3 shots
        helblaster_1.veteran = True
        helblaster_1.vanguard = True
        
        # Helblaster Volley Gun [135 pts]
        helblaster_2 = OldWorldUnit(
            name="Helblaster II", x=20, y=8, facing=90,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=0, ballistic_skill=4, strength=5,
            toughness=7, wounds=3, attacks=0, leadership=7,
            armor_save=6, player=1, color='darkblue', weapon_range=24
        )
        helblaster_2.multiple_shots = 9
        helblaster_2.veteran = True
        helblaster_2.vanguard = True
        
        # Helblaster Volley Gun [135 pts]
        helblaster_3 = OldWorldUnit(
            name="Helblaster III", x=30, y=8, facing=90,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=0, ballistic_skill=4, strength=5,
            toughness=7, wounds=3, attacks=0, leadership=7,
            armor_save=6, player=1, color='darkblue', weapon_range=24
        )
        helblaster_3.multiple_shots = 9
        helblaster_3.veteran = True
        helblaster_3.vanguard = True
        
        armies.extend([general_hans, engineer_rifle, master_mage, nuln_veterans, 
                      handgunners_1, handgunners_2, outriders_1, outriders_2,
                      great_cannon_1, great_cannon_2, helblaster_1, helblaster_2, helblaster_3])
        
        # ===============================
        # TROLL HORDE (ORCS & GOBLINS) - Player 2
        # ===============================
        
        # CHARACTERS
        # Orc Bigboss on Boar Chariot [219 pts + 90 pts]
        bigboss_chariot = OldWorldUnit(
            name="Bigboss on Boar Chariot", x=55, y=25, facing=270,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=7, weapon_skill=5, ballistic_skill=3, strength=5,
            toughness=5, wounds=3, attacks=4, leadership=8,
            armor_save=4, player=2, color='orange'  # Heavy armour
        )
        bigboss_chariot.has_standard = True  # Battle Standard Bearer
        bigboss_chariot.chariot = True
        bigboss_chariot.impact_hits = True
        
        # Orc Warboss on Wyvern [346 pts]
        warboss_wyvern = OldWorldUnit(
            name="Warboss on Wyvern", x=58, y=30, facing=270,
            models=1, max_models=1, unit_type=UnitType.MONSTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=6, weapon_skill=6, ballistic_skill=3, strength=6,
            toughness=5, wounds=4, attacks=5, leadership=9,
            armor_save=3, player=2, color='darkred'  # Armour of Mork
        )
        warboss_wyvern.fly = True
        warboss_wyvern.terror = True
        warboss_wyvern.poison_attacks = True  # Venomous tail
        
        # Orc Weirdnob Level 4 [230 pts]
        weirdnob = OldWorldUnit(
            name="Orc Weirdnob", x=52, y=22, facing=270,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=4,
            toughness=4, wounds=3, attacks=2, leadership=7,
            armor_save=7, player=2, color='purple'
        )
        weirdnob.wizard_level = 4
        weirdnob.magic_items = ["Ruby Ring of Ruin", "Lore Familiar"]
        
        # CORE UNITS
        # 8 Common Trolls [360 pts]
        common_trolls = OldWorldUnit(
            name="Common Troll Mob", x=50, y=28, facing=270,
            models=8, max_models=8, unit_type=UnitType.MONSTER,
            formation=FormationType.DEEP, width=4, depth=2,
            movement=6, weapon_skill=3, ballistic_skill=1, strength=5,
            toughness=4, wounds=3, attacks=3, leadership=4,
            armor_save=5, player=2, color='green'  # Calloused Hide
        )
        common_trolls.fear = True
        common_trolls.regeneration = True
        common_trolls.stupidity = True
        common_trolls.troll_vomit = True  # Ranged attack
        
        # 27 Orc Boys with Warbows [172 pts]
        orc_boys = OldWorldUnit(
            name="Orc Mob (Warbows)", x=45, y=20, facing=270,
            models=27, max_models=27, unit_type=UnitType.INFANTRY,
            formation=FormationType.DEEP, width=6, depth=5,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=4, wounds=1, attacks=1, leadership=7,
            armor_save=7, player=2, color='red', weapon_range=18
        )
        orc_boys.has_standard = True
        orc_boys.frenzy = True
        
        # 4 River Trolls [212 pts]
        river_trolls_1 = OldWorldUnit(
            name="River Troll Mob I", x=60, y=35, facing=270,
            models=4, max_models=4, unit_type=UnitType.MONSTER,
            formation=FormationType.WIDE, width=4, depth=1,
            movement=6, weapon_skill=3, ballistic_skill=1, strength=5,
            toughness=5, wounds=3, attacks=3, leadership=4,
            armor_save=5, player=2, color='darkgreen'
        )
        river_trolls_1.fear = True
        river_trolls_1.terror = True  # River trolls cause terror
        river_trolls_1.regeneration = True
        river_trolls_1.stupidity = True
        river_trolls_1.troll_vomit = True
        
        # SPECIAL UNITS
        # 4 River Trolls [212 pts]
        river_trolls_2 = OldWorldUnit(
            name="River Troll Mob II", x=42, y=32, facing=270,
            models=4, max_models=4, unit_type=UnitType.MONSTER,
            formation=FormationType.WIDE, width=4, depth=1,
            movement=6, weapon_skill=3, ballistic_skill=1, strength=5,
            toughness=5, wounds=3, attacks=3, leadership=4,
            armor_save=5, player=2, color='darkgreen'
        )
        river_trolls_2.fear = True
        river_trolls_2.terror = True
        river_trolls_2.regeneration = True
        river_trolls_2.stupidity = True
        river_trolls_2.troll_vomit = True
        
        # 4 River Trolls [212 pts]
        river_trolls_3 = OldWorldUnit(
            name="River Troll Mob III", x=48, y=15, facing=270,
            models=4, max_models=4, unit_type=UnitType.MONSTER,
            formation=FormationType.WIDE, width=4, depth=1,
            movement=6, weapon_skill=3, ballistic_skill=1, strength=5,
            toughness=5, wounds=3, attacks=3, leadership=4,
            armor_save=5, player=2, color='darkgreen'
        )
        river_trolls_3.fear = True
        river_trolls_3.terror = True
        river_trolls_3.regeneration = True
        river_trolls_3.stupidity = True
        river_trolls_3.troll_vomit = True
        
        armies.extend([bigboss_chariot, warboss_wyvern, weirdnob, common_trolls, 
                      orc_boys, river_trolls_1, river_trolls_2, river_trolls_3])
        
        return armies

def main():
    """Launch the Nuln vs Troll Horde battle"""
    print("🔥 ARMY OF NULN vs TROLL HORDE 🔥")
    print("=" * 50)
    print("⚔️  Authentic Old World Battle System  ⚔️")
    print("🏹 Engineering Precision vs Monstrous Fury 🧌")
    print()
    
    try:
        battle = NulnVsTrollBattle()
        battle.run_battle()
    except Exception as e:
        print(f"Battle error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏆 Battle Complete! May Sigmar guide the cannons! 🏆")

if __name__ == "__main__":
    main() 