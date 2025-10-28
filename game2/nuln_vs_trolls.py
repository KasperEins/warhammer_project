#!/usr/bin/env python3
"""
NULN ARMY vs TROLL HORDE - Authentic Old World Battle
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from old_world_battle import OldWorldBattle, OldWorldUnit, UnitType, FormationType

class NulnVsTrollBattle(OldWorldBattle):
    def create_armies(self):
        """Create authentic Nuln Army vs Troll Horde armies - positioned closer for action"""
        armies = []
        
        # ===============================
        # ARMY OF NULN (EMPIRE) - Player 1 [2000 pts]
        # ===============================
        
        # ++ Characters [770 pts] ++
        
        # General Hans von Löwenhacke [190 pts]
        general_hans = OldWorldUnit(
            name="General Hans von Löwenhacke", x=15, y=24, facing=90,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=4, weapon_skill=6, ballistic_skill=5, strength=4,
            toughness=4, wounds=3, attacks=4, leadership=9,
            armor_save=2, player=1, color='gold', points_cost=190
        )
        general_hans.has_standard = True
        general_hans.immune_to_fear = True
        
        # Empire Engineer [55 pts]
        engineer = OldWorldUnit(
            name="Empire Engineer", x=8, y=12, facing=90,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=4, weapon_skill=3, ballistic_skill=4, strength=3,
            toughness=3, wounds=2, attacks=1, leadership=8,
            armor_save=6, player=1, color='lightgray', weapon_range=30, points_cost=55
        )
        
        # Master Mage [90 pts] - Level 2 Wizard
        master_mage = OldWorldUnit(
            name="Master Mage", x=22, y=18, facing=90,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=2, attacks=1, leadership=8,
            armor_save=6, player=1, color='purple', points_cost=90
        )
        
        # Empire Engineer with War Wagon [185 pts]
        engineer_war_wagon = OldWorldUnit(
            name="Engineer (War Wagon)", x=28, y=8, facing=90,
            models=1, max_models=1, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=3, weapon_skill=4, ballistic_skill=5, strength=4,
            toughness=5, wounds=3, attacks=2, leadership=8,
            armor_save=4, player=1, color='gray', weapon_range=36, points_cost=185
        )
        
        # General of the Empire on Imperial Griffon [250 pts]
        general_griffon = OldWorldUnit(
            name="General on Imperial Griffon", x=18, y=30, facing=90,
            models=1, max_models=1, unit_type=UnitType.MONSTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=6, weapon_skill=6, ballistic_skill=0, strength=5,
            toughness=4, wounds=4, attacks=4, leadership=9,
            armor_save=4, player=1, color='goldenrod', points_cost=250
        )
        general_griffon.terror = True
        general_griffon.immune_to_fear = True
        
        # ++ Core Units [565 pts] ++
        
        # 20 Nuln Veteran State Troops with integrated missile troops [375 pts]
        nuln_veterans = OldWorldUnit(
            name="Nuln Veteran State Troops", x=12, y=25, facing=90,
            models=40, max_models=40, unit_type=UnitType.INFANTRY,
            formation=FormationType.DEEP, width=8, depth=5,
            movement=4, weapon_skill=4, ballistic_skill=4, strength=3,
            toughness=3, wounds=1, attacks=1, leadership=8,
            armor_save=5, player=1, color='blue', weapon_range=24, points_cost=375
        )
        nuln_veterans.has_standard = True
        nuln_veterans.has_musician = True
        
        # 5 Outriders I [95 pts]
        outriders_1 = OldWorldUnit(
            name="Outriders I", x=4, y=20, facing=90,
            models=5, max_models=5, unit_type=UnitType.CAVALRY,
            formation=FormationType.WIDE, width=5, depth=1,
            movement=8, weapon_skill=3, ballistic_skill=4, strength=3,
            toughness=3, wounds=1, attacks=1, leadership=7,
            armor_save=4, player=1, color='cyan', weapon_range=24, points_cost=95
        )
        outriders_1.fast_cavalry = True
        
        # 5 Outriders II [95 pts]
        outriders_2 = OldWorldUnit(
            name="Outriders II", x=4, y=28, facing=90,
            models=5, max_models=5, unit_type=UnitType.CAVALRY,
            formation=FormationType.WIDE, width=5, depth=1,
            movement=8, weapon_skill=3, ballistic_skill=4, strength=3,
            toughness=3, wounds=1, attacks=1, leadership=7,
            armor_save=4, player=1, color='cyan', weapon_range=24, points_cost=95
        )
        outriders_2.fast_cavalry = True
        
        # ++ Special Units [260 pts] ++
        
        # Great Cannon I [130 pts] - Veteran, Vanguard
        great_cannon_1 = OldWorldUnit(
            name="Great Cannon I (Veteran)", x=5, y=12, facing=90,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=4, ballistic_skill=5, strength=10,
            toughness=7, wounds=3, attacks=0, leadership=8,
            armor_save=6, player=1, color='navy', weapon_range=48, points_cost=130
        )
        great_cannon_1.armor_piercing = True
        
        # Great Cannon II [130 pts] - Veteran, Vanguard
        great_cannon_2 = OldWorldUnit(
            name="Great Cannon II (Veteran)", x=32, y=12, facing=90,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=4, ballistic_skill=5, strength=10,
            toughness=7, wounds=3, attacks=0, leadership=8,
            armor_save=6, player=1, color='navy', weapon_range=48, points_cost=130
        )
        great_cannon_2.armor_piercing = True
        
        # ++ Rare Units [405 pts] ++
        
        # Helblaster Volley Gun I [135 pts] - Veteran, Vanguard
        helblaster_1 = OldWorldUnit(
            name="Helblaster I (Veteran)", x=8, y=8, facing=90,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=4, ballistic_skill=5, strength=5,
            toughness=7, wounds=3, attacks=0, leadership=8,
            armor_save=6, player=1, color='darkblue', weapon_range=24, points_cost=135
        )
        
        # Helblaster Volley Gun II [135 pts] - Veteran, Vanguard
        helblaster_2 = OldWorldUnit(
            name="Helblaster II (Veteran)", x=18, y=8, facing=90,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=4, ballistic_skill=5, strength=5,
            toughness=7, wounds=3, attacks=0, leadership=8,
            armor_save=6, player=1, color='darkblue', weapon_range=24, points_cost=135
        )
        
        # Helblaster Volley Gun III [135 pts] - Veteran, Vanguard
        helblaster_3 = OldWorldUnit(
            name="Helblaster III (Veteran)", x=28, y=8, facing=90,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=4, ballistic_skill=5, strength=5,
            toughness=7, wounds=3, attacks=0, leadership=8,
            armor_save=6, player=1, color='darkblue', weapon_range=24, points_cost=135
        )
        
        armies.extend([general_hans, engineer, master_mage, engineer_war_wagon, general_griffon,
                      nuln_veterans, outriders_1, outriders_2, 
                      great_cannon_1, great_cannon_2,
                      helblaster_1, helblaster_2, helblaster_3])
        
        # ===============================
        # TROLL HORDE (ORCS & GOBLINS) - Player 2 [1963 pts]
        # ===============================
        
        # ++ Characters [795 pts] ++
        
        # Orc Bigboss [219 pts] + Orc Boar Chariot [90 pts] = 309 pts total
        bigboss_chariot = OldWorldUnit(
            name="Bigboss on Boar Chariot", x=45, y=28, facing=270,
            models=1, max_models=1, unit_type=UnitType.CAVALRY,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=7, weapon_skill=5, ballistic_skill=3, strength=5,
            toughness=5, wounds=3, attacks=4, leadership=8,
            armor_save=4, player=2, color='maroon', points_cost=309
        )
        bigboss_chariot.has_standard = True  # Battle Standard Bearer
        
        # Orc Warboss on Wyvern [346 pts]
        warboss_wyvern = OldWorldUnit(
            name="Warboss on Wyvern", x=50, y=32, facing=270,
            models=1, max_models=1, unit_type=UnitType.MONSTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=6, weapon_skill=6, ballistic_skill=3, strength=6,
            toughness=5, wounds=4, attacks=5, leadership=9,
            armor_save=3, player=2, color='darkred', points_cost=346
        )
        warboss_wyvern.terror = True
        warboss_wyvern.immune_to_fear = True
        
        # Orc Weirdnob [230 pts] - Level 4 Wizard
        weirdnob = OldWorldUnit(
            name="Weirdnob", x=48, y=25, facing=270,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=4,
            toughness=4, wounds=2, attacks=2, leadership=7,
            armor_save=6, player=2, color='purple', points_cost=230
        )
        
        # ++ Core Units [744 pts] ++
        
        # 8x Common Trolls [360 pts] - Core unit
        common_trolls = OldWorldUnit(
            name="Common Troll Mob", x=52, y=20, facing=270,
            models=8, max_models=8, unit_type=UnitType.MONSTER,
            formation=FormationType.WIDE, width=4, depth=2,
            movement=6, weapon_skill=3, ballistic_skill=1, strength=5,
            toughness=4, wounds=3, attacks=3, leadership=4,
            armor_save=6, player=2, color='darkgreen', points_cost=360
        )
        common_trolls.fear = True
        common_trolls.regeneration = True
        common_trolls.stupidity = True
        
        # 27x Orc Boys [172 pts] - Core unit  
        orc_boys = OldWorldUnit(
            name="Orc Mob", x=56, y=24, facing=270,
            models=27, max_models=27, unit_type=UnitType.INFANTRY,
            formation=FormationType.DEEP, width=6, depth=5,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=4, wounds=1, attacks=1, leadership=7,
            armor_save=6, player=2, color='green', weapon_range=24, points_cost=172
        )
        orc_boys.has_standard = True
        
        # ++ Special [424 pts] ++
        
        # 4x River Trolls I [212 pts] - Special unit
        river_trolls_1 = OldWorldUnit(
            name="River Troll Mob I", x=45, y=35, facing=270,
            models=4, max_models=4, unit_type=UnitType.MONSTER,
            formation=FormationType.WIDE, width=4, depth=1,
            movement=6, weapon_skill=3, ballistic_skill=1, strength=5,
            toughness=5, wounds=3, attacks=3, leadership=4,
            armor_save=5, player=2, color='blue', points_cost=212
        )
        river_trolls_1.fear = True
        river_trolls_1.terror = True
        river_trolls_1.regeneration = True
        river_trolls_1.stupidity = True
        
        # 4x River Trolls II [212 pts] - Special unit  
        river_trolls_2 = OldWorldUnit(
            name="River Troll Mob II", x=50, y=15, facing=270,
            models=4, max_models=4, unit_type=UnitType.MONSTER,
            formation=FormationType.WIDE, width=4, depth=1,
            movement=6, weapon_skill=3, ballistic_skill=1, strength=5,
            toughness=5, wounds=3, attacks=3, leadership=4,
            armor_save=5, player=2, color='blue', points_cost=212
        )
        river_trolls_2.fear = True
        river_trolls_2.terror = True
        river_trolls_2.regeneration = True
        river_trolls_2.stupidity = True
        
        armies.extend([bigboss_chariot, warboss_wyvern, weirdnob,
                      common_trolls, orc_boys, river_trolls_1, river_trolls_2])
        
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
    
    print("\n🏆 Battle Complete! 🏆")

if __name__ == "__main__":
    main() 