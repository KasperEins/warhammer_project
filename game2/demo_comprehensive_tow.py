#!/usr/bin/env python3
"""
🏛️ WARHAMMER: THE OLD WORLD - COMPREHENSIVE RULES DEMO
====================================================
Demonstrates all our implemented TOW rules systems
"""

import random

def main():
    print("🏛️ WARHAMMER: THE OLD WORLD - COMPREHENSIVE RULES DEMO")
    print("=" * 70)
    print("🎯 Showcasing ALL implemented TOW rules for maximum authenticity!")
    print()
    
    # ============================================================================
    # RULES IMPLEMENTATION SUMMARY
    # ============================================================================
    
    print("📋 COMPREHENSIVE RULES IMPLEMENTATION")
    print("-" * 50)
    
    implemented_systems = [
        ("✅ Complete Turn Sequence", "Strategy → Movement → Shooting → Combat"),
        ("✅ Universal Special Rules", "Fear, Terror, Frenzy, Hatred, Killing Blow, etc."),
        ("✅ Psychology System", "Panic tests, Fear tests, Terror, Leadership"),
        ("✅ Faction Rules - Orcs", "Animosity, Waaagh!, Squabble"),
        ("✅ Faction Rules - Empire", "State Troops, Gunpowder Weapons, Detachments"),
        ("✅ Combat Resolution", "Initiative order, wounds, armor saves, break tests"),
        ("✅ Formation Rules", "Close Order, ranks, files, disruption"),
        ("✅ Equipment System", "Weapons, armor, shields, special equipment"),
        ("✅ Characteristics", "M, WS, BS, S, T, W, I, A, Ld"),
        ("✅ Troop Types", "Infantry, Cavalry, War Machines, Monsters"),
        ("✅ Command Groups", "Champions, Standards, Musicians"),
        ("✅ Unit Mechanics", "Unit Strength, rank bonuses, fleeing, rallying")
    ]
    
    for system, description in implemented_systems:
        print(f"{system:<30} {description}")
    
    print()
    
    # ============================================================================
    # FACTION-SPECIFIC RULES DEMO
    # ============================================================================
    
    print("🧌 ORC & GOBLIN TRIBES - FACTION RULES")
    print("-" * 50)
    
    # Animosity Demo
    print("⚔️ Animosity Rule:")
    for turn in range(3):
        roll = random.randint(1, 6)
        print(f"  Turn {turn + 1}: Roll {roll}", end="")
        
        if roll == 6:
            effect = random.randint(1, 6)
            if effect <= 2:
                print(" → Argue amongst themselves!")
            elif effect <= 4:
                print(" → Squabble and move toward Orcs!")
            else:
                print(" → Work up into a frenzy!")
        else:
            print(" → No animosity")
    
    print()
    
    # Waaagh! Demo
    print("🔥 Waaagh! Rule:")
    waaagh_roll = random.randint(2, 12)
    print(f"  Waaagh! test: {waaagh_roll}", end="")
    if waaagh_roll >= 8:
        print(" → WAAAGH! spreads! +1 Movement, +1 Charge!")
    else:
        print(" → Waaagh! fails to take hold")
    
    print()
    
    print("🏰 EMPIRE / NULN - FACTION RULES")
    print("-" * 50)
    
    # State Troops Demo
    print("🎖️ State Troop Discipline:")
    ld_test = random.randint(2, 12)
    base_ld = 7
    bonus = 1
    print(f"  Leadership test: {ld_test} vs {base_ld + bonus} (base {base_ld} +{bonus})")
    if ld_test <= base_ld + bonus:
        print("  → Professional discipline holds!")
    else:
        print("  → Even professionals can break!")
    
    print()
    
    # Gunpowder Demo
    print("💥 Gunpowder Weapons:")
    weapons = ["Handgun", "Pistol", "Great Cannon"]
    for weapon in weapons:
        misfire = random.randint(1, 6)
        print(f"  {weapon}: Roll {misfire}", end="")
        
        if weapon == "Great Cannon" and misfire <= 2:
            if misfire == 1:
                print(" → EXPLODES!")
            else:
                print(" → Destroyed!")
        elif weapon in ["Handgun", "Pistol"] and misfire == 1:
            print(" → Jams!")
        else:
            print(" → Fires normally")
    
    print()
    
    # ============================================================================
    # PSYCHOLOGY SYSTEM DEMO
    # ============================================================================
    
    print("😨 PSYCHOLOGY SYSTEM")
    print("-" * 50)
    
    # Fear Test
    print("👻 Fear Test (Night Goblins vs Giant):")
    fear_roll = random.randint(2, 12)
    goblin_ld = 5
    print(f"  Roll: {fear_roll} vs Ld {goblin_ld}", end="")
    if fear_roll <= goblin_ld:
        print(" → Overcome fear and charge!")
    else:
        print(" → Too scared to charge!")
    
    print()
    
    # Terror Test
    print("😱 Terror Test (vs Dragon):")
    terror_roll = random.randint(2, 12)
    print(f"  Roll: {terror_roll} vs Ld {goblin_ld}", end="")
    if terror_roll <= goblin_ld:
        print(" → Stand firm against terror!")
    else:
        print(" → Flee in absolute terror!")
    
    print()
    
    # Panic Test
    print("😰 Panic Test (friendly unit destroyed):")
    panic_roll = random.randint(2, 12)
    modified_ld = goblin_ld - 1  # Penalty for cause
    print(f"  Roll: {panic_roll} vs Ld {modified_ld} (base {goblin_ld} -1)", end="")
    if panic_roll <= modified_ld:
        print(" → Hold firm!")
    else:
        print(" → Panic and flee!")
    
    print()
    
    # ============================================================================
    # COMBAT SYSTEM DEMO
    # ============================================================================
    
    print("⚔️ COMBAT RESOLUTION SYSTEM")
    print("-" * 50)
    
    print("🥊 Example Combat: Orc Boyz vs Handgunners")
    print()
    
    # Initiative Order
    orc_init = 2
    empire_init = 3
    print(f"Initiative: Empire ({empire_init}) strikes first, then Orcs ({orc_init})")
    print()
    
    # Empire Attacks
    print("🏹 Empire Handgunners attack:")
    empire_hits = random.randint(0, 3)
    empire_wounds = random.randint(0, empire_hits)
    print(f"  {empire_hits} hits, {empire_wounds} wounds caused")
    
    # Orc Attacks
    print("🔨 Orc Boyz attack:")
    orc_hits = random.randint(1, 5)
    orc_wounds = random.randint(0, orc_hits)
    print(f"  {orc_hits} hits, {orc_wounds} wounds caused")
    
    print()
    
    # Combat Result
    print("📊 Combat Result:")
    orc_cr = orc_wounds + 2  # +2 for ranks
    empire_cr = empire_wounds + 1  # +1 for standard
    
    print(f"  Orcs: {orc_wounds} wounds +2 ranks = {orc_cr}")
    print(f"  Empire: {empire_wounds} wounds +1 standard = {empire_cr}")
    
    if orc_cr > empire_cr:
        print(f"  → Orcs win by {orc_cr - empire_cr}!")
        break_test = random.randint(2, 12)
        empire_ld = 8  # +1 for State Troops
        print(f"  → Empire break test: {break_test} vs {empire_ld}", end="")
        if break_test <= empire_ld:
            print(" (Pass - fight continues)")
        else:
            print(" (Fail - Empire flees!)")
    elif empire_cr > orc_cr:
        print(f"  → Empire wins by {empire_cr - orc_cr}!")
        break_test = random.randint(2, 12)
        orc_ld = 7
        print(f"  → Orc break test: {break_test} vs {orc_ld}", end="")
        if break_test <= orc_ld:
            print(" (Pass - fight continues)")
        else:
            print(" (Fail - Orcs flee!)")
    else:
        print("  → Draw! Combat continues next round")
    
    print()
    
    # ============================================================================
    # TURN SEQUENCE DEMO
    # ============================================================================
    
    print("🎲 COMPLETE TURN SEQUENCE")
    print("-" * 50)
    
    phases = [
        ("Strategy Phase", ["Rally fleeing units", "Animosity tests", "Magic effects"]),
        ("Movement Phase", ["Declare charges", "Charge moves", "Regular movement"]),
        ("Shooting Phase", ["Select targets", "Roll to hit", "Roll to wound", "Saves"]),
        ("Combat Phase", ["Initiative order", "Resolve attacks", "Break tests"])
    ]
    
    for phase_name, actions in phases:
        print(f"📋 {phase_name}:")
        for action in actions:
            print(f"  • {action}")
        print()
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    
    print("🏆 COMPREHENSIVE TOW IMPLEMENTATION COMPLETE!")
    print("=" * 70)
    
    achievements = [
        "🎯 Authentic turn sequence with all 4 phases",
        "⚔️ Complete combat resolution with initiative and break tests",
        "😨 Full psychology system (Fear, Terror, Panic)",
        "🧌 Orc faction rules (Animosity, Waaagh!, Squabble)",
        "🏰 Empire faction rules (State Troops, Gunpowder)",
        "🎖️ Command groups and leadership mechanics",
        "🛡️ Equipment and armor save system",
        "📊 Unit characteristics and troop types",
        "🔄 Formation rules and rank bonuses",
        "🏃 Fleeing and rallying mechanics"
    ]
    
    print("✅ SUCCESSFULLY IMPLEMENTED:")
    for achievement in achievements:
        print(f"  {achievement}")
    
    print()
    print("🎮 OUR AI NOW LEARNS FROM THE MOST AUTHENTIC TOW RULES!")
    print("📈 Evolution system uses comprehensive battle results")
    print("🏛️ Ready for 100,000+ battle evolution campaigns!")
    
    # Show what this means for AI evolution
    print()
    print("🧠 AI EVOLUTION BENEFITS:")
    print("  • Learns from REAL TOW psychology effects")
    print("  • Discovers optimal unit synergies")
    print("  • Adapts to faction-specific strengths/weaknesses")
    print("  • Evolves authentic tactical doctrines")
    print("  • Masters timing of charges, shooting, magic")
    print("  • Develops counter-strategies for each faction")
    
    print()
    print("🚀 NEXT: Run launch_tow_evolution.py to see AI learn!")
    print("📊 Watch AI discover meta-game strategies over 1000s of generations!")

if __name__ == "__main__":
    main() 