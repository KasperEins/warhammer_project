#!/usr/bin/env python3
"""
🏛️ WARHAMMER: THE OLD WORLD - EVOLUTION DEMO
===========================================

Quick demonstration of the AI evolution system
showing how it learns and adapts army compositions.
"""

from tow_evolution_ai import *
import time

def quick_evolution_demo():
    """Run a quick demo with fewer battles"""
    print("🏛️ WARHAMMER: THE OLD WORLD - AI EVOLUTION DEMO")
    print("=" * 60)
    print("🎯 Running quick demo: 1,000 battles across 10 generations")
    print("🧬 Population: 20 AI generals per faction")
    print("📖 Using comprehensive TOW rules!")
    print("⚔️ Let the evolution begin!")
    
    # Create smaller evolution system for demo using new enhanced AI
    orc_ai = EnhancedEvolutionAI("Orc & Goblin Tribes", population_size=20)
    nuln_ai = EnhancedEvolutionAI("City-State of Nuln", population_size=20)
    
    # Show initial random armies
    print("\n📊 INITIAL RANDOM ARMIES")
    print("-" * 40)
    
    # Generate sample armies from initial DNA
    initial_orc_individual = orc_ai.population[0]
    initial_nuln_individual = nuln_ai.population[0]
    
    initial_orc_comp = orc_ai.dna_to_army_composition(initial_orc_individual["dna"])
    initial_nuln_comp = nuln_ai.dna_to_army_composition(initial_nuln_individual["dna"])
    
    print("🔍 Sample Initial Orc Army:")
    for unit, count in initial_orc_comp.items():
        if count > 0:
            print(f"   {unit}: {count}")
    
    print("\n🔍 Sample Initial Nuln Army:")
    for unit, count in initial_nuln_comp.items():
        if count > 0:
            print(f"   {unit}: {count}")
    
    # Run evolution for demo
    start_time = time.time()
    
    for generation in range(10):  # 10 generations
        print(f"\n🧬 Generation {generation + 1}/10...")
        
        # Evolve both factions
        orc_ai.evolve_generation_comprehensive()
        nuln_ai.evolve_generation_comprehensive()
        
        # Show progress every few generations
        if (generation + 1) % 3 == 0:
            print(f"\n📈 GENERATION {generation + 1} RESULTS:")
            print(f"   Orc best fitness: {orc_ai.population[0]['fitness']:.2f}")
            print(f"   Nuln best fitness: {nuln_ai.population[0]['fitness']:.2f}")
            
            # Show tactical insights if available
            if orc_ai.learning_history["tactical_insights"]:
                print(f"   🧌 Orc insights: {orc_ai.learning_history['tactical_insights'][-1]}")
            if nuln_ai.learning_history["tactical_insights"]:
                print(f"   🏰 Nuln insights: {nuln_ai.learning_history['tactical_insights'][-1]}")
    
    evolution_time = time.time() - start_time
    
    print("\n🎉 DEMO EVOLUTION COMPLETE!")
    print("=" * 50)
    print(f"⏱️ Evolution time: {evolution_time:.1f} seconds")
    print(f"🧬 Generations: 10")
    
    # Show evolved armies
    print("\n🧬 EVOLVED ARMIES")
    print("-" * 40)
    
    # Get best evolved individuals
    best_orc = orc_ai.population[0]
    best_nuln = nuln_ai.population[0]
    
    # Generate evolved armies
    evolved_orc_comp = orc_ai.dna_to_army_composition(best_orc["dna"])
    evolved_nuln_comp = nuln_ai.dna_to_army_composition(best_nuln["dna"])
    
    print("🏆 Best Evolved Orc Army:")
    print(f"   Fitness: {best_orc['fitness']:.2f}")
    for unit, count in evolved_orc_comp.items():
        if count > 0:
            print(f"   {unit}: {count}")
    
    print("\n🏆 Best Evolved Nuln Army:")
    print(f"   Fitness: {best_nuln['fitness']:.2f}")
    for unit, count in evolved_nuln_comp.items():
        if count > 0:
            print(f"   {unit}: {count}")
    
    # Show DNA evolution insights
    print("\n🧬 DNA EVOLUTION INSIGHTS")
    print("-" * 40)
    print(f"Best Orc DNA preferences:")
    print(f"   Aggression: {best_orc['dna'].get('aggression', 0):.2f}")
    print(f"   Magic Focus: {best_orc['dna'].get('magic_focus', 0):.2f}")
    print(f"   Ranged Focus: {best_orc['dna'].get('ranged_focus', 0):.2f}")
    print(f"   Elite Focus: {best_orc['dna'].get('elite_focus', 0):.2f}")
    
    print(f"\nBest Nuln DNA preferences:")
    print(f"   Aggression: {best_nuln['dna'].get('aggression', 0):.2f}")
    print(f"   Magic Focus: {best_nuln['dna'].get('magic_focus', 0):.2f}")
    print(f"   Ranged Focus: {best_nuln['dna'].get('ranged_focus', 0):.2f}")
    print(f"   Elite Focus: {best_nuln['dna'].get('elite_focus', 0):.2f}")
    
    # Show most preferred units
    if "unit_preferences" in best_orc["dna"]:
        print(f"\nTop 3 Orc units by evolved preference:")
        orc_prefs = sorted(best_orc["dna"]["unit_preferences"].items(), key=lambda x: x[1], reverse=True)[:3]
        for unit_name, preference in orc_prefs:
            print(f"   {unit_name}: {preference:.2f}")
    
    if "unit_preferences" in best_nuln["dna"]:
        print(f"\nTop 3 Nuln units by evolved preference:")
        nuln_prefs = sorted(best_nuln["dna"]["unit_preferences"].items(), key=lambda x: x[1], reverse=True)[:3]
        for unit_name, preference in nuln_prefs:
            print(f"   {unit_name}: {preference:.2f}")
    
    print("\n🚀 READY FOR FULL 100,000 BATTLE EVOLUTION!")
    print("   The AI has learned authentic TOW tactics!")
    
    return {
        "orc_ai": orc_ai,
        "nuln_ai": nuln_ai,
        "generations": 10,
        "evolution_time": evolution_time
    }

if __name__ == "__main__":
    demo_result = quick_evolution_demo() 