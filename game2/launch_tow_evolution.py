#!/usr/bin/env python3
"""
🏛️ WARHAMMER: THE OLD WORLD - EVOLUTION LAUNCHER
===============================================

Main launcher for the TOW AI evolution system.
Choose between quick demo or full 100,000 battle evolution!
"""

import sys
import os
from tow_evolution_ai import run_tow_evolution
from demo_evolution import quick_evolution_demo

def display_banner():
    """Display epic banner"""
    print("🏛️" * 20)
    print()
    print("    WARHAMMER: THE OLD WORLD")
    print("      AI EVOLUTION SYSTEM")
    print()
    print("🧬 Learn • Evolve • Dominate 🧬")
    print()
    print("🏛️" * 20)
    print()

def display_menu():
    """Display main menu"""
    print("🎯 CHOOSE YOUR EVOLUTION PATH:")
    print("=" * 50)
    print("1. 🚀 Quick Demo (1,000 battles, ~1 minute)")
    print("   - See AI evolution in action")
    print("   - 10 generations, 20 AI per faction")
    print("   - Perfect for testing and demonstration")
    print()
    print("2. ⚔️ Full Evolution (100,000 battles, ~several hours)")
    print("   - Complete AI evolution experience")
    print("   - 1,000 generations, 50 AI per faction")
    print("   - Discover optimal army compositions")
    print()
    print("3. 🛠️ Custom Evolution (choose your parameters)")
    print("   - Set your own battle count and population")
    print("   - Flexible evolution for specific testing")
    print()
    print("4. 📊 View Previous Results")
    print("   - Load and analyze previous evolution runs")
    print()
    print("5. 🏛️ Exit")
    print("=" * 50)

def get_user_choice():
    """Get user menu choice"""
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return int(choice)
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        except KeyboardInterrupt:
            print("\n👋 Evolution cancelled by user.")
            sys.exit(0)

def run_custom_evolution():
    """Run custom evolution with user parameters"""
    print("\n🛠️ CUSTOM EVOLUTION SETUP")
    print("=" * 40)
    
    try:
        battles = int(input("Enter total number of battles (default 10000): ") or "10000")
        population = int(input("Enter population size per faction (default 30): ") or "30")
        battles_per_gen = int(input("Enter battles per generation (default 100): ") or "100")
        
        print(f"\n🧬 Custom Evolution Parameters:")
        print(f"   Total battles: {battles:,}")
        print(f"   Population size: {population}")
        print(f"   Battles per generation: {battles_per_gen}")
        print(f"   Estimated generations: {battles // battles_per_gen}")
        
        confirm = input("\nProceed with custom evolution? (y/n): ").strip().lower()
        if confirm == 'y':
            from tow_evolution_ai import run_tow_evolution
            return run_tow_evolution(battles=battles, generations=battles//battles_per_gen)
        else:
            print("❌ Custom evolution cancelled.")
            return None
            
    except ValueError:
        print("❌ Invalid input. Please enter numbers only.")
        return None
    except KeyboardInterrupt:
        print("\n👋 Evolution cancelled by user.")
        return None

def view_previous_results():
    """View results from previous evolution runs"""
    print("\n📊 PREVIOUS EVOLUTION RESULTS")
    print("=" * 40)
    
    # Look for saved evolution files
    evolution_files = []
    for filename in os.listdir('.'):
        if filename.endswith('.pkl') and 'evolution' in filename:
            evolution_files.append(filename)
    
    if not evolution_files:
        print("❌ No previous evolution results found.")
        print("   Run an evolution first to generate results.")
        return
    
    print("Found evolution result files:")
    for i, filename in enumerate(evolution_files, 1):
        print(f"  {i}. {filename}")
    
    try:
        choice = int(input(f"\nChoose file to view (1-{len(evolution_files)}): "))
        if 1 <= choice <= len(evolution_files):
            filename = evolution_files[choice - 1]
            # Load and display results
            import pickle
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            
            stats = state['stats']
            print(f"\n📈 EVOLUTION RESULTS FROM {filename}")
            print("=" * 50)
            print(f"Generations: {stats.generation}")
            print(f"Battles fought: {stats.battles_fought:,}")
            print(f"Orc wins: {stats.orc_wins:,} ({stats.orc_win_rate:.1%})")
            print(f"Nuln wins: {stats.nuln_wins:,} ({stats.nuln_win_rate:.1%})")
            print(f"Draws: {stats.draws:,}")
            print(f"Average battle length: {stats.average_battle_length:.1f} turns")
            
            if stats.best_orc_army:
                print(f"\n🏆 Best Orc Army:")
                print(stats.best_orc_army.get_army_summary())
            
            if stats.best_nuln_army:
                print(f"\n🏆 Best Nuln Army:")
                print(stats.best_nuln_army.get_army_summary())
        else:
            print("❌ Invalid choice.")
    except (ValueError, FileNotFoundError, EOFError):
        print("❌ Error loading evolution results.")

def main():
    """Main launcher function"""
    display_banner()
    
    while True:
        display_menu()
        choice = get_user_choice()
        
        if choice == 1:
            # Quick Demo
            print("\n🚀 STARTING QUICK EVOLUTION DEMO")
            print("=" * 50)
            try:
                result = quick_evolution_demo()
                print("\n✅ Demo completed successfully!")
                input("\nPress Enter to continue...")
            except KeyboardInterrupt:
                print("\n👋 Demo cancelled by user.")
            except Exception as e:
                print(f"❌ Demo error: {e}")
        
        elif choice == 2:
            # Full Evolution
            print("\n⚔️ STARTING FULL EVOLUTION (100,000 BATTLES)")
            print("=" * 50)
            print("⚠️ This will take several hours to complete!")
            print("💾 Progress will be saved automatically every 100 generations.")
            print("🛑 Press Ctrl+C at any time to safely interrupt and save progress.")
            print("\n🚀 Starting evolution in 3 seconds...")
            
            import time
            try:
                time.sleep(3)
                result = run_tow_evolution()
                print("\n🎉 Full evolution completed successfully!")
                input("\nPress Enter to continue...")
            except KeyboardInterrupt:
                print("\n👋 Evolution cancelled by user. Progress has been saved.")
            except Exception as e:
                print(f"❌ Evolution error: {e}")
                input("\nPress Enter to continue...")
        
        elif choice == 3:
            # Custom Evolution
            result = run_custom_evolution()
            if result:
                print("\n✅ Custom evolution completed successfully!")
                input("\nPress Enter to continue...")
        
        elif choice == 4:
            # View Previous Results
            view_previous_results()
            input("\nPress Enter to continue...")
        
        elif choice == 5:
            # Exit
            print("\n👋 Thank you for using the TOW AI Evolution System!")
            print("🏛️ May your armies evolve into legends!")
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Evolution system shutting down. Goodbye!")
        sys.exit(0) 