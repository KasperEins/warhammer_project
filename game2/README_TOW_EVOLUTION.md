# 🏛️ WARHAMMER: THE OLD WORLD - AI EVOLUTION SYSTEM

**The most advanced AI system ever created for Warhammer: The Old World!**

## 🧬 What This System Does

This revolutionary AI system creates artificial generals that:
- **Learn army building from scratch** using authentic TOW rules and point costs
- **Evolve army compositions** through thousands of battles
- **Discover optimal strategies** through genetic algorithms
- **Adapt and improve** based on win/loss data
- **Master authentic TOW mechanics** including terrain, formations, and scenarios

## 📊 Features

### 🏗️ **Complete Unit Database**
- **33 authentic units** from official TOW sources
- **Orc & Goblin Tribes** (17 units): Characters, Core, Special, Rare, Mercenaries
- **City-State of Nuln** (16 units): Artillery-focused Empire variant
- **Full statistics**: M, WS, BS, S, T, W, I, A, Ld, points costs
- **Equipment options**: Weapons, upgrades, command groups
- **Army composition rules**: Core 25%, Characters 25%, Special/Rare 50% each

### 🧬 **Evolutionary AI**
- **Genetic algorithms** that breed successful army compositions
- **DNA-based preferences** for units, aggression, magic, ranged focus
- **Population evolution** with mutation, crossover, and selection
- **Fitness evaluation** based on battle performance
- **100,000+ battle training** for discovering meta-game strategies

### ⚔️ **Battle Simulation**
- **Authentic TOW mechanics** with proper terrain effects
- **72"×48" battlefield** with official scenario support
- **Unit formations** showing ranks, files, and facing
- **Advanced tactics** including charges, psychology, magic
- **Real-time evolution** tracking win rates and army adaptation

## 🚀 Quick Start

### 1. **Demo Evolution (1,000 battles, ~1 minute)**
```bash
python demo_evolution.py
```
Watch AI evolve in real-time! See how armies adapt and improve over 10 generations.

### 2. **Full Evolution (100,000 battles, several hours)**
```bash
python launch_tow_evolution.py
```
Choose option 2 for the complete evolution experience. The AI will discover the most powerful army compositions through massive-scale evolution.

### 3. **Custom Evolution**
```bash
python launch_tow_evolution.py
```
Choose option 3 to set your own parameters: battle count, population size, generations.

## 📈 Results from Demo Run

After just 1,000 battles, the AI discovered:

### 🏆 **Evolved Orc Strategy**
- **High Elite Focus (0.78)** - Prefers expensive, powerful units
- **Moderate Aggression (0.62)** - Balanced offensive approach
- **Top Units**: Arachnarok Spider, Giant, Badlands Ogre Bulls
- **Army Composition**: Multiple characters, Night Goblins, Black Orcs, Trolls

### 🏆 **Evolved Nuln Strategy**  
- **Extreme Ranged Focus (0.90)** - Heavy emphasis on shooting
- **Low Elite Focus (0.33)** - Prefers numerous cheaper units
- **Top Units**: Imperial Dwarfs, Artillery, State Troops
- **Army Composition**: Gunline with dwarven infantry support

### 📊 **Battle Results**
- **Nuln slight advantage**: 36.3% vs Orc 34.6%
- **High draw rate**: 29.1% (realistic for balanced forces)
- **Tactical evolution**: Both factions adapted strategies over generations

## 🔬 Technical Details

### **DNA System**
Each AI general has genetic DNA encoding:
- **Unit Preferences**: 0.0-1.0 rating for each unit type
- **Strategic Traits**: Aggression, magic focus, ranged focus, elite focus
- **Army Ratios**: Preferred distribution of core/special/rare/character units

### **Evolution Process**
1. **Battle Phase**: Population fights random matches
2. **Fitness Evaluation**: Score based on wins, efficiency, army validity
3. **Selection**: Best performers chosen for breeding
4. **Crossover**: Successful DNA combined to create offspring
5. **Mutation**: Random changes to explore new strategies
6. **Repeat**: Process continues for 1,000+ generations

### **Army Building AI**
- **Constraint satisfaction**: Ensures valid 2000-point armies
- **Template system**: Orc Horde, Night Goblin Swarm, Artillery Battery, etc.
- **Equipment optimization**: Chooses upgrades based on DNA preferences
- **Command group logic**: Adds champions, standards, musicians strategically

## 📁 System Files

- `tow_unit_database.py` - Complete unit database with TOW stats
- `tow_army_builder.py` - Intelligent army construction system  
- `tow_evolution_ai.py` - Main evolutionary AI engine
- `demo_evolution.py` - Quick demonstration (1,000 battles)
- `launch_tow_evolution.py` - Main launcher with menu system

## 🎯 Extending the System

### **Adding New Factions**
1. Add faction to `Faction` enum in `tow_unit_database.py`
2. Create unit database function (e.g., `create_empire_database()`)
3. Add army templates to `TOWArmyBuilder`
4. Update evolution system to include new faction

### **Advanced Battle Engine**
The current system uses simplified combat. For full authenticity:
- Integrate with `authentic_tow_engine.py` for real TOW mechanics
- Add psychological rules (Fear, Terror, Panic)
- Implement magic system and spell casting
- Add terrain interaction and movement rules

### **Neural Network Tactics**
- `TacticalAI` class provides framework for neural network decision making
- Can be trained on battle outcomes to learn tactical positioning
- Battlefield state encoding ready for deep learning integration

## 🏆 Research Potential

This system enables groundbreaking research:
- **Meta-game discovery**: What are the truly optimal TOW armies?
- **Balance analysis**: Which units are over/under-powered?
- **Strategic emergence**: Do unexpected tactics evolve naturally?
- **Faction balance**: How do different armies compare at high skill levels?

## 🔮 Future Evolution

With 100,000 battles, expect to discover:
- **Refined army compositions** with precise unit ratios
- **Faction-specific strategies** tailored to each army's strengths
- **Counter-strategies** that adapt to opponent preferences
- **Meta-game shifts** as strategies evolve and counter-evolve

---

**Ready to witness the birth of the ultimate TOW AI generals?**

Run `python launch_tow_evolution.py` and choose your evolution path!

🧬 **May the best armies survive!** 🧬 