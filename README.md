# ⚔️ Warhammer: The Old World - Web Battle Simulator

A real-time web-based battle simulator for Warhammer: The Old World featuring authentic tabletop mechanics, formations, and tactical combat.

## Features

- **Real-time Battle Simulation**: Watch Empire vs Orc armies clash in authentic Warhammer: The Old World combat
- **Authentic Mechanics**: Proper WS/BS/S/T stats, armor saves, formations, and psychology
- **Interactive Web Interface**: Canvas battlefield with unit tooltips and battle log
- **WebSocket Updates**: Live battle progression with phase-by-phase execution
- **Formation System**: Deep, Wide, and Skirmish formations with proper model positioning
- **Artillery System**: Cannons with scatter mechanics and misfire tables
- **Psychology Rules**: Fear, Terror, Panic, and Rally mechanics

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/warhammer_project.git
cd warhammer_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python app.py
```

4. Open your browser to: http://localhost:5001

## How to Play

1. **New Battle**: Click "🎲 New Battle" to deploy armies
2. **Start Battle**: Click "▶️ Start Battle" to begin auto-simulation
3. **Watch Phases**: 
   - Rally Phase: Fleeing units attempt to rally
   - Charge Phase: Cavalry declare charges
   - Movement Phase: Units advance toward enemies
   - Shooting Phase: Ranged units fire
   - Combat Phase: Melee combat resolution

## Battle System

### Armies
- **Empire Forces**: Halberdiers, Handgunners, Great Cannon, Empire Knights
- **Orc Forces**: Orc Boyz, Orc Archers, Wolf Riders

### Combat Mechanics
- Weapon Skill vs Weapon Skill to-hit tables
- Strength vs Toughness wound tables
- Armor saves (with armor-piercing weapons)
- Formation bonuses and rank bonuses
- Psychology tests (Fear, Terror, Panic)

### Terrain
- Darkwood Forest (difficult terrain)
- Valiant Hill (high ground advantage)
- The Great Road (movement bonus)
- Ancient Ruins (cover)

## Technical Details

- **Backend**: Flask with Socket.IO for real-time updates
- **Frontend**: HTML5 Canvas with JavaScript
- **Battle Engine**: Python-based rule implementation
- **Architecture**: Event-driven with WebSocket communication

## Controls

- `🎲 New Battle`: Reset battlefield and deploy armies
- `▶️ Start Battle`: Begin auto-simulation
- `⏸️ Pause`: Pause battle execution
- `⏭️ Next Turn`: Execute one turn manually

## Development

The battle system is modular and extensible:

- `app.py`: Flask web server and WebSocket handling
- `old_world_battle.py`: Core battle mechanics and unit system
- `templates/battle.html`: Frontend interface
- `static/`: CSS and JavaScript assets

## License

This project is for educational and entertainment purposes. Warhammer: The Old World is a trademark of Games Workshop Ltd.

## Contributing

Feel free to submit issues and enhancement requests!
