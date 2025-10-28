# ⚔️ Warhammer: The Old World - Epic Battle System

A stunning web-based implementation of Warhammer: The Old World tabletop battles with real-time animations, authentic game mechanics, and beautiful medieval-themed UI.

## 🎮 Features

### 🏛️ **Authentic Old World Experience**
- **Ranked Infantry Formations** - Deep, wide, and skirmish formations with proper model positioning
- **Cavalry Charges** - Thunderous cavalry charges with proper movement and combat bonuses
- **Artillery Bombardments** - Great cannons and war machines with realistic range and effects
- **Turn-Based Phases** - Movement → Shooting → Charges → Combat phases
- **Authentic Combat Tables** - WS vs WS to-hit charts and S vs T wound mechanics

### 🎨 **Beautiful Web Interface**
- **Canvas-Based Battlefield** - Interactive 1200x800 battlefield with zoom and pan
- **Real-Time Animations** - Projectiles, explosions, movement, and combat effects
- **Medieval UI Theme** - Dark fantasy styling with gold accents and atmospheric backgrounds
- **Responsive Design** - Works on desktop, tablet, and mobile devices
- **Interactive Tooltips** - Hover over units for detailed stats and information

### ⚡ **Advanced Technical Features**
- **Real-Time WebSocket Communication** - Live battle updates between frontend and backend
- **Python Flask Backend** - Robust game logic and battle calculations
- **Canvas Animations** - Smooth 60fps animations for all battle actions
- **Formation Visualization** - See unit ranks, files, and facing directions
- **Battle Chronicle** - Real-time battle log with color-coded events

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd game2
   ```

2. **Install dependencies:**
   ```bash
   pip install Flask Flask-SocketIO
   ```

3. **Start the server:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:5000`

## 🎲 How to Play

1. **Create New Battle** - Click "🎲 New Battle" to deploy armies
2. **Start Battle** - Click "▶️ Start Battle" for automatic turn progression
3. **Manual Control** - Use "⏭️ Next Turn" for step-by-step battles
4. **Interactive Viewing** - Click units to select them, hover for tooltips
5. **Monitor Progress** - Watch the Battle Chronicle for detailed combat results

## 🏰 Army Composition

### 🛡️ **Empire Forces**
- **Empire Halberdiers** - Elite infantry with deep formation (20 models)
- **Handgunners** - Ranged infantry with 24" range (15 models)
- **Great Cannon** - Artillery with 48" range and devastating power
- **Empire Knights** - Heavy cavalry for devastating charges (6 models)

### ⚔️ **Orc Forces**
- **Orc Boyz** - Brutal infantry in deep formation (25 models)
- **Orc Archers** - Ranged support with 18" range (15 models)
- **Wolf Riders** - Fast cavalry for flanking maneuvers (6 models)

## 🗺️ Battlefield Features

### **Terrain Elements**
- **Darkwood Forest** - Blocks line of sight and movement
- **Valiant Hill** - Provides elevation advantage
- **The Great Road** - Allows faster movement
- **Ancient Ruins** - Provides cover and strategic positions

### **Grid System**
- 72" x 48" battlefield (6' x 4' table)
- 6" grid squares for measurement
- Authentic Old World scale and distances

## 🔧 Technical Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   Frontend      │ <────────────> │   Backend       │
│                 │                 │                 │
│ • HTML5 Canvas  │                 │ • Flask Server  │
│ • JavaScript    │                 │ • Socket.IO     │
│ • CSS3 Styling  │                 │ • Game Logic    │
│ • Animations    │                 │ • Battle Rules  │
└─────────────────┘                 └─────────────────┘
```

### **Frontend (JavaScript/HTML/CSS)**
- Canvas-based battlefield rendering
- Real-time animation system
- Interactive unit selection and tooltips
- Responsive medieval-themed UI

### **Backend (Python/Flask)**
- Authentic Warhammer battle mechanics
- Turn-based game state management
- WebSocket communication for real-time updates
- Formation and movement calculations

## 🎯 Game Mechanics

### **Combat Resolution**
- Weapon Skill vs Weapon Skill to-hit tables
- Strength vs Toughness wound charts
- Armor saves and model removal
- Formation fighting with rank bonuses

### **Movement System**
- Formation movement with facing
- Cavalry charge mechanics
- Terrain interaction and modifiers
- Line of sight calculations

### **Phase Structure**
1. **Movement Phase** - Units reposition and advance
2. **Shooting Phase** - Ranged combat and artillery
3. **Charge Phase** - Cavalry and monster charges
4. **Combat Phase** - Melee combat resolution

## 🎨 Visual Features

- **Unit Formations** - Visual representation of ranks and files
- **Health Bars** - Color-coded unit strength indicators
- **Projectile Animations** - Arrows and cannonballs in flight
- **Explosion Effects** - Impact animations for hits
- **Combat Sparkles** - Visual feedback for melee engagement
- **Terrain Rendering** - Beautiful battlefield with obstacles

## 🔮 Future Enhancements

- [ ] Magic system with spell casting
- [ ] Character heroes and monsters
- [ ] Psychology rules (Fear, Terror, Frenzy)
- [ ] More army factions (Dwarfs, Elves, Chaos)
- [ ] Campaign system with multiple battles
- [ ] Army builder with point costs
- [ ] Multiplayer support for human opponents
- [ ] Tournament mode with victory conditions

## 📝 Technical Details

### **Dependencies**
- `Flask 3.0+` - Web framework
- `Flask-SocketIO 5.5+` - Real-time communication
- `python-socketio` - WebSocket support
- `python-engineio` - Engine.IO backend

### **Browser Compatibility**
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### **Performance**
- 60fps canvas animations
- Real-time WebSocket updates
- Optimized rendering pipeline
- Responsive design for all screen sizes

## 🏆 Credits

Created for the ultimate Warhammer: The Old World experience, combining authentic tabletop mechanics with modern web technology for epic digital battles!

**"In the grim darkness of the Old World, there is only war... and beautiful web interfaces!"** ⚔️ 