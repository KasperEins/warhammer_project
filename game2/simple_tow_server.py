#!/usr/bin/env python3
"""
Simplified Warhammer: The Old World Web Server
Focuses on core functionality with robust error handling
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import json
import time
from threading import Thread, Lock
from tow_web_battle import TOWBattle

# Flask app initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'warhammer_old_world_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global battle instance
battle = TOWBattle()

@app.route('/')
def index():
    return render_template('tow_battle.html')

@app.route('/debug')
def debug():
    with open('debug_frontend.html', 'r') as f:
        return f.read()

@app.route('/animated')
def animated():
    return render_template('animated_battle.html')

@app.route('/simple')
def simple():
    return render_template('simple_animated.html')

@app.route('/realistic')
def realistic():
    return render_template('realistic_battle.html')

@app.route('/api/battle_state')
def get_battle_state():
    """Get battle state with robust error handling"""
    try:
        state = battle.get_battle_state()
        # Double-check JSON serialization
        json.dumps(state)  # Test serialization
        return jsonify(state)
    except Exception as e:
        print(f"❌ Battle state error: {e}")
        # Return minimal safe state
        return jsonify({
            'units': [],
            'turn': getattr(battle, 'turn', 1),
            'phase': getattr(battle, 'phase', 'deployment'),
            'active_player': getattr(battle, 'active_player', 'nuln'),
            'battle_log': [f"Error: {str(e)}"],
            'battle_state': 'error'
        })

@app.route('/api/start_battle', methods=['POST'])
def start_battle():
    """Start battle with error handling"""
    try:
        success = battle.start_battle()
        return jsonify({'success': success})
    except Exception as e:
        print(f"❌ Start battle error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/reset_battle', methods=['POST'])
def reset_battle():
    """Reset battle"""
    try:
        battle.reset_battle()
        return jsonify({'success': True})
    except Exception as e:
        print(f"❌ Reset battle error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/battle_events')
def get_battle_events():
    """Get recent battle events for animations"""
    try:
        # Get last 10 log entries for animation triggers
        logs = getattr(battle, 'battle_log', [])
        recent_logs = logs[-10:] if logs else []
        
        events = []
        for log in recent_logs:
            if '💥' in log or '⚡' in log:
                events.append({'type': 'magic', 'message': log})
            elif '💀' in log:
                events.append({'type': 'death', 'message': log})
            elif '⚔️' in log:
                events.append({'type': 'combat', 'message': log})
            elif '🏹' in log:
                events.append({'type': 'shooting', 'message': log})
                
        return jsonify({'events': events})
    except Exception as e:
        print(f"❌ Battle events error: {e}")
        return jsonify({'events': []})

@socketio.on('connect')
def handle_connect():
    print("🌐 Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("🌐 Client disconnected")

if __name__ == '__main__':
    print("⚔️ WARHAMMER: THE OLD WORLD - SIMPLIFIED WEB BATTLE")
    print("==================================================")
    print("🌐 Starting simplified web server...")
    print("📱 Open your browser to http://localhost:5001")
    print("Features:")
    print("• Robust error handling")
    print("• Enhanced console logging")
    print("• Fixed AI battle mechanics")
    print("• Authentic TOW rules")
    
    socketio.run(app, host='0.0.0.0', port=5001, debug=True) 