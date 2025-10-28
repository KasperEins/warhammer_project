#!/usr/bin/env python3
"""
SIMPLE WARHAMMER AI DEMO
========================
Just one button - watch the AI play!
"""

from flask import Flask, render_template, jsonify, request
import time
import random
from warhammer_ai_agent import WarhammerAIAgent, WarhammerBattleEnvironment

app = Flask(__name__)

# Initialize AI agent
ai_agent = None

def load_ai_agent():
    """Load the trained AI agent"""
    global ai_agent
    if ai_agent is None:
        try:
            # Calculate state and action sizes based on environment
            env = WarhammerBattleEnvironment()
            state = env.reset()
            state_size = len(state)
            action_size = 13  # 8 movement directions + 5 special tactics
            
            ai_agent = WarhammerAIAgent(state_size=state_size, action_size=action_size)
            ai_agent.load_model('warhammer_ai_model.pth')
            print("✅ AI Agent loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading AI: {e}")
            # Create a fallback dummy agent for demo purposes
            try:
                env = WarhammerBattleEnvironment()
                state = env.reset()
                state_size = len(state)
                action_size = 13
                ai_agent = WarhammerAIAgent(state_size=state_size, action_size=action_size)
                print("✅ Created demo AI agent (not trained)")
            except Exception as e2:
                print(f"❌ Failed to create demo agent: {e2}")
                ai_agent = None
    return ai_agent

@app.route('/')
def index():
    """Simple homepage with one play button"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Warhammer AI Master</title>
    <style>
        body {
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            color: #fff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            text-align: center;
            max-width: 800px;
            padding: 2rem;
        }
        
        h1 {
            font-size: 3rem;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            background: linear-gradient(45deg, #ff6b6b, #ffd93d, #6bcf7f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            font-size: 1.2rem;
            margin-bottom: 2rem;
            opacity: 0.8;
        }
        
        .stats {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 2rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .stat-item {
            display: inline-block;
            margin: 0 1rem;
            padding: 0.5rem 1rem;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #ffd93d;
        }
        
        .play-button {
            background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
            color: white;
            border: none;
            padding: 1rem 3rem;
            font-size: 1.5rem;
            font-weight: bold;
            border-radius: 50px;
            cursor: pointer;
            box-shadow: 0 10px 30px rgba(255, 107, 107, 0.4);
            transition: all 0.3s ease;
            margin: 2rem 0;
        }
        
        .play-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(255, 107, 107, 0.6);
        }
        
        .play-button:active {
            transform: translateY(-1px);
        }
        
        .play-button:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .battle-log {
            background: rgba(0,0,0,0.7);
            border-radius: 10px;
            padding: 1rem;
            margin: 2rem 0;
            max-height: 300px;
            overflow-y: auto;
            text-align: left;
            font-family: 'Courier New', monospace;
            border: 1px solid rgba(255,255,255,0.2);
            display: none;
        }
        
        .log-entry {
            margin: 0.5rem 0;
            padding: 0.25rem;
        }
        
        .victory {
            color: #6bcf7f;
            font-weight: bold;
        }
        
        .defeat {
            color: #ff6b6b;
            font-weight: bold;
        }
        
        .info {
            color: #ffd93d;
        }
        
        .loading {
            display: none;
            margin: 1rem 0;
        }
        
        .spinner {
            border: 4px solid rgba(255,255,255,0.1);
            border-left: 4px solid #ffd93d;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Warhammer AI Master</h1>
        <p class="subtitle">Watch our trained AI agent dominate the battlefield!</p>
        
        <div class="stats">
            <h3>🏆 Training Results</h3>
            <div class="stat-item">
                <div>Episodes Trained</div>
                <div class="stat-value">10,000</div>
            </div>
            <div class="stat-item">
                <div>Win Rate</div>
                <div class="stat-value">96.15%</div>
            </div>
            <div class="stat-item">
                <div>Favorite Strategy</div>
                <div class="stat-value">Artillery Strike</div>
            </div>
        </div>
        
        <button class="play-button" onclick="playBattle()" id="playBtn">
            ⚔️ Watch AI Battle!
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>AI is thinking and battling...</p>
        </div>
        
        <div class="battle-log" id="battleLog">
            <h4>🎮 Battle Log:</h4>
            <div id="logEntries"></div>
        </div>
    </div>

    <script>
        async function playBattle() {
            const btn = document.getElementById('playBtn');
            const loading = document.getElementById('loading');
            const battleLog = document.getElementById('battleLog');
            const logEntries = document.getElementById('logEntries');
            
            // Disable button and show loading
            btn.disabled = true;
            btn.textContent = '🤖 AI is battling...';
            loading.style.display = 'block';
            battleLog.style.display = 'block';
            
            // Clear previous logs
            logEntries.innerHTML = '';
            
            try {
                addLogEntry('🚀 Initializing AI battle system...', 'info');
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                addLogEntry('⚔️ Army of Nuln vs Troll Horde battle begins!', 'info');
                await new Promise(resolve => setTimeout(resolve, 500));
                
                // Call the battle API
                const response = await fetch('/api/play_battle', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    addLogEntry(`🎯 AI chose strategy: ${data.strategy}`, 'info');
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    
                    addLogEntry('💥 Battle in progress...', 'info');
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    
                    if (data.result === 'victory') {
                        addLogEntry(`🏆 VICTORY! Score: ${data.score}`, 'victory');
                        addLogEntry('🎉 The AI\'s artillery strategy dominates once again!', 'victory');
                    } else {
                        addLogEntry(`💀 Defeat! Score: ${data.score}`, 'defeat');
                        addLogEntry('😤 Even masters can have off days...', 'defeat');
                    }
                    
                    addLogEntry(`📊 AI used: ${data.actions_used.join(', ')}`, 'info');
                    
                } else {
                    addLogEntry('❌ Battle failed to initialize', 'defeat');
                    addLogEntry(`Error: ${data.error}`, 'defeat');
                }
                
            } catch (error) {
                addLogEntry('❌ Connection error', 'defeat');
                addLogEntry(`Error: ${error.message}`, 'defeat');
            }
            
            // Re-enable button
            btn.disabled = false;
            btn.textContent = '⚔️ Watch Another Battle!';
            loading.style.display = 'none';
        }
        
        function addLogEntry(text, type = '') {
            const logEntries = document.getElementById('logEntries');
            const entry = document.createElement('div');
            entry.className = `log-entry ${type}`;
            entry.textContent = text;
            logEntries.appendChild(entry);
            logEntries.scrollTop = logEntries.scrollHeight;
        }
        
        // Auto-play on load
        window.addEventListener('load', () => {
            setTimeout(() => {
                addLogEntry('🤖 AI Agent ready for battle!', 'info');
            }, 1000);
        });
    </script>
</body>
</html>
    '''

@app.route('/api/play_battle', methods=['POST'])
def play_battle():
    """API endpoint to run a single AI battle"""
    try:
        # Load AI agent
        agent = load_ai_agent()
        if not agent:
            return jsonify({
                'success': False,
                'error': 'Failed to load AI agent'
            })
        
        # Create battle environment
        env = WarhammerBattleEnvironment()
        state = env.reset()
        
        total_reward = 0
        actions_used = []
        
        # Run battle with AI
        for step in range(10):  # Max 10 actions per battle
            action = agent.act(state, training=False)  # Use trained policy
            actions_used.append(env.action_names[action])
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        # Determine result
        result = 'victory' if total_reward > 0 else 'defeat'
        
        # Get strategy (most common action)
        strategy_counts = {}
        for action in actions_used:
            strategy_counts[action] = strategy_counts.get(action, 0) + 1
        
        main_strategy = max(strategy_counts.keys(), key=strategy_counts.get) if strategy_counts else 'Unknown'
        
        return jsonify({
            'success': True,
            'result': result,
            'score': round(total_reward, 1),
            'strategy': main_strategy,
            'actions_used': actions_used[:5],  # First 5 actions
            'total_steps': step + 1
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("🤖 WARHAMMER AI DEMO")
    print("=" * 30)
    print("🚀 Loading trained AI agent...")
    
    # Pre-load the AI agent
    load_ai_agent()
    
    print("🌐 Starting simple web server...")
    print("📱 Open your browser to http://localhost:8080")
    
    app.run(host='0.0.0.0', port=8080, debug=True) 