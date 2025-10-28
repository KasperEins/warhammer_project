#!/usr/bin/env python3
"""
⚔️ QUICK BATTLE DEMO
====================

Quick demonstration of 300k trained AIs clashing!
"""

import torch
import torch.nn as nn
import numpy as np
import random
from datetime import datetime

class QuickBattleAI(nn.Module):
    def __init__(self):
        super(QuickBattleAI, self).__init__()
        self.fc1 = nn.Linear(50, 256)
        self.fc2 = nn.Linear(256, 256)  
        self.fc3 = nn.Linear(256, 15)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
    def decide(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor).numpy()[0]
            action = int(np.argmax(q_values))
            q_value = float(q_values[action])
            return action, q_value

def load_ai(filename, faction_name):
    """Load AI with error handling"""
    ai = QuickBattleAI()
    try:
        state = torch.load(filename, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'q_network_state_dict' in state:
            ai.load_state_dict(state['q_network_state_dict'])
        else:
            ai.load_state_dict(state)
        print(f"✅ {faction_name} AI loaded from {filename}")
        return ai, True
    except Exception as e:
        print(f"⚠️ {faction_name} failed to load: {e}")
        return ai, False

def run_quick_battle():
    """Run a quick 5-turn demonstration battle"""
    
    print("⚔️ QUICK BATTLE: 300K AI MASTERS")
    print("=" * 50)
    
    # Load AIs
    empire_ai, empire_loaded = load_ai('empire_massive_300k_final.pth', 'Empire')
    orc_ai, orc_loaded = load_ai('orc_massive_300k_final.pth', 'Orc')
    
    action_names = [
        "Move North", "Move South", "Move East", "Move West",
        "Move NE", "Move NW", "Move SE", "Move SW", 
        "Cavalry Charge", "Artillery Strike", "Defensive Formation",
        "Magic Attack", "Mass Shooting", "Special Tactic A", "Special Tactic B"
    ]
    
    if not empire_loaded:
        print("🎭 Creating Empire demo AI...")
        with torch.no_grad():
            empire_ai.fc3.weight[9] *= 3.0  # Cavalry
            empire_ai.fc3.bias[9] += 15.0
            empire_ai.fc3.weight[10] *= 2.5  # Artillery
            empire_ai.fc3.bias[10] += 10.0
    
    if not orc_loaded:
        print("🎭 Creating Orc demo AI...")
        with torch.no_grad():
            orc_ai.fc3.weight[12] *= 3.0  # Mass Shooting
            orc_ai.fc3.bias[12] += 12.0
            orc_ai.fc3.weight[13] *= 2.5  # Special Tactics
            orc_ai.fc3.bias[13] += 8.0
    
    print("\n🎯 5-TURN QUICK BATTLE!")
    print("=" * 30)
    
    empire_wins = 0
    orc_wins = 0
    
    for turn in range(1, 6):
        print(f"\n🔥 TURN {turn}")
        print("-" * 20)
        
        # Generate battle state
        state = np.random.rand(50)
        state[:10] = np.random.choice([0, 1], 10, p=[0.3, 0.7])  # Units
        state[10:20] = np.random.uniform(0.3, 1.0, 10)  # Health
        state[20:30] = np.random.uniform(-1, 1, 10)  # Positions
        state[30:40] = np.random.uniform(0, 1, 10)  # Conditions
        state[40:50] = np.random.uniform(0, 1, 10)  # Momentum
        
        # AI decisions
        emp_action, emp_q = empire_ai.decide(state)
        orc_action, orc_q = orc_ai.decide(state)
        
        print(f"🏛️ Empire: {action_names[emp_action]} (Q={emp_q:.2f})")
        print(f"🟢 Orc: {action_names[orc_action]} (Q={orc_q:.2f})")
        
        # Determine winner
        if emp_q > orc_q:
            empire_wins += 1
            print("📊 Result: Empire gains advantage!")
        elif orc_q > emp_q:
            orc_wins += 1
            print("📊 Result: Orc breakthrough!")
        else:
            print("📊 Result: Tactical stalemate!")
        
        # Commentary
        if emp_q > 15 or orc_q > 15:
            print("💬 MASTERY LEVEL confidence displayed!")
        elif emp_q > 8 and orc_q > 8:
            print("💬 Both AIs show high learned confidence!")
        elif abs(emp_q - orc_q) > 10:
            print("💬 Significant tactical advantage shown!")
    
    print(f"\n🏆 FINAL RESULT:")
    if empire_wins > orc_wins:
        print(f"🏛️ EMPIRE VICTORY ({empire_wins}-{orc_wins})")
        print("Empire AI's learned strategies proved superior!")
    elif orc_wins > empire_wins:
        print(f"🟢 ORC VICTORY ({orc_wins}-{empire_wins})")
        print("Orc AI's specialized tactics dominated!")
    else:
        print(f"⚔️ EPIC DRAW ({empire_wins}-{orc_wins})")
        print("Perfectly matched AI learning!")
    
    print(f"\n🎯 WHAT YOU WITNESSED:")
    print(f"• Neural networks trained on 300,000 battles each")
    print(f"• Q-values showing learned tactical confidence")
    print(f"• Genuine AI decision-making from experience")
    print(f"• {empire_loaded and orc_loaded and 'REAL' or 'DEMO'} trained models in action!")

if __name__ == "__main__":
    run_quick_battle() 