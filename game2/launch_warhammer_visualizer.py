#!/usr/bin/env python3
"""
🚀 WARHAMMER AI VISUALIZER LAUNCHER
===================================

One-click launcher for the complete Warhammer AI battle visualization system.
Automatically handles AI model setup and starts the web interface.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_ai_models():
    """Check if AI models exist"""
    empire_model = Path("empire_ai_300k.pth")
    orc_model = Path("orc_ai_300k.pth")
    
    empire_exists = empire_model.exists()
    orc_exists = orc_model.exists()
    
    print("🔍 Checking AI models...")
    print(f"   Empire AI (300k): {'✅ Found' if empire_exists else '❌ Missing'}")
    print(f"   Orc AI (300k): {'✅ Found' if orc_exists else '❌ Missing'}")
    
    return empire_exists and orc_exists

def train_quick_models():
    """Train quick AI models if 300k models aren't available"""
    print("\n🚀 300k models not found. Training quick AI models...")
    print("⏱️ This will take about 2-3 minutes...")
    
    try:
        result = subprocess.run([sys.executable, "quick_ai_trainer.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Quick training completed successfully!")
            
            # Link the quick models to the expected names
            if Path("quick_empire_ai.pth").exists():
                os.system("ln -sf quick_empire_ai.pth empire_ai_300k.pth")
            if Path("quick_orc_ai.pth").exists():
                os.system("ln -sf quick_orc_ai.pth orc_ai_300k.pth")
            
            return True
        else:
            print(f"❌ Training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out. Using fallback models...")
        return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def start_web_visualizer():
    """Start the web visualizer"""
    print("\n🌐 Starting Warhammer AI Battle Visualizer...")
    print("📱 Open http://localhost:5001 in your browser")
    print("⚔️ Click 'Start Battle' to watch 300k-trained AIs fight!")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "warhammer_web_visualizer.py"])
    except KeyboardInterrupt:
        print("\n👋 Visualizer stopped. Thanks for watching the AI battles!")

def main():
    """Main launcher function"""
    print("🏛️ WARHAMMER AI BATTLE VISUALIZER LAUNCHER")
    print("=" * 50)
    print("🎯 Preparing epic AI battles...")
    
    # Check if we're in the right directory
    if not Path("warhammer_battle_core.py").exists():
        print("❌ Error: Please run this from the game2 directory")
        print("   Expected files: warhammer_battle_core.py, warhammer_web_visualizer.py")
        return
    
    # Check for AI models
    if check_ai_models():
        print("✅ AI models ready!")
    else:
        print("\n⚠️ AI models not found. Options:")
        print("   1. Train quick models (2-3 minutes)")
        print("   2. Use untrained models (random behavior)")
        
        choice = input("\nChoose option (1/2): ").strip()
        
        if choice == "1":
            if not train_quick_models():
                print("⚠️ Training failed. Continuing with untrained models...")
        else:
            print("📝 Using untrained models (AIs will behave randomly)")
    
    # Start the visualizer
    print("\n🎮 All systems ready!")
    time.sleep(1)
    start_web_visualizer()

if __name__ == "__main__":
    main() 