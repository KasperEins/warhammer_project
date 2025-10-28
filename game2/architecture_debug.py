#!/usr/bin/env python3
"""
Debug script to show the architecture mismatch problem
"""
import torch
import torch.nn as nn
from warhammer_ai_agent import WarhammerAIAgent, DQNNetwork
from mass_training_system import TrainingAI

def compare_architectures():
    print("🔍 ARCHITECTURE COMPARISON DEBUG")
    print("=" * 50)
    
    # Original architecture (WarhammerAIAgent)
    print("🏛️ ORIGINAL ARCHITECTURE (WarhammerAIAgent):")
    original_ai = WarhammerAIAgent(state_size=50, action_size=15)
    print("Network layers:")
    for name, param in original_ai.q_network.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print("\n🔧 TRAINING ARCHITECTURE (TrainingAI):")
    training_ai = TrainingAI(state_size=50, action_size=15)
    print("Network layers:")
    for name, param in training_ai.q_network.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print("\n🚨 COMPATIBILITY TEST:")
    
    # Try to load improved model into original architecture
    try:
        checkpoint = torch.load('orc_ai_improved_50000.pth', map_location='cpu')
        print("✅ Checkpoint loaded successfully")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'q_network_state' in checkpoint:
            print(f"Network state keys: {list(checkpoint['q_network_state'].keys())}")
            
            # Try to load into original
            try:
                original_ai.q_network.load_state_dict(checkpoint['q_network_state'])
                print("✅ Successfully loaded into original architecture")
            except Exception as e:
                print(f"❌ Failed to load into original: {e}")
                
        else:
            print("❌ No 'q_network_state' key found in checkpoint")
            
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
    
    print("\n💡 THE ISSUE:")
    print("The 'improved' training uses nn.Sequential with numbered layers (0, 3, 5)")
    print("The original system uses named layers (fc1, fc2, fc3)")
    print("This makes them completely incompatible! 😅")
    
    print("\n🔧 ARCHITECTURE DETAILS:")
    print("Original DQNNetwork:")
    print("  fc1: Linear(50 -> 256)")
    print("  fc2: Linear(256 -> 256)")  
    print("  fc3: Linear(256 -> 15)")
    
    print("\nTraining AI Sequential:")
    print("  0: Linear(50 -> 256)")
    print("  3: Linear(256 -> 128)")  # Different size!
    print("  5: Linear(128 -> 15)")

if __name__ == "__main__":
    compare_architectures() 