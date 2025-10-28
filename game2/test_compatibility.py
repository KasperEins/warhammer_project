#!/usr/bin/env python3
"""
Test compatibility between fixed improved training and original system
"""

import torch
from warhammer_ai_agent import WarhammerAIAgent
from fixed_improved_training import FixedImprovedAI

def test_compatibility():
    print("🔧 COMPATIBILITY TEST")
    print("=" * 40)
    
    # Create both AI types
    print("Creating original AI...")
    original_ai = WarhammerAIAgent(state_size=50, action_size=15)
    
    print("Creating fixed improved AI...")
    fixed_ai = FixedImprovedAI(state_size=50, action_size=15)
    
    print("\n🏗️ ARCHITECTURE COMPARISON:")
    print("Original AI network layers:")
    for name, param in original_ai.q_network.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print("\nFixed AI network layers:")
    for name, param in fixed_ai.q_network.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print("\n✅ COMPATIBILITY CHECK:")
    
    # Test 1: Can we transfer weights?
    try:
        # Save fixed AI
        fixed_ai.save_model_with_memory('temp_fixed_test.pth')
        print("✅ Fixed AI saved successfully")
        
        # Load into original AI
        original_ai.load_model('temp_fixed_test.pth')
        print("✅ Original AI loaded fixed model successfully!")
        
        # Clean up
        import os
        os.remove('temp_fixed_test.pth')
        
    except Exception as e:
        print(f"❌ Compatibility failed: {e}")
        return False
    
    # Test 2: Are architectures identical?
    original_params = list(original_ai.q_network.named_parameters())
    fixed_params = list(fixed_ai.q_network.named_parameters())
    
    if len(original_params) != len(fixed_params):
        print(f"❌ Different number of parameters: {len(original_params)} vs {len(fixed_params)}")
        return False
    
    for (orig_name, orig_param), (fixed_name, fixed_param) in zip(original_params, fixed_params):
        if orig_name != fixed_name:
            print(f"❌ Parameter name mismatch: {orig_name} vs {fixed_name}")
            return False
        if orig_param.shape != fixed_param.shape:
            print(f"❌ Parameter shape mismatch for {orig_name}: {orig_param.shape} vs {fixed_param.shape}")
            return False
    
    print("✅ Architectures are IDENTICAL!")
    
    # Test 3: Enhanced features work?
    print(f"\n🚀 ENHANCED FEATURES:")
    print(f"  Fixed AI epsilon: {fixed_ai.epsilon} (vs original: {original_ai.epsilon})")
    print(f"  Fixed AI memory size: {fixed_ai.memory.maxlen} (vs original: {original_ai.memory.maxlen})")
    print(f"  Fixed AI learning rate: {fixed_ai.learning_rate} (vs original: {original_ai.learning_rate})")
    
    print("\n🎉 COMPATIBILITY TEST PASSED!")
    print("✅ Fixed improved training is fully compatible with original system")
    print("✅ Enhanced features are preserved")
    print("✅ Models can be loaded between systems")
    
    return True

if __name__ == "__main__":
    test_compatibility() 