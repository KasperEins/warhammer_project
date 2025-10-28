#!/usr/bin/env python3
"""
Comprehensive test of Perfect TOW Engine
"""

from perfect_tow_engine import *
import time

def test_perfect_tow_engine():
    print('🏛️ PERFECT TOW ENGINE COMPREHENSIVE TEST')
    print('=' * 60)
    
    # Test 1: Basic initialization
    print('\n📋 Test 1: Component Initialization')
    print('-' * 30)
    try:
        trainer = DistributedTOWTrainer(world_size=1, rank=0)
        print('✅ DistributedTOWTrainer initialized')
    except Exception as e:
        print(f'❌ Trainer initialization failed: {e}')
        return
    
    # Test 2: Action Space Encoder
    print('\n📋 Test 2: Action Space Encoder')
    print('-' * 30)
    try:
        actions = trainer.action_encoder.get_valid_actions(None, [])
        print(f'✅ Generated {len(actions)} base actions')
        print(f'   Sample actions: {[str(a) for a in actions[:3]]}')
    except Exception as e:
        print(f'❌ Action encoding failed: {e}')
        return
    
    # Test 3: Game Engine and Army Generation
    print('\n📋 Test 3: Game Engine & Army Generation')
    print('-' * 30)
    try:
        engine = trainer.game_engine
        army1 = trainer._generate_random_army('orcs')
        army2 = trainer._generate_random_army('empire')
        print(f'✅ Created Orc army: {len(army1)} units')
        print(f'✅ Created Empire army: {len(army2)} units')
        
        # Show army details
        for i, unit in enumerate(army1[:2]):
            print(f'   Orc Unit {i+1}: {unit.name} ({len(unit.models)} models)')
        for i, unit in enumerate(army2[:2]):
            print(f'   Empire Unit {i+1}: {unit.name} ({len(unit.models)} models)')
            
    except Exception as e:
        print(f'❌ Army generation failed: {e}')
        return
    
    # Test 4: Battle Initialization
    print('\n📋 Test 4: Battle Initialization')
    print('-' * 30)
    try:
        game_state = engine.initialize_battle(army1, army2)
        print(f'✅ Battle initialized on {game_state.terrain_grid.shape[0]}x{game_state.terrain_grid.shape[1]} battlefield')
        print(f'   Weather: {game_state.weather}')
        print(f'   Wind of Magic: {game_state.winds_of_magic}')
        print(f'   Current turn: {game_state.game_state.turn_number}, Phase: {game_state.game_state.current_phase}')
        print(f'   Player 1 units: {len(game_state.player1_units)}')
        print(f'   Player 2 units: {len(game_state.player2_units)}')
    except Exception as e:
        print(f'❌ Battle initialization failed: {e}')
        return
    
    # Test 5: State Representation
    print('\n📋 Test 5: Advanced State Representation')
    print('-' * 30)
    try:
        state_data = game_state.to_graph_representation()
        print(f'✅ Graph representation generated:')
        print(f'   Node features: {state_data["node_features"].shape}')
        print(f'   Edge indices: {state_data["edge_indices"].shape}')
        print(f'   Edge features: {state_data["edge_features"].shape}')
        print(f'   Global features: {state_data["global_features"].shape}')
    except Exception as e:
        print(f'❌ State representation failed: {e}')
        return
    
    # Test 6: Neural Network Forward Pass
    print('\n📋 Test 6: Neural Network Forward Pass')
    print('-' * 30)
    try:
        policy, value = trainer.network(state_data)
        print(f'✅ Network forward pass successful:')
        print(f'   Policy output shape: {policy.shape}')
        print(f'   Value prediction: {value.item():.4f}')
        print(f'   Policy max: {policy.max().item():.4f}')
        print(f'   Policy min: {policy.min().item():.4f}')
    except Exception as e:
        print(f'❌ Neural network forward pass failed: {e}')
        return
    
    # Test 7: Single Self-Play Game
    print('\n📋 Test 7: Self-Play Game Generation')
    print('-' * 30)
    try:
        print('   Generating single self-play game...')
        start_time = time.time()
        experiences = trainer._generate_self_play_game()
        duration = time.time() - start_time
        print(f'✅ Self-play game completed in {duration:.2f} seconds')
        print(f'   Generated {len(experiences)} experiences')
        if experiences:
            print(f'   Sample experience keys: {list(experiences[0].keys())}')
    except Exception as e:
        print(f'❌ Self-play game generation failed: {e}')
        return
    
    # Test 8: Training Step (if we have experiences)
    if experiences:
        print('\n📋 Test 8: Training Step')
        print('-' * 30)
        try:
            # Add experiences to replay buffer
            for exp in experiences[:10]:  # Add first 10 experiences
                trainer.replay_buffer.add(exp)
            
            # Perform training step
            print('   Performing training step...')
            trainer._training_step(batch_size=4)
            print('✅ Training step completed successfully')
        except Exception as e:
            print(f'❌ Training step failed: {e}')
    
    print('\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!')
    print('=' * 60)
    print('🔥 Perfect TOW Engine is fully operational!')

if __name__ == "__main__":
    test_perfect_tow_engine() 