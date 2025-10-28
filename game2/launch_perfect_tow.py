#!/usr/bin/env python3
"""
🏛️ PERFECT TOW SYSTEM LAUNCHER
===============================

Easy launcher for the Perfect Warhammer: The Old World AI system.
Provides different modes and configurations for various use cases.

Usage:
    python launch_perfect_tow.py --mode demo
    python launch_perfect_tow.py --mode training --gpus 4
    python launch_perfect_tow.py --mode meta-learning --generations 1000
"""

import argparse
import sys
import os
import logging
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perfect_tow_engine import (
    launch_perfect_tow_training, MetaLearningTOW, 
    DistributedTOWTrainer, PerfectTOWGameEngine
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                🏛️ PERFECT WARHAMMER: THE OLD WORLD        ║
    ║                     AI SYSTEM LAUNCHER                   ║
    ║                                                          ║
    ║  The Ultimate Tabletop Wargaming AI                      ║
    ║  ✅ Complete TOW Rules Integration                        ║
    ║  ✅ 10,000+ Action Space Encoding                        ║
    ║  ✅ Advanced Graph Neural Networks                       ║
    ║  ✅ Distributed GPU Training                             ║
    ║  ✅ Multi-Faction Co-Evolution                           ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
        else:
            print("⚠️  CUDA not available - will use CPU")
            
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    # Check other dependencies
    required_packages = ['numpy', 'matplotlib']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not found")
            return False
    
    print("✅ All requirements satisfied\n")
    return True

def run_demo_mode():
    """Run demonstration mode"""
    print("🎯 LAUNCHING DEMONSTRATION MODE")
    print("=" * 40)
    
    try:
        # Import demo script
        from demo_perfect_tow import main as demo_main
        demo_main()
        
    except ImportError:
        print("⚠️  Demo script not found, running basic demo...")
        
        # Basic demo
        print("🔧 Initializing Perfect TOW Engine...")
        game_engine = PerfectTOWGameEngine()
        
        print("⚔️ Creating armies...")
        from tow_comprehensive_rules import create_orc_army, create_nuln_army
        orc_army = create_orc_army()
        empire_army = create_nuln_army()
        
        print(f"✅ Created armies: {len(orc_army)} Orcs vs {len(empire_army)} Empire")
        
        print("🎮 Initializing battle...")
        game_state = game_engine.initialize_battle(orc_army, empire_army)
        
        print("🧠 Testing neural network...")
        from perfect_tow_engine import AdvancedTOWNetwork
        network = AdvancedTOWNetwork()
        
        graph_data = game_state.to_graph_representation()
        with torch.no_grad():
            policy, value = network(graph_data)
        
        print(f"✅ Network output - Policy: {policy.shape}, Value: {value.item():.3f}")
        print("🎉 Basic demo complete!")

def run_training_mode(gpus: int, generations: int, batch_size: int):
    """Run training mode"""
    print(f"🚀 LAUNCHING TRAINING MODE")
    print(f"   GPUs: {gpus}")
    print(f"   Generations: {generations}")
    print(f"   Batch Size: {batch_size}")
    print("=" * 40)
    
    if gpus > 1 and torch.cuda.is_available():
        # Distributed training
        print("🌐 Starting distributed training...")
        launch_perfect_tow_training(
            world_size=gpus,
            generations=generations
        )
    else:
        # Single GPU/CPU training
        print("🔧 Starting single-process training...")
        trainer = DistributedTOWTrainer(world_size=1, rank=0)
        
        for generation in range(generations):
            trainer.train_epoch(num_games=100, batch_size=batch_size)
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}/{generations} complete")
        
        print("✅ Training complete!")

def run_meta_learning_mode(generations: int, factions: list):
    """Run meta-learning mode"""
    print(f"🧬 LAUNCHING META-LEARNING MODE")
    print(f"   Generations: {generations}")
    print(f"   Factions: {', '.join(factions)}")
    print("=" * 40)
    
    # Initialize meta-learning system
    meta_learner = MetaLearningTOW(factions=factions)
    
    print("🏁 Starting co-evolution...")
    final_ratings = meta_learner.co_evolve(
        generations=generations,
        games_per_matchup=50
    )
    
    print("\n🏆 FINAL FACTION RANKINGS:")
    print("=" * 30)
    sorted_factions = sorted(
        final_ratings.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (faction, rating) in enumerate(sorted_factions):
        print(f"{i+1}. {faction.capitalize()}: {rating:.1f}")

def run_benchmark_mode():
    """Run benchmark mode"""
    print("⚡ LAUNCHING BENCHMARK MODE")
    print("=" * 40)
    
    import time
    
    # Benchmark game engine
    print("🎯 Benchmarking Game Engine...")
    game_engine = PerfectTOWGameEngine()
    
    from tow_comprehensive_rules import create_orc_army, create_nuln_army
    
    start_time = time.time()
    for i in range(10):
        orc_army = create_orc_army()
        empire_army = create_nuln_army()
        game_state = game_engine.initialize_battle(orc_army, empire_army)
    
    game_init_time = (time.time() - start_time) / 10
    print(f"   Game initialization: {game_init_time*1000:.2f}ms per battle")
    
    # Benchmark neural network
    print("🧠 Benchmarking Neural Network...")
    from perfect_tow_engine import AdvancedTOWNetwork
    network = AdvancedTOWNetwork()
    
    if torch.cuda.is_available():
        network = network.cuda()
        device = "GPU"
    else:
        device = "CPU"
    
    # Warmup
    dummy_input = {
        'node_features': torch.randn(20, 64),
        'edge_indices': torch.randint(0, 20, (2, 50)),
        'edge_features': torch.randn(50, 16),
        'global_features': torch.randn(32)
    }
    
    if torch.cuda.is_available():
        dummy_input = {k: v.cuda() if torch.is_tensor(v) else v 
                      for k, v in dummy_input.items()}
    
    # Warmup runs
    for _ in range(5):
        with torch.no_grad():
            _, _ = network(dummy_input)
    
    # Benchmark runs
    start_time = time.time()
    for i in range(100):
        with torch.no_grad():
            policy, value = network(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_time = (time.time() - start_time) / 100
    print(f"   Neural network inference ({device}): {inference_time*1000:.2f}ms per forward pass")
    
    # Benchmark action encoding
    print("🎯 Benchmarking Action Encoding...")
    from perfect_tow_engine import ActionSpaceEncoder
    action_encoder = ActionSpaceEncoder()
    
    start_time = time.time()
    for i in range(100):
        valid_actions = action_encoder.get_valid_actions(
            game_state.game_state,
            game_state.player1_units + game_state.player2_units
        )
    
    action_time = (time.time() - start_time) / 100
    print(f"   Action space generation: {action_time*1000:.2f}ms per call")
    print(f"   Average actions per state: {len(valid_actions)}")
    
    print("\n📊 BENCHMARK SUMMARY:")
    print(f"   Game Engine: {game_init_time*1000:.1f}ms")
    print(f"   Neural Network: {inference_time*1000:.1f}ms")
    print(f"   Action Encoding: {action_time*1000:.1f}ms")
    print(f"   Total per move: {(game_init_time + inference_time + action_time)*1000:.1f}ms")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Perfect Warhammer: The Old World AI System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_perfect_tow.py --mode demo
  python launch_perfect_tow.py --mode training --gpus 4 --generations 1000
  python launch_perfect_tow.py --mode meta-learning --factions orcs empire dwarfs
  python launch_perfect_tow.py --mode benchmark
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['demo', 'training', 'meta-learning', 'benchmark'],
        default='demo',
        help='Operation mode (default: demo)'
    )
    
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs for distributed training (default: 1)'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=100,
        help='Number of training generations (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    
    parser.add_argument(
        '--factions',
        nargs='+',
        default=['orcs', 'empire', 'dwarfs', 'elves'],
        help='Factions for meta-learning (default: orcs empire dwarfs elves)'
    )
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Disable CUDA if requested
    if args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Print banner
    print_banner()
    
    # Check requirements
    if not check_system_requirements():
        print("❌ System requirements not met")
        sys.exit(1)
    
    # Run selected mode
    try:
        if args.mode == 'demo':
            run_demo_mode()
            
        elif args.mode == 'training':
            run_training_mode(args.gpus, args.generations, args.batch_size)
            
        elif args.mode == 'meta-learning':
            run_meta_learning_mode(args.generations, args.factions)
            
        elif args.mode == 'benchmark':
            run_benchmark_mode()
        
        print(f"\n🎉 {args.mode.upper()} MODE COMPLETED SUCCESSFULLY!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 