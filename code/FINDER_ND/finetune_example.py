#!/usr/bin/env python3
"""
Example script demonstrating FINDER_ND fine-tuning
This script shows how to run fine-tuning with different configurations
"""

import os
import sys
from finetune_config import get_config, list_configs
from finetune import FineTuneGraphDQN

def run_single_finetune(config_name, max_epochs=10):
    """Run fine-tuning with a specific configuration"""
    print(f"\n{'='*60}")
    print(f"Running fine-tuning with configuration: {config_name}")
    print(f"{'='*60}")
    
    try:
        # Get configuration
        config = get_config(config_name)
        
        # Override epochs for quick demonstration
        config['num_epochs'] = max_epochs
        
        # Initialize fine-tuner
        finetuner = FineTuneGraphDQN(**config)
        
        # Train model
        finetuner.train()
        
        # Evaluate
        real_graph = finetuner.load_real_graph()
        test_graphs = finetuner.create_graph_batches(real_graph, num_batches=5)
        loss = finetuner.evaluate(test_graphs)
        
        print(f"\n✅ {config_name} completed successfully!")
        print(f"Final evaluation loss: {loss:.6f}")
        
        return loss
        
    except Exception as e:
        print(f"❌ Error in {config_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_modes(dataset='Digg', max_epochs=10):
    """Compare freezing vs full fine-tuning modes"""
    print(f"\n{'='*60}")
    print(f"Comparing fine-tuning modes for {dataset} dataset")
    print(f"{'='*60}")
    
    results = {}
    
    # Test freezing mode
    freeze_config = f"{dataset}_freeze"
    freeze_loss = run_single_finetune(freeze_config, max_epochs)
    results['freeze'] = freeze_loss
    
    # Test full fine-tuning mode
    full_config = f"{dataset}_full"
    full_loss = run_single_finetune(full_config, max_epochs)
    results['full'] = full_loss
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    if results['freeze'] is not None and results['full'] is not None:
        print(f"Freezing mode loss: {results['freeze']:.6f}")
        print(f"Full fine-tuning loss: {results['full']:.6f}")
    else:
        print("❌ Could not compare results due to errors")

def run_quick_demo():
    """Run a quick demonstration with Crime dataset"""
    print("🚀 Running quick fine-tuning demonstration...")
    
    # Use Crime dataset (smaller, faster)
    config_name = 'Crime_freeze'
    
    try:
        config = get_config(config_name)
        config['num_epochs'] = 5  # Very quick demo
        config['batch_size'] = 8
        
        print(f"Configuration: {config}")
        
        finetuner = FineTuneGraphDQN(**config)
        finetuner.train()
        
        print("✅ Quick demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def main():
    """Main function"""
    print("🎯 FINDER_ND Fine-tuning Example")
    print("This script demonstrates fine-tuning on real graphs")
    
    # Show available configurations
    print("\n📋 Available configurations:")
    list_configs()
    
    # Ask user what to run
    print("\n" + "="*60)
    print("Choose an option:")
    print("1. Quick demo (Crime dataset, 5 epochs)")
    run_quick_demo()
    exit()
    print("2. Compare freeze and full modes (Digg dataset)")
    compare_modes('Digg', max_epochs=10)
    print("3. Run specific configuration")
    config_name = input("Enter configuration name (e.g., 'crime_freeze'): ").strip()
    max_epochs = int(input("Enter number of epochs (default 10): ") or "10")
    run_single_finetune(config_name, max_epochs)



if __name__ == "__main__":
    main() 