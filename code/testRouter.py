#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Router Analysis Pipeline
Runs the full analysis pipeline for MoE router behavior
"""

import os
import sys

import argparse
import subprocess
import json
from pathlib import Path
from testUtils import load_config


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Command completed successfully")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Complete MoE Router Analysis Pipeline')
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained MoE model directory e.g. ./FINDER_moe_v2/models/MoE_BA_nrange_50_100_m_6")
    parser.add_argument("--analysis_dir", type=str, default="./router_analysis_results",
                       help="Directory to save all analysis results")
    parser.add_argument("--n_synthetic", type=int, default=50,
                       help="Number of synthetic graphs per configuration")
    parser.add_argument("--real_datasets", type=str, nargs='+', 
                       default=['Flickr', 'Crime'],
                       help="Real datasets to analyze")
    parser.add_argument("--skip_extraction", action="store_true",
                       help="Skip weight extraction if data already exists")
    parser.add_argument("--skip_analysis", action="store_true",
                       help="Skip router analysis if results already exist")
    parser.add_argument("--skip_visualization", action="store_true",
                       help="Skip visualization generation")
    
    args = parser.parse_args()

    # -----------------------------------------LOAD modules----------------------------------------------
    FINDER_type = args.model_path.split("/")[1]
    # sys.path.append(os.path.dirname(__file__) + os.sep + '../')
    if "moe" in FINDER_type:
        sys.path.append(os.path.dirname(__file__) + os.sep + f'{FINDER_type}/')
        sys.path.append(os.path.dirname(__file__) + os.sep + f'FINDER/')
        from GraphDQN import GraphDQN
        from MoEGraphDQN import MoEGraphDQN
    else:
        sys.path.append(os.path.dirname(__file__) + os.sep + f'{FINDER_type}/')
        from GraphDQN import GraphDQN
    
    # -------------------------------------LOAD configuration---------------------------------------------
    config = load_config()
    model_config = config['model_config']

    # Update model config if model_path is provided
    model_config = config['model_config']
    model_path_config_file = os.path.join(args.model_path,"config.json")
    if os.path.exists(model_path_config_file):
        # Parse model path to extract parameters
        with open(model_path_config_file, 'r') as f:
            model_config.update(json.load(f)["model_config"])

    # Create analysis directory
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(exist_ok=True)
    # Define subdirectories
    weights_dir = analysis_dir / "router_weights"
    router_analysis_dir = analysis_dir / "router_analysis"
    visualizations_dir = analysis_dir / "visualizations"
    
    weights_dir.mkdir(exist_ok=True)
    router_analysis_dir.mkdir(exist_ok=True)
    visualizations_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("MoE Router Analysis Pipeline")
    print("=" * 80)

    # Check if model path exists
    if not os.path.exists(f"{args.model_path}"):
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1
    
    success_count = 0
    total_steps = 3
    
    # -----------------------------Step 1: Extract router weights---------------------
    weights_file = weights_dir / "router_weights_analysis.csv"
    
    if not args.skip_extraction or not weights_file.exists():
        print(f"\nStep 1/{total_steps}: Extracting router weights...")
        
        cmd = [
            sys.executable, f"{os.path.dirname(__file__)}/extract_router_weights.py",
            "--model_path", str(args.model_path),
            "--save_dir", str(weights_dir),
            "--n_synthetic", str(args.n_synthetic),
            "--real_datasets"
        ] + args.real_datasets
        
        if run_command(cmd, "Router Weight Extraction"):
            success_count += 1
            print(f"✓ Router weights extracted to: {weights_dir}")
        else:
            print(f"✗ Failed to extract router weights")
    else:
        print(f"\nStep 1/{total_steps}: Skipping weight extraction (file exists)")
        success_count += 1
    
    # --------------------------Step 2: Run router analysis-----------------------------
    if not args.skip_analysis:
        print(f"\nStep 2/{total_steps}: Running router behavior analysis...")
        
        # Check if weights data exists
        if weights_file.exists():
            cmd = [
                sys.executable, f"{os.path.dirname(__file__)}/analyze_router.py",
                "--data_path", str(weights_file),
                "--save_dir", str(router_analysis_dir)
            ]
            
            if run_command(cmd, "Router Behavior Analysis"):
                success_count += 1
                print(f"✓ Router analysis completed: {router_analysis_dir}")
            else:
                print(f"✗ Failed to complete router analysis")
        else:
            print(f"✗ Cannot run analysis: weights file not found at {weights_file}")
            print("  Make sure Step 1 (weight extraction) completed successfully")
    else:
        print(f"\nStep 2/{total_steps}: Skipping router analysis")
        success_count += 1
    
    # --------------------------------Step 3: Generate visualizations-----------------------------
    if not args.skip_visualization:
        print(f"\nStep 3/{total_steps}: Generating visualizations...")
        
        # Use extracted weights data if available, otherwise use analysis data
        data_file = None
        if weights_file.exists():
            data_file = str(weights_file)
        else:
            # Look for analysis CSV files
            analysis_csv = router_analysis_dir / "routing_data.csv"
            if analysis_csv.exists():
                data_file = str(analysis_csv)
        
        if data_file:
            cmd = [
                sys.executable, f"{os.path.dirname(__file__)}/visualize_router_analysis.py",
                "--data_path", data_file,
                "--save_dir", str(visualizations_dir)
            ]
            
            if run_command(cmd, "Visualization Generation"):
                success_count += 1
                print(f"✓ Visualizations created: {visualizations_dir}")
            else:
                print(f"✗ Failed to generate visualizations")
        else:
            print(f"✗ No data file found for visualization")
    else:
        print(f"\nStep 3/{total_steps}: Skipping visualization generation")
        success_count += 1
    
    # -------------------------------------Summary----------------------------------
    print(f"\n{'='*80}")
    print("Pipeline Summary")
    print(f"{'='*80}")
    print(f"Completed steps: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("✓ All steps completed successfully!")
        print(f"\nResults available in:")
        print(f"  - Router weights: {weights_dir}")
        print(f"  - Analysis results: {router_analysis_dir}")
        print(f"  - Visualizations: {visualizations_dir}")
        
        # List key output files
        key_files = [
            weights_dir / "router_weights_analysis.csv",
            router_analysis_dir / "router_analysis_report.txt",
            visualizations_dir / "expert_usage_overview.png",
            visualizations_dir / "router_analysis_summary.txt"
        ]
        
        print(f"\nKey output files:")
        for file_path in key_files:
            if file_path.exists():
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (not found)")
        
        return 0
    else:
        print(f"✗ {total_steps - success_count} steps failed")
        print("Check the error messages above for details")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)