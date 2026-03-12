#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Router Analysis Script for MoE GraphDQN
Analyzes router behavior using data from extract_router_weights.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from collections import defaultdict
import pickle

class RouterAnalyzer:
    """Analyzer for MoE router network behavior using extracted data"""
    
    def __init__(self, data_path, save_dir="./router_analysis"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Load extracted router data
        self.load_router_data(data_path)
    
    def load_router_data(self, data_path):
        """Load router data from extract_router_weights.py output"""
        print(f"Loading router data from {data_path}")
        
        if data_path.endswith('.csv'):
            self.df = pd.read_csv(data_path)
            print(f"Loaded CSV data: {len(self.df)} graphs")
        elif data_path.endswith('_full.pkl'):
            with open(data_path, 'rb') as f:
                raw_data = pickle.load(f)
            self.df = self.convert_raw_to_df(raw_data)
            print(f"Loaded pickle data: {len(self.df)} graphs")
        else:
            raise ValueError("Data path must be .csv or _full.pkl file from extract_router_weights.py")
        
        # Add derived features for analysis
        self.add_derived_features()
    
    def convert_raw_to_df(self, raw_data):
        """Convert raw pickle data to DataFrame"""
        rows = []
        for item in raw_data:
            row = {
                'graph_type': item['graph_type'],
                'graph_label': item['graph_label'],
                'n_nodes': item['n_nodes'],
                'n_edges': item['n_edges'],
                'density': item['density'],
                'avg_degree': item['avg_degree'],
                'selected_expert': item['selected_expert'],
                'max_weight': item['max_weight'],
                'entropy': item['entropy'],
            }
            
            # Add expert weights
            for i in range(4):
                row[f'expert_{i}_weight'] = item['expert_weights'][i]
                row[f'expert_{i}_normalized'] = item['normalized_weights'][i]
            
            # Add graph features
            feature_names = ['norm_nodes', 'norm_m', 'norm_density', 'norm_avg_degree']
            for i, name in enumerate(feature_names):
                row[name] = item['graph_features'][i]
            
            # Add top-k experts
            row['top_k_expert_1'] = item['top_k_experts'][0]
            row['top_k_expert_2'] = item['top_k_experts'][1]
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def add_derived_features(self):
        """Add derived features for analysis"""
        # Graph size categories
        self.df['size_category'] = pd.cut(
            self.df['n_nodes'], 
            bins=[0, 30, 50, 100, 200, float('inf')], 
            labels=['Tiny (<30)', 'Small (30-50)', 'Medium (50-100)', 'Large (100-200)', 'Huge (>200)']
        )
        
        # Degree categories  
        self.df['degree_category'] = pd.cut(
            self.df['avg_degree'], 
            bins=[0, 2, 4, 8, 16, float('inf')], 
            labels=['Very Sparse (<2)', 'Sparse (2-4)', 'Medium (4-8)', 'Dense (8-16)', 'Very Dense (>16)']
        )
        
        # Confidence categories
        self.df['confidence_category'] = pd.cut(
            self.df['max_weight'], 
            bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0], 
            labels=['Very Low (<0.3)', 'Low (0.3-0.5)', 'Medium (0.5-0.7)', 'High (0.7-0.9)', 'Very High (>0.9)']
        )
        
        # Extract graph type details
        self.df['base_graph_type'] = self.df['graph_type'].str.extract(r'(synthetic|real)')[0]
        
        print(f"Added derived features for {len(self.df)} graphs")
    
    def analyze_routing_patterns(self):
        """Analyze routing patterns from the loaded data"""
        print("Analyzing routing patterns...")
        
        # Basic statistics
        total_graphs = len(self.df)
        print(f"Total graphs: {total_graphs}")
        
        # Expert usage analysis
        expert_usage = self.df['selected_expert'].value_counts().sort_index()
        print("\nExpert Usage:")
        for expert in range(4):
            count = expert_usage.get(expert, 0)
            percentage = count / total_graphs * 100
            print(f"  Expert {expert}: {count} graphs ({percentage:.1f}%)")
        
        # Confidence analysis
        avg_confidence = self.df['max_weight'].mean()
        avg_entropy = self.df['entropy'].mean()
        print(f"\nRouter Performance:")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Average entropy: {avg_entropy:.3f}")
        
        # Specialization analysis
        print("\nExpert Specialization:")
        for expert in range(4):
            expert_data = self.df[self.df['selected_expert'] == expert]
            if len(expert_data) > 0:
                avg_nodes = expert_data['n_nodes'].mean()
                avg_degree = expert_data['avg_degree'].mean()
                avg_density = expert_data['density'].mean()
                print(f"  Expert {expert}: avg_nodes={avg_nodes:.1f}, avg_degree={avg_degree:.2f}, avg_density={avg_density:.4f}")
        
        return {
            'total_graphs': total_graphs,
            'expert_usage': expert_usage.to_dict(),
            'avg_confidence': avg_confidence,
            'avg_entropy': avg_entropy
        }
    
    def create_basic_visualizations(self):
        """Create basic visualizations of router behavior"""
        print(f"Creating basic visualizations in {self.save_dir}...")
        
        # 1. Expert selection distribution
        plt.figure(figsize=(10, 6))
        expert_counts = self.df['selected_expert'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, 4))
        bars = plt.bar(range(4), [expert_counts.get(i, 0) for i in range(4)], color=colors)
        plt.xlabel('Expert')
        plt.ylabel('Selection Count')
        plt.title('Expert Selection Distribution')
        plt.xticks(range(4), [f'Expert {i}' for i in range(4)])
        
        # Add percentage labels
        total = len(self.df)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = height / total * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                   f'{percentage:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "expert_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Router confidence distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['max_weight'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        plt.axvline(self.df['max_weight'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["max_weight"].mean():.3f}')
        plt.xlabel('Maximum Expert Weight')
        plt.ylabel('Frequency')
        plt.title('Router Confidence Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(self.df['entropy'], bins=30, alpha=0.7, edgecolor='black', color='lightcoral')
        plt.axvline(self.df['entropy'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["entropy"].mean():.3f}')
        plt.xlabel('Entropy')
        plt.ylabel('Frequency')
        plt.title('Router Uncertainty Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "router_confidence.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Expert selection by graph characteristics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Nodes vs Expert
        ax = axes[0, 0]
        scatter = ax.scatter(self.df['n_nodes'], self.df['selected_expert'], 
                           c=self.df['selected_expert'], cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Selected Expert')
        ax.set_title('Expert Selection vs Graph Size')
        ax.set_yticks(range(4))
        ax.set_yticklabels([f'Expert {i}' for i in range(4)])
        
        # Degree vs Expert
        ax = axes[0, 1]
        scatter = ax.scatter(self.df['avg_degree'], self.df['selected_expert'], 
                           c=self.df['selected_expert'], cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Average Degree')
        ax.set_ylabel('Selected Expert')
        ax.set_title('Expert Selection vs Average Degree')
        ax.set_yticks(range(4))
        ax.set_yticklabels([f'Expert {i}' for i in range(4)])
        
        # Density vs Expert
        ax = axes[1, 0]
        scatter = ax.scatter(self.df['density'], self.df['selected_expert'], 
                           c=self.df['selected_expert'], cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Density')
        ax.set_ylabel('Selected Expert')
        ax.set_title('Expert Selection vs Density')
        ax.set_yticks(range(4))
        ax.set_yticklabels([f'Expert {i}' for i in range(4)])
        
        # Expert weights by size category
        ax = axes[1, 1]
        if 'size_category' in self.df.columns:
            size_expert_crosstab = pd.crosstab(self.df['size_category'], self.df['selected_expert'], normalize='index')
            # Use matplotlib directly instead of pandas plot to avoid compatibility issues
            bottom = np.zeros(len(size_expert_crosstab))
            for i in range(4):
                if i in size_expert_crosstab.columns:
                    values = size_expert_crosstab[i].values
                    ax.bar(range(len(size_expert_crosstab)), values, bottom=bottom, 
                          color=colors[i], label=f'Expert {i}', alpha=0.8)
                    bottom += values
            ax.set_xlabel('Graph Size Category')
            ax.set_ylabel('Proportion')
            ax.set_title('Expert Selection by Graph Size')
            ax.set_xticks(range(len(size_expert_crosstab)))
            ax.set_xticklabels(size_expert_crosstab.index, rotation=45)
            ax.legend(title='Expert')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "expert_by_characteristics.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Basic visualizations created successfully!")
    
    def generate_summary_report(self):
        """Generate a summary report of router analysis"""
        report_path = os.path.join(self.save_dir, "router_analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("MoE Router Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write("Basic Statistics:\n")
            f.write(f"Total graphs analyzed: {len(self.df)}\n")
            f.write(f"Graph types: {self.df['graph_type'].unique().tolist()}\n\n")
            
            # Expert usage
            f.write("Expert Usage:\n")
            expert_counts = self.df['selected_expert'].value_counts().sort_index()
            for expert in range(4):
                count = expert_counts.get(expert, 0)
                percentage = count / len(self.df) * 100
                f.write(f"Expert {expert}: {count} graphs ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Router confidence
            f.write("Router Confidence:\n")
            f.write(f"Average max weight: {self.df['max_weight'].mean():.3f} ± {self.df['max_weight'].std():.3f}\n")
            f.write(f"Average entropy: {self.df['entropy'].mean():.3f} ± {self.df['entropy'].std():.3f}\n\n")
            
            # Graph characteristics summary
            f.write("Graph Characteristics Summary:\n")
            numeric_cols = ['n_nodes', 'avg_degree', 'density']
            for col in numeric_cols:
                if col in self.df.columns:
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    f.write(f"{col}: {mean_val:.3f} ± {std_val:.3f}\n")
            f.write("\n")
            
            # Expert specialization insights
            f.write("Expert Specialization Insights:\n")
            for expert in range(4):
                expert_data = self.df[self.df['selected_expert'] == expert]
                if len(expert_data) > 0:
                    f.write(f"Expert {expert}:\n")
                    f.write(f"  Graphs handled: {len(expert_data)}\n")
                    f.write(f"  Avg confidence: {expert_data['max_weight'].mean():.3f}\n")
                    f.write(f"  Avg nodes: {expert_data['n_nodes'].mean():.1f}\n")
                    f.write(f"  Avg degree: {expert_data['avg_degree'].mean():.2f}\n")
                    f.write(f"  Avg density: {expert_data['density'].mean():.4f}\n")
                    f.write("\n")
        
        print(f"Summary report saved to: {report_path}")
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("Running router analysis...")
        
        # Analyze patterns
        patterns = self.analyze_routing_patterns()
        
        # Create visualizations
        self.create_basic_visualizations()
        
        # Generate report
        self.generate_summary_report()
        
        # Save processed data
        self.df.to_csv(os.path.join(self.save_dir, "processed_router_data.csv"), index=False)
        
        print(f"\nAnalysis complete! Results saved to: {self.save_dir}")
        return patterns


def main():
    parser = argparse.ArgumentParser(description='Analyze MoE router network behavior')
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to router data from extract_router_weights.py (.csv or _full.pkl)")
    parser.add_argument("--save_dir", type=str, default="./router_analysis",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return
    
    print("=" * 60)
    print("MoE Router Analysis")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Save directory: {args.save_dir}")
    
    try:
        # Create analyzer and run analysis
        analyzer = RouterAnalyzer(args.data_path, args.save_dir)
        patterns = analyzer.run_analysis()
        
        print(f"\n{'='*60}")
        print("Router analysis completed successfully!")
        print(f"Results saved to: {args.save_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()