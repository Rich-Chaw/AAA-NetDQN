#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize router analysis results
Creates comprehensive plots and analysis of MoE router behavior using extracted data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
import os
import pickle

class RouterVisualizationSuite:
    """Comprehensive visualization suite for router analysis using extracted data"""
    
    def __init__(self, data_path, save_dir="./router_visualizations"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Load data from extract_router_weights.py output
        self.load_data(data_path)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self, data_path):
        """Load router analysis data from extract_router_weights.py output"""
        print(f"Loading data from {data_path}")
        
        if data_path.endswith('.csv'):
            self.df = pd.read_csv(data_path)
            print(f"Loaded CSV data: {len(self.df)} graphs")
        elif data_path.endswith('_full.pkl'):
            with open(data_path, 'rb') as f:
                self.raw_data = pickle.load(f)
            # Convert to DataFrame
            self.df = self.convert_raw_to_df(self.raw_data)
            print(f"Loaded pickle data: {len(self.df)} graphs")
        else:
            # Try to find corresponding files
            if data_path.endswith('.pkl'):
                csv_path = data_path.replace('.pkl', '.csv')
                if os.path.exists(csv_path):
                    self.df = pd.read_csv(csv_path)
                    print(f"Loaded corresponding CSV data: {len(self.df)} graphs")
                else:
                    raise ValueError(f"No CSV file found at {csv_path}")
            else:
                raise ValueError("Data file must be .csv or _full.pkl from extract_router_weights.py")
        
        # Add derived columns
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
                if 'normalized_weights' in item:
                    row[f'expert_{i}_normalized'] = item['normalized_weights'][i]
            
            # Add graph features
            if 'graph_features' in item:
                feature_names = ['norm_nodes', 'norm_m', 'norm_density', 'norm_avg_degree']
                for i, name in enumerate(feature_names):
                    row[name] = item['graph_features'][i]
            
            # Add top-k experts
            if 'top_k_experts' in item:
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
        
        # Density categories
        self.df['density_category'] = pd.cut(
            self.df['density'], 
            bins=[0, 0.01, 0.05, 0.1, 0.3, 1.0], 
            labels=['Very Low (<0.01)', 'Low (0.01-0.05)', 'Medium (0.05-0.1)', 'High (0.1-0.3)', 'Very High (>0.3)']
        )
        
        # Confidence categories
        self.df['confidence_category'] = pd.cut(
            self.df['max_weight'], 
            bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0], 
            labels=['Very Low (<0.3)', 'Low (0.3-0.5)', 'Medium (0.5-0.7)', 'High (0.7-0.9)', 'Very High (>0.9)']
        )
        
        # Extract graph type details
        self.df['base_graph_type'] = self.df['graph_type'].str.extract(r'(synthetic|real)')[0]
        self.df['graph_family'] = self.df['graph_type'].str.extract(r'(BA|real)')[0]
        
        print(f"Added derived features for {len(self.df)} graphs")
    
    def plot_expert_usage_overview(self):
        """Plot comprehensive expert usage analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Overall expert distribution
        ax = axes[0, 0]
        expert_counts = self.df['selected_expert'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, 4))
        bars = ax.bar(range(4), [expert_counts.get(i, 0) for i in range(4)], color=colors)
        ax.set_xlabel('Expert')
        ax.set_ylabel('Selection Count')
        ax.set_title('Overall Expert Selection Distribution')
        ax.set_xticks(range(4))
        ax.set_xticklabels([f'Expert {i}' for i in range(4)])
        
        # Add percentage labels
        total = len(self.df)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = height / total * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                   f'{percentage:.1f}%', ha='center', va='bottom')
        
        # 2. Expert selection by graph size
        ax = axes[0, 1]
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
        
        # 3. Expert selection by degree
        ax = axes[0, 2]
        if 'degree_category' in self.df.columns:
            degree_expert_crosstab = pd.crosstab(self.df['degree_category'], self.df['selected_expert'], normalize='index')
            # Use matplotlib directly instead of pandas plot to avoid compatibility issues
            bottom = np.zeros(len(degree_expert_crosstab))
            for i in range(4):
                if i in degree_expert_crosstab.columns:
                    values = degree_expert_crosstab[i].values
                    ax.bar(range(len(degree_expert_crosstab)), values, bottom=bottom, 
                          color=colors[i], label=f'Expert {i}', alpha=0.8)
                    bottom += values
            ax.set_xlabel('Degree Category')
            ax.set_ylabel('Proportion')
            ax.set_title('Expert Selection by Average Degree')
            ax.set_xticks(range(len(degree_expert_crosstab)))
            ax.set_xticklabels(degree_expert_crosstab.index, rotation=45)
            ax.legend(title='Expert')
        
        # 4. Router confidence distribution
        ax = axes[1, 0]
        ax.hist(self.df['max_weight'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        ax.axvline(self.df['max_weight'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {self.df["max_weight"].mean():.3f}')
        ax.set_xlabel('Maximum Expert Weight')
        ax.set_ylabel('Frequency')
        ax.set_title('Router Confidence Distribution')
        ax.legend()
        
        # 5. Entropy distribution
        ax = axes[1, 1]
        ax.hist(self.df['entropy'], bins=30, alpha=0.7, edgecolor='black', color='lightcoral')
        ax.axvline(self.df['entropy'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {self.df["entropy"].mean():.3f}')
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Frequency')
        ax.set_title('Router Uncertainty Distribution')
        ax.legend()
        
        # 6. Confidence vs Entropy scatter
        ax = axes[1, 2]
        scatter = ax.scatter(self.df['max_weight'], self.df['entropy'], 
                           c=self.df['selected_expert'], cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Maximum Expert Weight')
        ax.set_ylabel('Entropy')
        ax.set_title('Confidence vs Uncertainty')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Selected Expert')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "expert_usage_overview.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Created expert usage overview")
    
    def plot_expert_specialization_analysis(self):
        """Plot detailed expert specialization analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Expert weights by graph size (if available)
        ax = axes[0, 0]
        expert_cols = [f'expert_{i}_weight' for i in range(4)]
        
        if all(col in self.df.columns for col in expert_cols) and 'size_category' in self.df.columns:
            size_groups = self.df.groupby('size_category')[expert_cols].mean()
            # Use matplotlib directly instead of pandas plot to avoid compatibility issues
            x = np.arange(len(size_groups))
            width = 0.2
            colors = plt.cm.Set3(np.linspace(0, 1, 4))
            
            for i, col in enumerate(expert_cols):
                ax.bar(x + i * width, size_groups[col].values, width, 
                      label=f'Expert {i}', color=colors[i], alpha=0.8)
            
            ax.set_xlabel('Graph Size Category')
            ax.set_ylabel('Average Expert Weight')
            ax.set_title('Average Expert Weights by Graph Size')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(size_groups.index, rotation=45)
            ax.legend(title='Expert')
        else:
            ax.text(0.5, 0.5, 'Expert weights not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Expert Weights by Graph Size (N/A)')
        
        # 2. Specialization scatter plot
        ax = axes[0, 1]
        colors = plt.cm.Set1(np.linspace(0, 1, 4))
        
        for expert in range(4):
            expert_data = self.df[self.df['selected_expert'] == expert]
            if len(expert_data) > 0:
                ax.scatter(expert_data['n_nodes'], expert_data['avg_degree'], 
                          color=colors[expert], label=f'Expert {expert}', alpha=0.6, s=30)
        
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Average Degree')
        ax.set_title('Expert Specialization: Nodes vs Degree')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # 3. Expert confidence by specialization
        ax = axes[1, 0]
        
        # Box plot of confidence by expert
        expert_confidence_data = []
        expert_labels = []
        
        for expert in range(4):
            expert_data = self.df[self.df['selected_expert'] == expert]
            if len(expert_data) > 0:
                expert_confidence_data.append(expert_data['max_weight'].values)
                expert_labels.append(f'Expert {expert}')
        
        if expert_confidence_data:
            ax.boxplot(expert_confidence_data, labels=expert_labels)
            ax.set_ylabel('Router Confidence (Max Weight)')
            ax.set_title('Router Confidence by Selected Expert')
            ax.grid(True, alpha=0.3)
        
        # 4. Expert diversity analysis
        ax = axes[1, 1]
        
        # Expert usage frequency and confidence relationship
        expert_stats = []
        for expert in range(4):
            expert_data = self.df[self.df['selected_expert'] == expert]
            if len(expert_data) > 0:
                usage_freq = len(expert_data) / len(self.df)
                avg_confidence = expert_data['max_weight'].mean()
                expert_stats.append((expert, usage_freq, avg_confidence))
        
        if expert_stats:
            experts, frequencies, confidences = zip(*expert_stats)
            
            # Create bubble plot
            sizes = [f * 1000 for f in frequencies]  # Scale for visibility
            colors_bubble = plt.cm.viridis(np.linspace(0, 1, len(experts)))
            
            scatter = ax.scatter(experts, confidences, s=sizes, c=colors_bubble, alpha=0.6)
            
            ax.set_xlabel('Expert')
            ax.set_ylabel('Average Confidence')
            ax.set_title('Expert Usage vs Confidence\n(Bubble size = Usage frequency)')
            ax.set_xticks(range(4))
            ax.set_xticklabels([f'Expert {i}' for i in range(4)])
            ax.grid(True, alpha=0.3)
            
            # Add frequency labels
            for expert, freq, conf in expert_stats:
                ax.annotate(f'{freq:.1%}', (expert, conf), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "expert_specialization_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Created expert specialization analysis")
    
    def plot_graph_characteristics_analysis(self):
        """Plot graph characteristics and their relationship to routing"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Nodes vs Edges colored by expert
        ax = axes[0, 0]
        scatter = ax.scatter(self.df['n_nodes'], self.df['n_edges'], 
                           c=self.df['selected_expert'], cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Number of Edges')
        ax.set_title('Graph Size Distribution by Expert')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.colorbar(scatter, ax=ax, label='Selected Expert')
        
        # 2. Density vs Degree colored by expert
        ax = axes[0, 1]
        scatter = ax.scatter(self.df['density'], self.df['avg_degree'], 
                           c=self.df['selected_expert'], cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Density')
        ax.set_ylabel('Average Degree')
        ax.set_title('Density vs Degree by Expert')
        plt.colorbar(scatter, ax=ax, label='Selected Expert')
        
        # 3. Graph characteristics distribution
        ax = axes[0, 2]
        characteristics = ['n_nodes', 'avg_degree', 'density']
        
        # Normalize characteristics for comparison
        normalized_data = []
        labels = []
        
        for char in characteristics:
            if char in self.df.columns:
                # Normalize to 0-1 range
                char_data = self.df[char]
                if char_data.max() > char_data.min():
                    normalized = (char_data - char_data.min()) / (char_data.max() - char_data.min())
                    normalized_data.append(normalized.values)
                    labels.append(char.replace('_', ' ').title())
        
        if normalized_data:
            ax.boxplot(normalized_data, labels=labels)
            ax.set_ylabel('Normalized Value')
            ax.set_title('Graph Characteristics Distribution')
            ax.tick_params(axis='x', rotation=45)
        
        # 4. Expert selection by graph type
        ax = axes[1, 0]
        if 'base_graph_type' in self.df.columns:
            type_expert_crosstab = pd.crosstab(self.df['base_graph_type'], self.df['selected_expert'], normalize='index')
            # Use matplotlib directly instead of pandas plot to avoid compatibility issues
            colors = plt.cm.Set3(np.linspace(0, 1, 4))
            bottom = np.zeros(len(type_expert_crosstab))
            for i in range(4):
                if i in type_expert_crosstab.columns:
                    values = type_expert_crosstab[i].values
                    ax.bar(range(len(type_expert_crosstab)), values, bottom=bottom, 
                          color=colors[i], label=f'Expert {i}', alpha=0.8)
                    bottom += values
            ax.set_xlabel('Graph Type')
            ax.set_ylabel('Proportion')
            ax.set_title('Expert Selection by Graph Type')
            ax.set_xticks(range(len(type_expert_crosstab)))
            ax.set_xticklabels(type_expert_crosstab.index, rotation=45)
            ax.legend(title='Expert')
        
        # 5. Feature correlation matrix
        ax = axes[1, 1]
        
        corr_features = ['n_nodes', 'avg_degree', 'density', 'max_weight']
        available_features = [f for f in corr_features if f in self.df.columns]
        
        if len(available_features) > 1:
            corr_matrix = self.df[available_features].corr()
            im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(available_features)))
            ax.set_yticks(range(len(available_features)))
            ax.set_xticklabels([f.replace('_', ' ').title() for f in available_features], rotation=45)
            ax.set_yticklabels([f.replace('_', ' ').title() for f in available_features])
            ax.set_title('Feature Correlation Matrix')
            
            # Add correlation values
            for i in range(len(available_features)):
                for j in range(len(available_features)):
                    ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', 
                           color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
            
            plt.colorbar(im, ax=ax)
        
        # 6. Router performance by graph size
        ax = axes[1, 2]
        
        if 'size_category' in self.df.columns:
            size_performance = self.df.groupby('size_category').agg({
                'max_weight': 'mean',
                'entropy': 'mean'
            })
            
            x = range(len(size_performance))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], size_performance['max_weight'], 
                  width, label='Avg Confidence', alpha=0.8)
            ax2 = ax.twinx()
            ax2.bar([i + width/2 for i in x], size_performance['entropy'], 
                   width, label='Avg Entropy', alpha=0.8, color='orange')
            
            ax.set_xlabel('Graph Size Category')
            ax.set_ylabel('Average Confidence', color='blue')
            ax2.set_ylabel('Average Entropy', color='orange')
            ax.set_title('Router Performance by Graph Size')
            ax.set_xticks(x)
            ax.set_xticklabels(size_performance.index, rotation=45)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "graph_characteristics_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Created graph characteristics analysis")
    
    def plot_routing_patterns_detailed(self):
        """Plot detailed routing patterns and decision boundaries"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Decision boundaries in 2D space (nodes vs degree)
        ax = axes[0, 0]
        
        if len(self.df) > 10:  # Only if we have enough data
            # Plot actual data points
            colors = plt.cm.Set1(np.linspace(0, 1, 4))
            for expert in range(4):
                expert_data = self.df[self.df['selected_expert'] == expert]
                if len(expert_data) > 0:
                    ax.scatter(expert_data['n_nodes'], expert_data['avg_degree'], 
                             color=colors[expert], label=f'Expert {expert}', alpha=0.7, s=50)
            
            ax.set_xlabel('Number of Nodes')
            ax.set_ylabel('Average Degree')
            ax.set_title('Expert Selection Decision Space')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Expert weight distributions (if available)
        ax = axes[0, 1]
        expert_cols = [f'expert_{i}_weight' for i in range(4)]
        
        if all(col in self.df.columns for col in expert_cols):
            weight_data = []
            labels = []
            
            for i in range(4):
                col = f'expert_{i}_weight'
                weight_data.append(self.df[col].values)
                labels.append(f'Expert {i}')
            
            bp = ax.boxplot(weight_data, labels=labels, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, 4))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Expert Weight')
            ax.set_title('Expert Weight Distributions')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Expert weights not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Expert Weight Distributions (N/A)')
        
        # 3. Routing confidence patterns
        ax = axes[1, 0]
        
        if 'size_category' in self.df.columns:
            confidence_by_size = []
            size_labels = []
            
            for size_cat in self.df['size_category'].cat.categories:
                size_data = self.df[self.df['size_category'] == size_cat]
                if len(size_data) > 0:
                    confidence_by_size.append(size_data['max_weight'].values)
                    size_labels.append(size_cat)
            
            if confidence_by_size:
                ax.boxplot(confidence_by_size, labels=size_labels)
                ax.set_ylabel('Router Confidence')
                ax.set_title('Confidence Distribution by Graph Size')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        # 4. Expert usage vs confidence bubble plot
        ax = axes[1, 1]
        
        expert_stats = []
        for expert in range(4):
            expert_data = self.df[self.df['selected_expert'] == expert]
            if len(expert_data) > 0:
                usage_freq = len(expert_data) / len(self.df)
                avg_confidence = expert_data['max_weight'].mean()
                expert_stats.append((expert, usage_freq, avg_confidence))
        
        if expert_stats:
            experts, frequencies, confidences = zip(*expert_stats)
            
            # Create bubble plot
            sizes = [f * 1000 for f in frequencies]  # Scale for visibility
            colors_bubble = plt.cm.viridis(np.linspace(0, 1, len(experts)))
            
            scatter = ax.scatter(experts, confidences, s=sizes, c=colors_bubble, alpha=0.6)
            
            ax.set_xlabel('Expert')
            ax.set_ylabel('Average Confidence')
            ax.set_title('Expert Usage vs Confidence\n(Bubble size = Usage frequency)')
            ax.set_xticks(range(4))
            ax.set_xticklabels([f'Expert {i}' for i in range(4)])
            ax.grid(True, alpha=0.3)
            
            # Add frequency labels
            for expert, freq, conf in expert_stats:
                ax.annotate(f'{freq:.1%}', (expert, conf), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "routing_patterns_detailed.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Created detailed routing patterns analysis")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        report_path = os.path.join(self.save_dir, "router_analysis_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("MoE Router Analysis - Comprehensive Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Dataset overview
            f.write("Dataset Overview:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total graphs analyzed: {len(self.df)}\n")
            
            if 'graph_type' in self.df.columns:
                type_counts = self.df['graph_type'].value_counts()
                f.write("Graph types:\n")
                for graph_type, count in type_counts.items():
                    f.write(f"  {graph_type}: {count} graphs\n")
            f.write("\n")
            
            # Expert usage analysis
            f.write("Expert Usage Analysis:\n")
            f.write("-" * 25 + "\n")
            expert_counts = self.df['selected_expert'].value_counts().sort_index()
            
            for expert in range(4):
                count = expert_counts.get(expert, 0)
                percentage = count / len(self.df) * 100
                f.write(f"Expert {expert}: {count} selections ({percentage:.1f}%)\n")
            
            # Most and least used experts
            if len(expert_counts) > 0:
                most_used = expert_counts.idxmax()
                least_used = expert_counts.idxmin()
                f.write(f"\nMost used expert: Expert {most_used} ({expert_counts[most_used]} selections)\n")
                f.write(f"Least used expert: Expert {least_used} ({expert_counts[least_used]} selections)\n\n")
            
            # Router confidence analysis
            f.write("Router Confidence Analysis:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average confidence (max weight): {self.df['max_weight'].mean():.3f} ± {self.df['max_weight'].std():.3f}\n")
            f.write(f"Median confidence: {self.df['max_weight'].median():.3f}\n")
            f.write(f"Min confidence: {self.df['max_weight'].min():.3f}\n")
            f.write(f"Max confidence: {self.df['max_weight'].max():.3f}\n")
            
            f.write(f"\nAverage entropy: {self.df['entropy'].mean():.3f} ± {self.df['entropy'].std():.3f}\n")
            f.write(f"Median entropy: {self.df['entropy'].median():.3f}\n\n")
            
            # Graph characteristics analysis
            f.write("Graph Characteristics Analysis:\n")
            f.write("-" * 35 + "\n")
            
            char_cols = ['n_nodes', 'n_edges', 'avg_degree', 'density']
            available_chars = [col for col in char_cols if col in self.df.columns]
            
            for col in available_chars:
                f.write(f"{col.replace('_', ' ').title()}:\n")
                f.write(f"  Mean: {self.df[col].mean():.3f}\n")
                f.write(f"  Std: {self.df[col].std():.3f}\n")
                f.write(f"  Range: {self.df[col].min():.3f} - {self.df[col].max():.3f}\n")
            f.write("\n")
            
            # Expert specialization insights
            f.write("Expert Specialization Insights:\n")
            f.write("-" * 35 + "\n")
            
            for expert in range(4):
                expert_data = self.df[self.df['selected_expert'] == expert]
                if len(expert_data) > 0:
                    f.write(f"Expert {expert}:\n")
                    f.write(f"  Graphs handled: {len(expert_data)}\n")
                    f.write(f"  Avg confidence: {expert_data['max_weight'].mean():.3f}\n")
                    
                    if 'n_nodes' in expert_data.columns:
                        f.write(f"  Avg nodes: {expert_data['n_nodes'].mean():.1f}\n")
                    if 'avg_degree' in expert_data.columns:
                        f.write(f"  Avg degree: {expert_data['avg_degree'].mean():.2f}\n")
                    if 'density' in expert_data.columns:
                        f.write(f"  Avg density: {expert_data['density'].mean():.4f}\n")
                    f.write("\n")
            
            # Recommendations
            f.write("Recommendations for Router Improvement:\n")
            f.write("-" * 40 + "\n")
            
            # Check for imbalanced expert usage
            if len(expert_counts) > 0:
                usage_std = expert_counts.std()
                usage_mean = expert_counts.mean()
                if usage_std / usage_mean > 0.5:
                    f.write("* Expert usage is imbalanced. Consider:\n")
                    f.write("  - Adjusting router network architecture\n")
                    f.write("  - Modifying load balancing loss weight\n")
                    f.write("  - Rebalancing training data\n\n")
            
            # Check confidence levels
            low_confidence_ratio = (self.df['max_weight'] < 0.5).sum() / len(self.df)
            if low_confidence_ratio > 0.3:
                f.write("* Router shows low confidence on many graphs. Consider:\n")
                f.write("  - Increasing router network capacity\n")
                f.write("  - Adding more diverse graph features\n")
                f.write("  - Adjusting top-k parameter\n\n")
            
            # Check for unused experts
            unused_experts = [i for i in range(4) if expert_counts.get(i, 0) == 0]
            if unused_experts:
                f.write(f"* Experts {unused_experts} are never selected. Consider:\n")
                f.write("  - Reviewing expert architectures\n")
                f.write("  - Adjusting routing criteria\n")
                f.write("  - Adding more diverse training data\n\n")
        
        print(f"Generated comprehensive summary report: {report_path}")
    
    def create_all_visualizations(self):
        """Create all visualizations and reports"""
        print("Creating comprehensive router analysis visualizations...")
        
        self.plot_expert_usage_overview()
        self.plot_expert_specialization_analysis()
        self.plot_graph_characteristics_analysis()
        self.plot_routing_patterns_detailed()
        self.generate_summary_report()
        
        print(f"\n{'='*60}")
        print("Router Analysis Visualization Complete!")
        print(f"All plots and reports saved to: {self.save_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Visualize router analysis results from extract_router_weights.py')
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to router analysis data (.csv or _full.pkl from extract_router_weights.py)")
    parser.add_argument("--save_dir", type=str, default="./router_visualizations",
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return
    
    print("=" * 60)
    print("Router Analysis Visualization Suite")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Save directory: {args.save_dir}")
    
    try:
        # Create visualization suite
        viz_suite = RouterVisualizationSuite(args.data_path, args.save_dir)
        
        # Create all visualizations
        viz_suite.create_all_visualizations()
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()