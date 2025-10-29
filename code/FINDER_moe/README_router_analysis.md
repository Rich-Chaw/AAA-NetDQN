# MoE Router Analysis Suite

This suite provides comprehensive analysis tools for understanding how the Mixture of Experts (MoE) router network behaves when processing different types of graphs in the network dismantling task.

## Overview

The MoE GraphDQN model uses a router network to decide which expert should handle each graph based on graph characteristics. This analysis suite helps you understand:

- Which experts are selected for different graph types
- How confident the router is in its decisions
- What graph characteristics influence expert selection
- Whether experts are properly specialized
- How to improve the router architecture

## Files

### Core Analysis Scripts
1. **`extract_router_weights.py`** - Extract actual router weights
   - Extracts router network outputs from trained models
   - Provides detailed routing data for analysis
   - Saves results in both CSV and pickle formats

2. **`analyze_router.py`** - Main router behavior analysis
   - Analyzes routing patterns across different graph types
   - Generates comprehensive visualizations
   - Creates summary reports

3. **`visualize_router_analysis.py`** - Advanced visualization suite
   - Creates publication-quality plots
   - Generates detailed analysis reports
   - Supports multiple data formats

4. **`run_router_analysis.py`** - Complete analysis pipeline
   - Runs all analysis steps in sequence
   - Handles error checking and file management
   - Provides comprehensive results

## Quick Start

### Option 1: Complete Pipeline (Recommended)

Run the complete analysis pipeline with a single command:

```bash
python run_router_analysis.py \
    --model_path "./models/MoE_BA_nrange_50_100_m_6" \
    --analysis_dir "./router_analysis_results" \
    --n_synthetic 100 \
    --real_datasets Flickr Crime Digg
```

### Option 2: Step-by-Step Analysis

If you prefer to run each step individually:

```bash
# Step 1: Extract router weights (requires trained model)
python extract_router_weights.py \
    --model_path "MoE_BA_nrange_50_100_m_6" \
    --save_dir "./router_weights" \
    --n_synthetic 50 \
    --real_datasets Flickr Crime

# Step 2: Analyze router behavior (uses extracted data)
python analyze_router.py \
    --data_path "./router_weights/router_weights_analysis.csv" \
    --save_dir "./router_analysis"

# Step 3: Generate visualizations (uses extracted data)
python visualize_router_analysis.py \
    --data_path "./router_weights/router_weights_analysis.csv" \
    --save_dir "./router_visualizations"
```

## Command Line Arguments

### extract_router_weights.py Arguments

- `--model_path`: Path to trained MoE model directory (required)
- `--save_dir`: Directory to save extracted weights (default: "./router_weights")
- `--n_synthetic`: Number of synthetic graphs per configuration (default: 50)
- `--real_datasets`: List of real datasets to analyze (default: ['Flickr', 'Crime'])

### analyze_router.py Arguments

- `--data_path`: Path to router data from extract_router_weights.py (required)
- `--save_dir`: Directory to save analysis results (default: "./router_analysis")

### visualize_router_analysis.py Arguments

- `--data_path`: Path to router data from extract_router_weights.py (required)
- `--save_dir`: Directory to save visualizations (default: "./router_visualizations")

### run_router_analysis.py Arguments (Pipeline)

- `--model_path`: Path to trained MoE model directory (required)
- `--analysis_dir`: Root directory for all analysis results (default: "./router_analysis_results")
- `--n_synthetic`: Number of synthetic graphs per configuration (default: 50)
- `--real_datasets`: List of real datasets to analyze (default: ['Flickr', 'Crime'])
- `--skip_extraction`: Skip weight extraction if data exists
- `--skip_analysis`: Skip router analysis if results exist
- `--skip_visualization`: Skip visualization generation

## Output Files

The analysis generates several types of output:

### Data Files
- `router_weights_analysis.csv` - Extracted router weights and graph features
- `router_weights_analysis_full.pkl` - Complete routing data (pickle format)
- `routing_data.csv` - Router behavior analysis results
- `graph_features_data.csv` - Graph characteristics and routing decisions

### Visualizations
- `expert_usage_overview.png` - Overall expert selection patterns
- `expert_specialization_analysis.png` - Expert specialization details
- `graph_characteristics_analysis.png` - Graph features vs routing
- `routing_patterns_detailed.png` - Detailed routing decision analysis
- `router_confidence.png` - Router confidence metrics

### Reports
- `router_analysis_report.txt` - Comprehensive analysis summary
- `router_analysis_summary.txt` - Detailed findings and recommendations

## Understanding the Results

### Expert Specialization

The analysis will show you:
- **Expert 0**: Typically handles sparse BA graphs (low m parameter)
- **Expert 1**: Typically handles dense BA graphs (high m parameter)  
- **Expert 2**: Typically handles small graphs (n < 50 nodes)
- **Expert 3**: Typically handles large graphs (n > 100 nodes)

### Key Metrics

1. **Router Confidence**: Maximum expert weight (higher = more confident)
2. **Entropy**: Uncertainty in expert selection (lower = more decisive)
3. **Expert Usage**: How frequently each expert is selected
4. **Specialization**: How well experts focus on specific graph types

### Interpretation Guidelines

- **High confidence + Low entropy**: Router is decisive and confident
- **Low confidence + High entropy**: Router is uncertain, may need improvement
- **Imbalanced expert usage**: Some experts may be underutilized
- **Poor specialization**: Experts may not be learning distinct patterns

## Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure the model path contains valid checkpoints
2. **Memory issues**: Reduce `n_synthetic` for large graph analysis
3. **Missing dependencies**: Install required packages (matplotlib, seaborn, pandas)
4. **Data format errors**: Check that CSV files have expected columns

### Performance Tips

- Use smaller `n_synthetic` values for initial exploration
- Run analysis on a subset of real datasets first
- Use `--skip_*` flags to avoid re-running completed steps

## Customization

### Adding New Graph Types

To analyze additional graph types, modify the `synthetic_configs` in the analysis scripts:

```python
synthetic_configs = [
    {'nrange': '30_50', 'm': 2},
    {'nrange': '50_100', 'm': 4},
    # Add your configurations here
]
```

### Custom Visualizations

The visualization suite is modular. You can:
- Modify existing plot functions
- Add new analysis metrics
- Customize color schemes and layouts
- Export data for external analysis tools

## Integration with Training

Use the analysis results to improve your MoE model:

1. **Adjust router architecture** based on confidence metrics
2. **Modify load balancing weights** if expert usage is imbalanced
3. **Add graph features** if routing decisions seem suboptimal
4. **Retrain with balanced data** if some experts are underutilized

## Example Workflow

```bash
# 1. Train your MoE model
python train_moe.py

# 2. Run complete router analysis
python run_router_analysis.py --model_path "./models/MoE_BA_nrange_50_100_m_6"

# 3. Review results
# - Check router_analysis_summary.txt for key findings
# - Examine visualizations for patterns
# - Use insights to improve model architecture

# 4. Iterate and improve
# - Modify router network based on findings
# - Retrain model with improvements
# - Re-run analysis to validate improvements
```

This analysis suite provides deep insights into your MoE router's behavior and helps you build more effective mixture of experts models for network dismantling tasks.