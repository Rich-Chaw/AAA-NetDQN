# GraphDQN Configuration System

This document explains how to use the configuration system for the GraphDQN model.

## Configuration File Structure

The configuration is stored in `config.json` with the following structure:

```json
{
    "model_config": {
        "g_type": "barabasi_albert",
        "gnn_model": "GIN",
        "target_graph": "Digg",
        "num_min": 30,
        "num_max": 50,
        "ckpt_file": "GIN_nrange_30_50_iter_2700.ckpt"
    },
    "evaluation_config": {
        "datasets": ["Crime", "HI-II-14", "Digg"],
        "step_ratio": 0.01,
        "strategy_id": 0,
        "save_dir": "result/temp",
        "random_remove_ratios": [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
        "repeat_times": 100
    },
    "train_config": {
        "batch_size": 256,
        "learning_rate": 0.0001,
        "embedding_size": 64,
        "max_iteration": 1000000,
        "memory_size": 500000,
        "update_time": 1000
    },
    "data_config": {
        "data_path": "../../dataset/real",
        "model_path": "./FINDER_ND/models/barabasi_albert/"
    }
}
```

## Configuration Parameters

### Model Configuration (`model_config`)

- **g_type**: Graph type for training (`barabasi_albert`, `erdos_renyi`, `powerlaw`, `small-world`, `ego`)
- **gnn_model**: GNN model type (`GIN`, `GraphSage`, `GCN`)
- **target_graph**: Target graph name for evaluation
- **num_min**: Minimum number of nodes in training graphs
- **num_max**: Maximum number of nodes in training graphs
- **ckpt_file**: Checkpoint file name to load (optional, can be `null`)

### Evaluation Configuration (`evaluation_config`)

- **datasets**: List of dataset names to evaluate
- **step_ratio**: Step ratio for solution generation
- **strategy_id**: Strategy ID for evaluation (0: no insert, 1: count, 2: rank, 3: multiply)
- **save_dir**: Directory to save results
- **random_remove_ratios**: List of ratios for random node removal tests
- **repeat_times**: Number of repetitions for random removal tests

### Training Configuration (`train_config`)

- **batch_size**: Training batch size
- **learning_rate**: Learning rate for optimization
- **embedding_size**: Size of node embeddings
- **max_iteration**: Maximum training iterations
- **memory_size**: Size of replay memory
- **update_time**: Frequency of target network updates

### Data Configuration (`data_config`)

- **data_path**: Path to dataset directory
- **model_path**: Path to model checkpoint directory

## Usage Examples

### Basic Usage

```python
from testReal import load_config, create_graphdqn_from_config

# Load configuration
config = load_config('config.json')

# Create GraphDQN model
dqn = create_graphdqn_from_config(config)

# Use the model
print(f"Model type: {dqn.g_type}")
print(f"GNN model: {dqn.embeddingMethod}")
```

### Running Evaluation

```python
# Run the evaluation script
python testReal.py
```

This will:
1. Load the configuration from `config.json`
2. Create a GraphDQN model with the specified parameters
3. Run evaluation on the specified datasets
4. Save results to the configured directory

### Example Configuration

Here's an example configuration for a GIN model trained on Barabasi-Albert graphs:

```json
{
    "model_config": {
        "g_type": "barabasi_albert",
        "gnn_model": "GIN",
        "target_graph": "Digg",
        "num_min": 30,
        "num_max": 50,
        "ckpt_file": "GIN_nrange_30_50_iter_2700.ckpt"
    },
    "evaluation_config": {
        "datasets": ["Crime", "HI-II-14", "Digg"],
        "step_ratio": 0.01,
        "strategy_id": 0,
        "save_dir": "result/temp"
    }
}
```

## Modifying Configuration

To use different parameters:

1. Edit `config.json` with your desired settings
2. Run `python testReal.py` to use the new configuration
3. Or use the configuration programmatically:

```python
config = load_config()
config['model_config']['gnn_model'] = 'GraphSage'
config['evaluation_config']['datasets'] = ['Crime', 'Digg']
dqn = create_graphdqn_from_config(config)
```

## Error Handling

The configuration system includes error handling for:
- Missing configuration file
- Invalid JSON format
- Missing required parameters
- Import errors for optional dependencies

## Testing the Configuration

Run the example script to test your configuration:

```bash
python example_usage.py
```

This will load the configuration and attempt to create a GraphDQN model, providing feedback on any issues. 