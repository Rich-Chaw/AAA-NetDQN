# FINDER_ND Fine-tuning Guide

This guide explains how to fine-tune the pre-trained GraphDQN model on real graphs using two different strategies.

## Overview

The fine-tuning script implements two strategies for adapting the pre-trained GraphDQN model to real-world graphs:

1. **Freezing Mode**: Freeze the encoder parameters and train only the decoder
2. **Full Fine-tuning Mode**: Train both encoder and decoder end-to-end

## Key Components

### 1. GraphEncoder (from `graph_dqn_modules.py`)
- **GIN (Graph Isomorphism Network)**: Multi-layer GNN with learnable epsilon parameters
- **GraphSage**: Neighborhood aggregation with concatenation
- **S2V**: Simplified structure2vec implementation

### 2. MLPDecoder (from `graph_dqn_modules.py`)
- 2-layer MLP that maps state-action embeddings to Q-values
- Cross-product operation between action and graph embeddings
- Auxiliary features integration

### 3. PrepareBatchGraph (from `PrepareBatchGraph.pyx`)
- Handles graph batching and sparse matrix creation
- Implements node masking for covered nodes
- Creates efficient sparse representations

## Fine-tuning Modes

### Freezing Mode (`fine_tune_mode='freeze'`)

**Strategy**: Treat the pre-trained encoder as a static feature extractor and train only the decoder.

**Advantages**:
- Faster training (fewer parameters to update)
- Less memory usage
- Preserves pre-trained graph representations
- Good for small datasets or when encoder is well-trained

**Use Cases**:
- Limited computational resources
- Small target datasets
- When pre-trained encoder is already well-optimized
- Quick adaptation to new domains

**Implementation**:
```python
# Only decoder parameters are trained
encoder_vars = tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
decoder_vars = tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

# Freeze encoder, train only decoder
self.train_op = tf1.train.AdamOptimizer(self.learning_rate).minimize(
    self.loss_op, var_list=decoder_vars
)
```

### Full Fine-tuning Mode (`fine_tune_mode='full'`)

**Strategy**: Initialize encoder with pre-trained weights and train both encoder and decoder end-to-end.

**Advantages**:
- Better adaptation to target domain
- Can learn domain-specific graph representations
- Higher potential performance
- Full model optimization

**Use Cases**:
- Large target datasets
- Significant domain shift from pre-training
- When maximum performance is required
- Sufficient computational resources available

**Implementation**:
```python
# Train all parameters
self.train_op = tf1.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op)
```

## Usage

### Basic Usage

```python
from FINDER_ND_FineTune import FineTuneGraphDQN

# Freezing mode
finetuner = FineTuneGraphDQN(
    pretrained_ckpt_path='./models',
    real_graph_path='dataset/real/Digg.txt',
    fine_tune_mode='freeze',
    learning_rate=0.001,
    batch_size=16,
    num_epochs=50
)

finetuner.train()
```

### Using Configuration Files

```python
from finetune_config import get_config
from FINDER_ND_FineTune import FineTuneGraphDQN

# Get pre-configured settings
config = get_config('digg_freeze')  # or 'digg_full'

# Initialize and train
finetuner = FineTuneGraphDQN(**config)
finetuner.train()
```

### Command Line Usage

```bash
# List available configurations
python finetune_config.py

# Run fine-tuning with specific configuration
python FINDER_ND_FineTune.py
```

## Configuration Options

### Dataset-Specific Configurations

| Dataset | Nodes | Edges | Recommended Mode | Learning Rate | Batch Size |
|---------|-------|-------|------------------|---------------|------------|
| Crime | ~1,474 | ~2,200 | Both | 0.001 | 32 |
| Digg | ~30,398 | ~85,155 | Freeze | 0.0005 | 8 |
| Enron | ~36,692 | ~183,831 | Freeze | 0.0005 | 16 |
| Facebook | ~4,039 | ~88,234 | Freeze | 0.0001 | 4 |

### Key Parameters

- **`embedding_method`**: 'GIN', 'graphSage', or 'S2V'
- **`embedding_size`**: Dimension of node/graph embeddings (default: 64)
- **`learning_rate`**: Lower for larger graphs or full fine-tuning
- **`batch_size`**: Smaller for larger graphs due to memory constraints
- **`num_epochs`**: More epochs for larger graphs or full fine-tuning

## Data Processing

### Real Graph Loading

The script automatically:
1. Loads real graphs from edge list format
2. Creates subgraphs of 30-50 nodes for training
3. Generates synthetic training labels (replace with real objectives)
4. Handles node masking for covered nodes

### Graph Batching

```python
# Create training batches from large real graph
graph_batches = finetuner.create_graph_batches(real_graph, num_batches=50)

# Each batch contains subgraphs of 30-50 nodes
for batch in graph_batches:
    print(f"Subgraph: {batch.num_nodes} nodes, {batch.num_edges} edges")
```

### Node Masking

The script uses the same node masking approach as the original FINDER_ND:
- Covered nodes are masked out using `idx_map`
- Sparse matrices are created for efficient computation
- No graph modification, only masking

## Training Process

### 1. Model Initialization
```python
# Build encoder and decoder
encoder = GraphEncoder(embeddingMethod='GIN', ...)
decoder = MLPDecoder(embedding_size=64, ...)

# Load pre-trained weights
saver.restore(session, pretrained_ckpt_path)
```

### 2. Data Generation
```python
# Generate training examples with synthetic labels
training_data = finetuner.generate_training_data(graph_batches)

# Each example contains:
# - Graph embeddings
# - Node embeddings  
# - Action selections
# - Synthetic labels (replace with real objectives)
```

### 3. Training Loop
```python
for epoch in range(num_epochs):
    for batch in training_data:
        # Forward pass
        predictions = model(batch)
        
        # Calculate loss
        loss = mean_squared_error(predictions, batch['labels'])
        
        # Backward pass (encoder frozen in freeze mode)
        optimizer.minimize(loss)
```

## Evaluation

### Model Evaluation
```python
# Evaluate on test graphs
test_graphs = finetuner.create_graph_batches(real_graph, num_batches=10)
avg_loss = finetuner.evaluate(test_graphs)
```

### Performance Comparison

Compare freezing vs full fine-tuning:

```python
# Test both modes
for mode in ['freeze', 'full']:
    config = get_config(f'digg_{mode}')
    finetuner = FineTuneGraphDQN(**config)
    finetuner.train()
    loss = finetuner.evaluate(test_graphs)
    print(f"{mode} mode - Loss: {loss:.6f}")
```

## Best Practices

### 1. Choose the Right Mode
- **Freezing**: Use for large graphs, limited resources, or when encoder is well-trained
- **Full**: Use for small graphs, significant domain shift, or maximum performance

### 2. Learning Rate Selection
- Start with lower learning rates for larger graphs
- Use even lower rates for full fine-tuning mode
- Monitor loss curves for convergence

### 3. Batch Size Optimization
- Larger batch sizes for smaller graphs
- Smaller batch sizes for memory-constrained scenarios
- Balance between memory usage and training stability

### 4. Data Preparation
- Replace synthetic labels with real objectives
- Ensure proper graph format (edge list)
- Validate graph connectivity

### 5. Monitoring
- Track training loss every 10 epochs
- Monitor for overfitting
- Save checkpoints regularly

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use freezing mode
2. **Slow Training**: Use freezing mode or reduce graph size
3. **Poor Convergence**: Adjust learning rate or increase epochs
4. **Checkpoint Loading**: Ensure pre-trained model exists in `./models/`

### Performance Tips

1. **GPU Usage**: Ensure TensorFlow GPU support is enabled
2. **Sparse Operations**: The script uses sparse matrices for efficiency
3. **Batch Processing**: Process multiple graphs simultaneously
4. **Early Stopping**: Implement early stopping for large graphs

## File Structure

```
├── FINDER_ND_FineTune.py          # Main fine-tuning script
├── finetune_config.py             # Configuration management
├── FINDER_ND_FineTune_README.md   # This documentation
├── AAA-NetDQN/code/FINDER_ND/     # Original FINDER_ND code
│   ├── graph_dqn_modules.py       # Encoder and decoder
│   ├── PrepareBatchGraph.pyx      # Graph processing
│   └── GraphDQN.pyx              # Main GraphDQN class
└── dataset/real/                  # Real graph datasets
    ├── Digg.txt
    ├── Crime.txt
    └── ...
```

## Example Workflow

```python
# 1. Load configuration
from finetune_config import get_config
config = get_config('digg_freeze')

# 2. Initialize fine-tuner
from FINDER_ND_FineTune import FineTuneGraphDQN
finetuner = FineTuneGraphDQN(**config)

# 3. Train model
finetuner.train()

# 4. Evaluate performance
real_graph = finetuner.load_real_graph()
test_graphs = finetuner.create_graph_batches(real_graph, num_batches=10)
loss = finetuner.evaluate(test_graphs)

print(f"Final evaluation loss: {loss:.6f}")
```

This fine-tuning framework provides a flexible way to adapt the pre-trained GraphDQN model to various real-world graph datasets while maintaining the efficiency and effectiveness of the original FINDER_ND approach. 