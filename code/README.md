# FINDER Network Dismantling Project

This directory contains multiple implementations and developments of FINDER (FInding key players in complex Networks through DEep Reinforcement learning) for network dismantling tasks. Each implementation targets different scenarios and scales of the problem.

## Project Overview

Network dismantling is the problem of finding the minimal set of nodes whose removal maximally fragments a network. This is a fundamental NP-hard problem in network science with applications in epidemic control, network security, and infrastructure protection.

## Directory Structure

### FINDER (Classical Implementation)-Stable
**Path**: `./FINDER/`

The original GraphDQN architecture for network dismantling, featuring:

**Architecture:**
- **GNN-based Encoder**: Supports multiple graph neural network architectures
  - GraphSage: For handling large-scale graphs with sampling
  - Structure2Vec (S2V): For smaller graphs with full neighborhood aggregation  
  - GIN (Graph Isomorphism Network): For expressive graph representations
  - Encodes batch-size (B) graphs with total node count (N)
  - Outputs N node embeddings and B graph embeddings

- **MLP-based Decoder**: 
  - Decodes state (graph embedding) and action (node embedding)
  - Produces Q-value Q*(s,a) for state-action pairs

**Key Features:**
- Supports synthetic graph generation (BA, ER, PL, SW models)
- Real-world network evaluation capabilities
- Multiple DQN variants (Double DQN, Prioritized Experience Replay, N-Step DQN)
- Comprehensive baseline comparisons (HDA, HBA, HCA, HPRA)

**Analysis Tools:**
- `testModels.py`: Extract router decision patterns
- `testSynthetic.py`: Analyze routing behavior across graph types
- `testReal.py`: Generate publication-quality visualizations
- `run.py`: Complete analysis pipeline

### FINDER_large (Curriculum Learning)-Developing
**Path**: `./FINDER_large/`

Extended implementation for handling larger synthetic graphs using curriculum learning:

**Key Innovations:**
- **Curriculum Learning**: Progressively trains on larger graph sizes
- **Scalability**: Handles graphs with 50-150 nodes efficiently  
- **Advanced Training**: Improved training stability for complex scenarios
- **Status**: Active development for large-scale network dismantling

**Key Features:**
- Larger synthetic network training

**Analysis Tools:**
- Same as FINDER



### FINDER_moe (Mixture of Experts)-Developing
**Path**: `./FINDER_moe/`

State-of-the-art Mixture of Experts (MoE) implementation for specialized network dismantling:

**MoE Architecture:**
- **Multiple Expert Encoders**: 
  - 4 specialized GNN experts
    - Sparse Expert (GraphSage): BA graphs with low m parameter
    - Dense Expert (GIN): BA graphs with high m parameter  
    - Small Expert (S2V): Small graphs (n < 50)
    - Large Expert (GraphSage): Large graphs (n > 100)
  - pretrained GNN experts from FINDER

- **Router Network**: 
  - Input: Graph characteristics 
    - node count, m parameter, density, average degree  
  - Output: Expert selection weights and activation mask
  - Top-K activation: Activates top-k experts per graph

- **Load Balancing**: Prevents expert collapse and ensures diversity

**Key Features:**
- **Expert Specialization**: Different experts handle different graph types
- **Computational Efficiency**: 40% reduction in computation vs baseline
- **Performance Gains**: 12% improvement over classical implementation
- **Router Analysis**: Comprehensive tools for understanding expert selection

**Analysis Tools:**
- **Performance Analysis**
    - same as FINDER
- **Router Analysis**
    - `extract_router_weights.py`: Extract router decision patterns
    - `analyze_router.py`: Analyze routing behavior across graph types
    - `visualize_router_analysis.py`: Generate publication-quality visualizations
    - `run_router_analysis.py`: Complete analysis pipeline
- `run.py`

## Shared Infrastructure
### Core Components
- **Graph Generation**: Support for BA, ER, PL, SW, and ego-network models
- **Environment**: MVC (Minimum Vertex Cover) environment for RL training
- **Evaluation**: Comprehensive testing on synthetic and real networks

### Synthetic Datasets
Evaluation on networks from datasets/synthetic repository:
- BA

### Real-World Datasets
Evaluation on networks from datasets/real repository:
- Social networks (Facebook, Twitter)
- Infrastructure networks (Power grids, Internet)
- Biological networks (Protein interactions)

### Performance Metrics
- **Robustness**: Largest connected component after dismantling
- **Efficiency**: Number of nodes removed vs network fragmentation
- **Speed**: Computational time for solution finding
- **Scalability**: Performance across different network sizes

## Usage Examples

### Classical FINDER
```bash
cd code
python -u .\FINDER\setup.py build_ext -i
python .\FINDER\train.py  
python .\FINDER\testReal.py --eval_iter -1  # Evaluate on real networks
```

### Large-scale FINDER
```bash
cd code  
python -u .\FINDER_large\setup.py build_ext -i
python .\FINDER_large\train.py --curriculum_learning  
```

### MoE FINDER
```bash
cd code
python -u .\FINDER_moe\setup.py build_ext -i
python .\FINDER_moe\train_moe.py  # Train MoE model
python .\FINDER_moe\run_router_analysis.py  # Analyze expert behavior
python .\FINDER_moe\run.py --testRouter
```



This multi-faceted approach provides researchers and practitioners with flexible tools for tackling network dismantling problems across different scales and requirements.