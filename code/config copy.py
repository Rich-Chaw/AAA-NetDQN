
from matplotlib.colors import BASE_COLORS
from networkx.classes import graph


BA_CONFIG= {
    "g_type": "barabasi_albert",
    "g_params" : {'num_min': 30,
                  'num_max' : 50,
                  'm':1},
    "num_min" : 30,
    "num_max" : 50
}

GIN_CONFIG = {
    "gnn_model": "GIN",
    "gnn_layers": 3
}

GRAPHSAGE_CONFIG = {
    "gnn_model": "graphSage",
    "gnn_layers": 3
}

path_config = {
        "model_dir": "./models",
        "dataset_dir": "../../dataset",
        "result_dir": "result/temp"
}

train_config = {
        'feature_size': 2,
        'reg_hidden': 64,
        'aux_dim': 4,
        "embedding_size": 64,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "max_iteration": 1000000,
        "memory_size": 500000,
        "update_time": 1000
}

SYNTHETIC_CONFIG = {
    "datasets": ["30-50","50-100"]
}
REAL_CONFIG = {
    "datasets":
}

eval_config = {
        "eval_mode":"synthetic",
        "datasets": ["30-50","50-100"],
        "step_ratio": 0.01,
        "strategy_id": 0,
        "min_iter":0,
        "max_iter": 600,
        "iter_step": 300
    }

random_eval_config:{
        "random_remove_ratios": [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
        "repeat_times": 100
    }



# "eval_BA_GIN_on_REAL"
# "train_BA_graphSage"
def load_config(mode ):


def get_config(config_name):
    """Get configuration by name"""
    if config_name in ALL_CONFIGS:
        return ALL_CONFIGS[config_name]
    else:
        raise ValueError(f"Configuration '{config_name}' not found. Available: {list(ALL_CONFIGS.keys())}")


