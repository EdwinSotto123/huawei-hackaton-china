"""
Training Configuration
======================
Hyperparameters and settings for ISLR model training.
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_dim: int = 708
    num_classes: int = 250
    dim: int = 192
    kernel_size: int = 17
    num_heads: int = 4
    expand_ratio: int = 4
    ffn_expand: int = 2
    drop_rate: float = 0.2
    attn_dropout: float = 0.2


@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str = "dataset/processed/train"
    val_path: str = "dataset/processed/val"
    max_seq_len: int = 384
    batch_size: int = 64
    num_workers: int = 4
    
    # Augmentation
    use_augmentation: bool = True
    time_mask_ratio: float = 0.1
    spatial_noise: float = 0.02


@dataclass
class TrainConfig:
    """Training configuration."""
    # Training
    epochs: int = 100
    learning_rate: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # Scheduler
    scheduler: str = "cosine"  # cosine, step, constant
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Checkpointing
    save_dir: str = "outputs/checkpoints"
    save_top_k: int = 3
    
    # Logging
    log_dir: str = "outputs/logs"
    log_interval: int = 100  # Log every N batches
    
    # Mixed precision
    use_amp: bool = True
    
    # Device
    device_target: str = "GPU"  # GPU, CPU, Ascend
    device_id: int = 0


@dataclass
class Config:
    """Complete configuration."""
    model: ModelConfig = None
    data: DataConfig = None
    train: TrainConfig = None
    
    # Experiment
    experiment_name: str = "islr_v1"
    seed: int = 42
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.train is None:
            self.train = TrainConfig()


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(path: str) -> Config:
    """Load configuration from YAML file."""
    import yaml
    
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    config = Config()
    
    if 'model' in cfg_dict:
        config.model = ModelConfig(**cfg_dict['model'])
    if 'data' in cfg_dict:
        config.data = DataConfig(**cfg_dict['data'])
    if 'train' in cfg_dict:
        config.train = TrainConfig(**cfg_dict['train'])
    if 'experiment_name' in cfg_dict:
        config.experiment_name = cfg_dict['experiment_name']
    if 'seed' in cfg_dict:
        config.seed = cfg_dict['seed']
    
    return config


def save_config(config: Config, path: str):
    """Save configuration to YAML file."""
    import yaml
    from dataclasses import asdict
    
    cfg_dict = {
        'experiment_name': config.experiment_name,
        'seed': config.seed,
        'model': asdict(config.model),
        'data': asdict(config.data),
        'train': asdict(config.train)
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


if __name__ == "__main__":
    # Print default config
    config = get_default_config()
    print("Default Configuration:")
    print("=" * 60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Seed: {config.seed}")
    print("\nModel:")
    print(f"  Input dim: {config.model.input_dim}")
    print(f"  Classes: {config.model.num_classes}")
    print(f"  Hidden dim: {config.model.dim}")
    print("\nData:")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Max seq len: {config.data.max_seq_len}")
    print("\nTraining:")
    print(f"  Epochs: {config.train.epochs}")
    print(f"  Learning rate: {config.train.learning_rate}")
    print(f"  Device: {config.train.device_target}")
