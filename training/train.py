"""
Training Script
===============
Training script for Squeezeformer sign language recognition model.

Usage:
    python train.py [--config config.yaml]
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import mindspore
import mindspore.numpy as mnp
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import context, Tensor, nn
from mindspore.train import Model
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.nn.optim import Adam, AdamWeightDecay
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ISLRModelV2, create_islr_model_v2
from training.config import Config, get_default_config, load_config, save_config


class TrainingMetrics(Callback):
    """Custom callback for logging training metrics."""
    
    def __init__(self, log_interval: int = 100, log_file: str = None):
        super().__init__()
        self.log_interval = log_interval
        self.log_file = log_file
        self.step = 0
        self.epoch_loss = 0.0
        self.epoch_steps = 0
        self.start_time = None
    
    def epoch_begin(self, run_context):
        self.epoch_loss = 0.0
        self.epoch_steps = 0
        self.start_time = time.time()
    
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        
        self.step += 1
        self.epoch_steps += 1
        self.epoch_loss += float(loss)
        
        if self.step % self.log_interval == 0:
            avg_loss = self.epoch_loss / self.epoch_steps
            print(f"  Step {self.step}: loss={avg_loss:.4f}")
    
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        elapsed = time.time() - self.start_time
        
        avg_loss = self.epoch_loss / max(1, self.epoch_steps)
        
        msg = f"Epoch {epoch} completed in {elapsed:.1f}s | Avg Loss: {avg_loss:.4f}"
        print(msg)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch},{avg_loss},{elapsed}\n")


class EarlyStopping(Callback):
    """Early stopping callback."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
    
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping at epoch {cb_params.cur_epoch_num}")
                run_context.request_stop()


def create_dummy_dataset(config: Config, num_samples: int = 1000):
    """Create dummy dataset for testing using MindSpore ops."""
    mindspore.set_seed(config.seed)
    
    # Generate random features using MindSpore ops
    features_tensor = ops.StandardNormal()((
        num_samples, 
        config.data.max_seq_len, 
        config.model.input_dim
    ))
    features = features_tensor.asnumpy().astype('float32')
    
    # Generate random labels
    labels_tensor = ops.UniformInt(minval=0, maxval=config.model.num_classes)((num_samples,))
    labels = labels_tensor.asnumpy().astype('int32')
    
    # Create dataset
    dataset = ds.NumpySlicesDataset(
        {"features": features, "labels": labels},
        shuffle=True
    )
    
    dataset = dataset.batch(config.data.batch_size, drop_remainder=True)
    
    return dataset


def train(config: Config):
    """Main training function using Squeezeformer model."""
    
    print("=" * 60)
    print("MindSpore Sign Language Recognition Training")
    print("Model: Squeezeformer (V2)")
    print("=" * 60)
    
    # Set context
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config.train.device_target,
        device_id=config.train.device_id
    )
    
    # Set seed
    mindspore.set_seed(config.seed)
    
    # Create directories
    Path(config.train.save_dir).mkdir(parents=True, exist_ok=True)
    Path(config.train.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = Path(config.train.log_dir) / "config.yaml"
    save_config(config, str(config_path))
    print(f"Config saved: {config_path}")
    
    # Create Squeezeformer model
    print("\nCreating Squeezeformer model...")
    model = create_islr_model_v2(
        num_classes=config.model.num_classes,
        variant='base',
        input_dim=config.model.input_dim,
        dim=config.model.dim,
        dropout=config.model.drop_rate
    )
    
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Create dataset
    print("\nCreating dataset...")
    train_dataset = create_dummy_dataset(config, num_samples=5000)
    val_dataset = create_dummy_dataset(config, num_samples=500)
    
    print(f"Train batches: {train_dataset.get_dataset_size()}")
    print(f"Val batches: {val_dataset.get_dataset_size()}")
    
    # Loss and optimizer
    loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    optimizer = AdamWeightDecay(
        params=model.trainable_params(),
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay
    )
    
    # Callbacks
    log_file = Path(config.train.log_dir) / "training_log.csv"
    with open(log_file, 'w') as f:
        f.write("epoch,loss,time\n")
    
    callbacks = [
        LossMonitor(per_print_times=config.train.log_interval),
        TrainingMetrics(log_interval=config.train.log_interval, log_file=str(log_file)),
        EarlyStopping(patience=config.train.patience),
    ]
    
    # Checkpoint
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=train_dataset.get_dataset_size(),
        keep_checkpoint_max=config.train.save_top_k
    )
    ckpt_cb = ModelCheckpoint(
        prefix="squeezeformer",
        directory=config.train.save_dir,
        config=ckpt_config
    )
    callbacks.append(ckpt_cb)
    
    # Create Model wrapper and train
    print("\nStarting training...")
    print("=" * 60)
    
    trainer = Model(model, loss_fn, optimizer, metrics={'acc': nn.Accuracy()})
    
    trainer.train(
        config.train.epochs,
        train_dataset,
        callbacks=callbacks
    )
    
    print("=" * 60)
    print("Training completed!")
    print(f"Checkpoints saved in: {config.train.save_dir}")
    print(f"Logs saved in: {config.train.log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Squeezeformer model")
    parser.add_argument("--config", "-c", type=str, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--device", type=str, choices=["GPU", "CPU", "Ascend"], 
                        help="Device target")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override from command line
    if args.epochs:
        config.train.epochs = args.epochs
    if args.lr:
        config.train.learning_rate = args.lr
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.device:
        config.train.device_target = args.device
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
