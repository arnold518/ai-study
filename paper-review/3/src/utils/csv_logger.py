"""CSV logger for training metrics and configuration."""

import csv
import os
from datetime import datetime
from pathlib import Path


class CSVLogger:
    """Logger for writing training metrics to CSV file."""

    def __init__(self, log_path, config):
        """
        Initialize CSV logger.

        Args:
            log_path: Path to CSV file
            config: Training configuration object
        """
        self.log_path = Path(log_path)
        self.config = config
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists to determine if we need to write header
        self.file_exists = self.log_path.exists()

        # Define all columns we want to log
        self.columns = self._get_columns()

        # Create/open file and write header if needed
        if not self.file_exists:
            self._write_header()

    def _get_columns(self):
        """Define all columns for the CSV file."""
        columns = [
            # Timestamp and identification
            'timestamp',
            'epoch',
            'global_step',

            # Training metrics
            'train_loss',
            'train_ppl',
            'train_kl_div',

            # Validation metrics
            'val_loss',
            'val_ppl',
            'val_kl_div',
            'val_bleu',

            # Learning rate and optimization
            'learning_rate',
            'grad_norm',

            # Best model tracking
            'best_train_loss',
            'best_val_loss',
            'best_bleu',
            'is_best_loss',
            'is_best_bleu',

            # Checkpoint information
            'checkpoint_path',
            'checkpoint_type',  # 'periodic', 'best_loss', 'best_bleu', or ''

            # Model architecture (from config)
            'd_model',
            'num_heads',
            'num_encoder_layers',
            'num_decoder_layers',
            'd_ff',
            'dropout',

            # Training hyperparameters
            'batch_size',
            'max_seq_length',
            'learning_rate_factor',
            'warmup_steps',
            'label_smoothing',
            'grad_clip',

            # Vocabulary
            'src_vocab_size',
            'tgt_vocab_size',

            # Evaluation settings
            'bleu_num_samples',
            'inference_num_examples',

            # Dataset sizes (if available)
            'train_size',
            'val_size',

            # Timing
            'epoch_time_seconds',
            'cumulative_time_seconds',
        ]

        return columns

    def _write_header(self):
        """Write CSV header."""
        with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

    def log(self, metrics):
        """
        Log metrics to CSV file.

        Args:
            metrics: Dictionary of metrics to log
        """
        # Add timestamp
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Add config parameters if not already present
        config_params = self._get_config_params()
        for key, value in config_params.items():
            if key not in metrics:
                metrics[key] = value

        # Ensure all columns exist (fill missing with empty string)
        row = {col: metrics.get(col, '') for col in self.columns}

        # Write to CSV
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)

    def _get_config_params(self):
        """Extract relevant parameters from config."""
        params = {}

        # Model architecture
        if hasattr(self.config, 'd_model'):
            params['d_model'] = self.config.d_model
        if hasattr(self.config, 'num_heads'):
            params['num_heads'] = self.config.num_heads
        if hasattr(self.config, 'num_encoder_layers'):
            params['num_encoder_layers'] = self.config.num_encoder_layers
        if hasattr(self.config, 'num_decoder_layers'):
            params['num_decoder_layers'] = self.config.num_decoder_layers
        if hasattr(self.config, 'd_ff'):
            params['d_ff'] = self.config.d_ff
        if hasattr(self.config, 'dropout'):
            params['dropout'] = self.config.dropout

        # Training hyperparameters
        if hasattr(self.config, 'batch_size'):
            params['batch_size'] = self.config.batch_size
        if hasattr(self.config, 'max_seq_length'):
            params['max_seq_length'] = self.config.max_seq_length
        if hasattr(self.config, 'learning_rate'):
            params['learning_rate_factor'] = self.config.learning_rate
        if hasattr(self.config, 'warmup_steps'):
            params['warmup_steps'] = self.config.warmup_steps
        if hasattr(self.config, 'label_smoothing'):
            params['label_smoothing'] = self.config.label_smoothing
        if hasattr(self.config, 'grad_clip'):
            params['grad_clip'] = self.config.grad_clip

        # Evaluation settings
        if hasattr(self.config, 'bleu_num_samples'):
            params['bleu_num_samples'] = self.config.bleu_num_samples
        if hasattr(self.config, 'inference_num_examples'):
            params['inference_num_examples'] = self.config.inference_num_examples

        return params

    def log_epoch(self, epoch, metrics):
        """
        Convenience method to log an epoch with standard metrics.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        metrics['epoch'] = epoch
        self.log(metrics)
