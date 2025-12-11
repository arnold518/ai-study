"""Training loop for translation models."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import math
import time
from collections import defaultdict

from src.utils.checkpointing import save_checkpoint
from src.utils.metrics import compute_bleu
from src.inference.translator import Translator
from src.utils.csv_logger import CSVLogger
from src.utils.checkpoint_cleanup import cleanup_checkpoints, should_cleanup


class Trainer:
    """Generic trainer for translation models."""

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, config,
                 src_tokenizer=None, tgt_tokenizer=None, val_dataset=None):
        """
        Args:
            model: Translation model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            config: Training configuration
            src_tokenizer: Source tokenizer (for BLEU computation)
            tgt_tokenizer: Target tokenizer (for BLEU computation)
            val_dataset: Validation dataset (for getting original texts)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = config.device
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.best_bleu = 0.0
        self.current_epoch = 0
        self.global_step = 0

        # Early stopping
        self.early_stopping_patience = getattr(config, 'early_stopping_patience', 8)
        self.early_stopping_min_delta = getattr(config, 'early_stopping_min_delta', 0.0001)
        self.epochs_without_improvement = 0

        # For BLEU computation and inference examples
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.val_dataset = val_dataset

        # Vocabulary sizes
        self.src_vocab_size = src_tokenizer.vocab_size if src_tokenizer else None
        self.tgt_vocab_size = tgt_tokenizer.vocab_size if tgt_tokenizer else None

        # Dataset sizes
        self.train_size = len(train_loader.dataset) if train_loader else None
        self.val_size = len(val_loader.dataset) if val_loader else None

        # Timing
        self.cumulative_time = 0.0

        # Create translator if tokenizers available
        self.translator = None
        if src_tokenizer and tgt_tokenizer:
            self.translator = Translator(
                model=model,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=self.device,
                max_length=config.max_seq_length
            )

        # Initialize CSV logger
        log_dir = getattr(config, 'log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(log_dir, f'training_log_{timestamp}.csv')
        self.csv_logger = CSVLogger(csv_path, config)
        print(f"CSV logging enabled: {csv_path}")

        # Mixed Precision Training
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', False)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        if self.use_mixed_precision:
            print("✓ Mixed precision training enabled (AMP)")

        # Gradient Monitoring
        self.monitor_gradients = getattr(config, 'monitor_gradients', False)
        self.gradient_stats = {
            'layer_norms': defaultdict(list),
            'anomalies': {'nan': 0, 'inf': 0, 'vanishing': 0, 'exploding': 0}
        }
        if self.monitor_gradients:
            print("✓ Gradient monitoring enabled")

    def compute_gradient_stats(self):
        """
        Compute detailed gradient statistics for monitoring.

        Returns:
            stats: Dictionary with gradient statistics
        """
        total_norm = 0.0
        layer_norms = {}
        param_stats = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Compute norm for this parameter
                param_norm = param.grad.data.norm(2).item()
                layer_norms[name] = param_norm
                total_norm += param_norm ** 2

                # Compute statistics
                grad_data = param.grad.data
                param_stats.append({
                    'name': name,
                    'norm': param_norm,
                    'mean': grad_data.mean().item(),
                    'std': grad_data.std().item(),
                    'min': grad_data.min().item(),
                    'max': grad_data.max().item(),
                    'has_nan': torch.isnan(grad_data).any().item(),
                    'has_inf': torch.isinf(grad_data).any().item()
                })

        total_norm = total_norm ** 0.5

        return {
            'total_norm': total_norm,
            'layer_norms': layer_norms,
            'param_stats': param_stats
        }

    def check_gradient_anomalies(self, grad_stats, vanishing_threshold=1e-6, exploding_threshold=100.0):
        """
        Check for gradient anomalies (NaN, Inf, vanishing, exploding).

        Args:
            grad_stats: Gradient statistics from compute_gradient_stats()
            vanishing_threshold: Threshold for vanishing gradients
            exploding_threshold: Threshold for exploding gradients

        Returns:
            anomalies: Dictionary with anomaly counts
        """
        anomalies = {
            'nan': 0,
            'inf': 0,
            'vanishing': 0,
            'exploding': 0,
            'details': []
        }

        for param_stat in grad_stats['param_stats']:
            # Check for NaN
            if param_stat['has_nan']:
                anomalies['nan'] += 1
                anomalies['details'].append(f"NaN in {param_stat['name']}")

            # Check for Inf
            if param_stat['has_inf']:
                anomalies['inf'] += 1
                anomalies['details'].append(f"Inf in {param_stat['name']}")

            # Check for vanishing gradients
            if param_stat['norm'] < vanishing_threshold:
                anomalies['vanishing'] += 1
                anomalies['details'].append(f"Vanishing gradient in {param_stat['name']} (norm={param_stat['norm']:.2e})")

            # Check for exploding gradients
            if param_stat['norm'] > exploding_threshold:
                anomalies['exploding'] += 1
                anomalies['details'].append(f"Exploding gradient in {param_stat['name']} (norm={param_stat['norm']:.2e})")

        return anomalies

    def print_gradient_stats(self, grad_stats, anomalies=None, top_n=5):
        """
        Print gradient statistics.

        Args:
            grad_stats: Gradient statistics from compute_gradient_stats()
            anomalies: Optional anomaly information
            top_n: Number of top layers to show
        """
        print(f"\n  Gradient Statistics:")
        print(f"    Total Norm: {grad_stats['total_norm']:.4f}")

        # Sort layers by norm
        sorted_layers = sorted(grad_stats['layer_norms'].items(), key=lambda x: x[1], reverse=True)

        print(f"    Top {top_n} layers by gradient norm:")
        for name, norm in sorted_layers[:top_n]:
            print(f"      {name}: {norm:.4f}")

        # Print anomalies if any
        if anomalies:
            total_anomalies = sum(anomalies[k] for k in ['nan', 'inf', 'vanishing', 'exploding'])
            if total_anomalies > 0:
                print(f"\n    ⚠️  Gradient Anomalies Detected:")
                if anomalies['nan'] > 0:
                    print(f"      NaN: {anomalies['nan']} parameters")
                if anomalies['inf'] > 0:
                    print(f"      Inf: {anomalies['inf']} parameters")
                if anomalies['vanishing'] > 0:
                    print(f"      Vanishing: {anomalies['vanishing']} parameters")
                if anomalies['exploding'] > 0:
                    print(f"      Exploding: {anomalies['exploding']} parameters")

                # Print details for critical anomalies
                critical = [d for d in anomalies['details'] if 'NaN' in d or 'Inf' in d]
                if critical:
                    print(f"\n    Critical Issues:")
                    for detail in critical[:3]:  # Show first 3
                        print(f"      - {detail}")

    def train_epoch(self):
        """Train for one epoch with gradient accumulation support."""
        self.model.train()
        total_loss = 0
        total_grad_norm = 0
        num_batches = 0

        # Get gradient accumulation steps from config
        accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)

        # Zero gradients at start
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            # Move batch to device
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            src_mask = batch['src_mask'].to(self.device)
            tgt_mask = batch['tgt_mask'].to(self.device)
            cross_mask = batch['cross_mask'].to(self.device)

            # Prepare inputs and targets for training
            # Input to decoder: [<bos>, token1, token2, ...]
            # Target for loss: [token1, token2, ..., <eos>]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Adjust masks for decoder input (remove last position)
            tgt_input_mask = tgt_mask[:, :, :-1, :-1]
            cross_input_mask = cross_mask[:, :, :-1, :]

            # Forward pass with mixed precision
            with autocast(enabled=self.use_mixed_precision):
                logits = self.model(src, tgt_input, src_mask, tgt_input_mask, cross_input_mask)

                # Reshape for loss computation
                # logits: [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
                # targets: [batch, seq_len] -> [batch * seq_len]
                logits = logits.contiguous().view(-1, logits.size(-1))
                targets = tgt_output.contiguous().view(-1)

                # Compute loss (which is KL divergence with label smoothing)
                loss = self.criterion(logits, targets)

                # Scale loss by accumulation steps (important for correct gradient magnitude)
                loss = loss / accumulation_steps

            # Backward pass (accumulate gradients) with gradient scaling
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.use_mixed_precision:
                    # Unscale gradients before clipping (required for AMP)
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping and tracking
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                total_grad_norm += grad_norm.item()

                # Detailed gradient monitoring (optional, every N steps)
                if self.monitor_gradients and self.global_step % 100 == 0:
                    grad_stats = self.compute_gradient_stats()
                    anomalies = self.check_gradient_anomalies(grad_stats)

                    # Track anomalies
                    for key in ['nan', 'inf', 'vanishing', 'exploding']:
                        self.gradient_stats['anomalies'][key] += anomalies[key]

                    # Print if anomalies detected
                    total_anomalies = sum(anomalies[k] for k in ['nan', 'inf', 'vanishing', 'exploding'])
                    if total_anomalies > 0:
                        print(f"\n  [Step {self.global_step}] Gradient anomalies detected:")
                        self.print_gradient_stats(grad_stats, anomalies, top_n=3)

                # Update weights and learning rate
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Zero gradients for next accumulation cycle
                self.optimizer.zero_grad()

                self.global_step += 1

            # Track loss (multiply back by accumulation_steps for correct reporting)
            total_loss += loss.item() * accumulation_steps
            num_batches += 1

        # Handle remaining gradients if batch count not divisible by accumulation_steps
        if num_batches % accumulation_steps != 0:
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            total_grad_norm += grad_norm.item()

            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.global_step += 1

        avg_loss = total_loss / num_batches
        # Divide by number of actual optimizer steps, not batches
        num_optimizer_steps = (num_batches + accumulation_steps - 1) // accumulation_steps
        avg_grad_norm = total_grad_norm / num_optimizer_steps if num_optimizer_steps > 0 else 0
        return avg_loss, avg_grad_norm

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                src_mask = batch['src_mask'].to(self.device)
                tgt_mask = batch['tgt_mask'].to(self.device)
                cross_mask = batch['cross_mask'].to(self.device)

                # Prepare inputs and targets
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_input_mask = tgt_mask[:, :, :-1, :-1]
                cross_input_mask = cross_mask[:, :, :-1, :]

                # Forward pass
                logits = self.model(src, tgt_input, src_mask, tgt_input_mask, cross_input_mask)

                # Reshape for loss computation
                logits = logits.contiguous().view(-1, logits.size(-1))
                targets = tgt_output.contiguous().view(-1)

                # Compute loss
                loss = self.criterion(logits, targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def compute_bleu_score(self, num_samples=None):
        """
        Compute BLEU score on a subset of validation data.

        Args:
            num_samples: Number of samples to use for BLEU computation.
                        If None, uses config.bleu_num_samples

        Returns:
            bleu_score: BLEU score (float)
        """
        if not self.translator or not self.val_dataset:
            return None

        self.model.eval()

        # Get a subset of validation data
        if num_samples is None:
            num_samples = self.config.bleu_num_samples
        num_samples = min(num_samples, len(self.val_dataset))
        indices = torch.randperm(len(self.val_dataset))[:num_samples]

        predictions = []
        references = []

        print(f"  Computing BLEU on {num_samples} samples...")

        with torch.no_grad():
            for idx in tqdm(indices, desc="  BLEU", leave=False):
                # Get source and target texts
                src_text = self.val_dataset.src_lines[idx].strip()
                tgt_text = self.val_dataset.tgt_lines[idx].strip()

                # Translate
                try:
                    pred_text = self.translator.translate(src_text, method='greedy')
                    predictions.append(pred_text)
                    references.append(tgt_text)
                except Exception as e:
                    # Skip if translation fails
                    continue

        if not predictions:
            return None

        # Compute BLEU
        bleu = compute_bleu(predictions, references)
        return bleu.score

    def generate_inference_examples(self, num_examples=None):
        """
        Generate inference examples from validation set.

        Args:
            num_examples: Number of examples to generate.
                         If None, uses config.inference_num_examples

        Returns:
            examples: List of (source, reference, prediction) tuples
        """
        if not self.translator or not self.val_dataset:
            return []

        self.model.eval()

        # Get random samples
        if num_examples is None:
            num_examples = self.config.inference_num_examples
        num_examples = min(num_examples, len(self.val_dataset))
        indices = torch.randperm(len(self.val_dataset))[:num_examples]

        examples = []

        with torch.no_grad():
            for idx in indices:
                src_text = self.val_dataset.src_lines[idx].strip()
                tgt_text = self.val_dataset.tgt_lines[idx].strip()

                try:
                    pred_text = self.translator.translate(src_text, method='greedy')
                    examples.append((src_text, tgt_text, pred_text))
                except Exception:
                    continue

        return examples

    def train(self):
        """Full training loop with checkpointing."""
        print(f"\nStarting training for {self.config.num_epochs} epochs")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            epoch_start_time = time.time()

            # Train
            train_loss, train_grad_norm = self.train_epoch()
            train_ppl = math.exp(min(train_loss, 100))  # Cap to avoid overflow

            # Track best training loss
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss

            # Validate (if it's time)
            if (epoch + 1) % self.config.eval_every == 0:
                val_loss = self.validate()
                val_ppl = math.exp(min(val_loss, 100))

                print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
                print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
                print(f"  Learning Rate: {self.optimizer._get_lr():.6f}")

                # Compute BLEU score
                bleu_score = None
                if self.translator:
                    bleu_score = self.compute_bleu_score()
                    if bleu_score is not None:
                        print(f"  BLEU Score: {bleu_score:.2f}")

                # Generate inference examples
                if self.translator:
                    examples = self.generate_inference_examples()
                    if examples:
                        print(f"\n  Translation Examples:")
                        for i, (src, ref, pred) in enumerate(examples, 1):
                            print(f"    [{i}] Source:     {src}")
                            print(f"        Reference:  {ref}")
                            print(f"        Prediction: {pred}")
                            print()

                # Track checkpoint information
                checkpoint_path = ""
                checkpoint_type = ""
                is_best_loss = False
                is_best_bleu = False

                # Save best model (based on validation loss)
                if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    is_best_loss = True
                    self.epochs_without_improvement = 0  # Reset counter
                    best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
                    save_checkpoint(
                        self.model,
                        self.optimizer.optimizer,  # Save underlying Adam optimizer
                        epoch + 1,
                        val_loss,
                        best_path
                    )
                    checkpoint_path = best_path
                    checkpoint_type = 'best_loss'
                    print(f"  -> New best model saved (Val Loss: {val_loss:.4f})!")
                else:
                    self.epochs_without_improvement += 1
                    print(f"  -> No improvement for {self.epochs_without_improvement} epoch(s) (patience: {self.early_stopping_patience})")

                # Also save best BLEU model
                if bleu_score is not None and bleu_score > self.best_bleu:
                    self.best_bleu = bleu_score
                    is_best_bleu = True
                    best_bleu_path = os.path.join(self.config.checkpoint_dir, 'best_bleu_model.pt')
                    save_checkpoint(
                        self.model,
                        self.optimizer.optimizer,
                        epoch + 1,
                        val_loss,
                        best_bleu_path
                    )
                    if checkpoint_type:
                        checkpoint_type += ',best_bleu'
                    else:
                        checkpoint_type = 'best_bleu'
                        checkpoint_path = best_bleu_path
                    print(f"  -> New best BLEU model saved (BLEU: {bleu_score:.2f})!")

                # Compute epoch time
                epoch_time = time.time() - epoch_start_time
                self.cumulative_time += epoch_time

                # Log to CSV
                metrics = {
                    'epoch': epoch + 1,
                    'global_step': self.global_step,
                    'train_loss': train_loss,
                    'train_ppl': train_ppl,
                    'train_kl_div': train_loss,  # Loss is KL divergence
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'val_kl_div': val_loss,  # Loss is KL divergence
                    'val_bleu': bleu_score if bleu_score is not None else '',
                    'learning_rate': self.optimizer._get_lr(),
                    'grad_norm': train_grad_norm,
                    'best_train_loss': self.best_train_loss,
                    'best_val_loss': self.best_val_loss,
                    'best_bleu': self.best_bleu,
                    'is_best_loss': is_best_loss,
                    'is_best_bleu': is_best_bleu,
                    'checkpoint_path': checkpoint_path,
                    'checkpoint_type': checkpoint_type,
                    'src_vocab_size': self.src_vocab_size,
                    'tgt_vocab_size': self.tgt_vocab_size,
                    'train_size': self.train_size,
                    'val_size': self.val_size,
                    'epoch_time_seconds': epoch_time,
                    'cumulative_time_seconds': self.cumulative_time,
                }
                self.csv_logger.log(metrics)

            else:
                # No validation this epoch
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
                print(f"  Learning Rate: {self.optimizer._get_lr():.6f}")

                # Compute epoch time
                epoch_time = time.time() - epoch_start_time
                self.cumulative_time += epoch_time

                # Log to CSV (without validation metrics)
                metrics = {
                    'epoch': epoch + 1,
                    'global_step': self.global_step,
                    'train_loss': train_loss,
                    'train_ppl': train_ppl,
                    'train_kl_div': train_loss,
                    'learning_rate': self.optimizer._get_lr(),
                    'grad_norm': train_grad_norm,
                    'best_train_loss': self.best_train_loss,
                    'best_val_loss': self.best_val_loss,
                    'best_bleu': self.best_bleu,
                    'src_vocab_size': self.src_vocab_size,
                    'tgt_vocab_size': self.tgt_vocab_size,
                    'train_size': self.train_size,
                    'val_size': self.val_size,
                    'epoch_time_seconds': epoch_time,
                    'cumulative_time_seconds': self.cumulative_time,
                }
                self.csv_logger.log(metrics)

            # Save checkpoint periodically
            if (epoch + 1) % self.config.save_every == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f'checkpoint_epoch_{epoch+1}.pt'
                )
                save_checkpoint(
                    self.model,
                    self.optimizer.optimizer,
                    epoch + 1,
                    train_loss,
                    checkpoint_path
                )
                print(f"  Checkpoint saved: {checkpoint_path}")

                # Cleanup old checkpoints if needed
                max_size = getattr(self.config, 'max_checkpoint_size_gb', None)
                keep_n = getattr(self.config, 'keep_n_recent_checkpoints', 3)

                if max_size is not None and should_cleanup(self.config.checkpoint_dir, max_size):
                    print(f"\n  Cleaning up checkpoints (directory > {max_size:.1f} GB)...")
                    cleanup_stats = cleanup_checkpoints(
                        self.config.checkpoint_dir,
                        max_size_gb=max_size,
                        keep_n_recent=keep_n,
                        dry_run=False,
                        verbose=True
                    )
                    if cleanup_stats['deleted_files']:
                        print(f"  ✓ Freed {cleanup_stats['freed_space_gb']:.2f} GB "
                              f"({cleanup_stats['freed_space_percent']:.1f}%)")
                        print(f"  ✓ Deleted {len(cleanup_stats['deleted_files'])} old checkpoint(s)")

                # Show inference examples for periodic checkpoints too
                if self.translator:
                    examples = self.generate_inference_examples()
                    if examples:
                        print(f"\n  Translation Examples:")
                        for i, (src, ref, pred) in enumerate(examples, 1):
                            print(f"    [{i}] Source:     {src}")
                            print(f"        Reference:  {ref}")
                            print(f"        Prediction: {pred}")
                            print()

            # Check early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n{'='*80}")
                print(f"EARLY STOPPING TRIGGERED")
                print(f"{'='*80}")
                print(f"No improvement in validation loss for {self.epochs_without_improvement} epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                print(f"Stopping training at epoch {epoch + 1}/{self.config.num_epochs}")
                print(f"{'='*80}\n")
                break

        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
