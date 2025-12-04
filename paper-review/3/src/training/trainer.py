"""Training loop for translation models."""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import math
import time

from src.utils.checkpointing import save_checkpoint
from src.utils.metrics import compute_bleu
from src.inference.translator import Translator
from src.utils.csv_logger import CSVLogger


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

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_grad_norm = 0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training"):
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

            # Forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_input_mask, cross_input_mask)

            # Reshape for loss computation
            # logits: [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
            # targets: [batch, seq_len] -> [batch * seq_len]
            logits = logits.contiguous().view(-1, logits.size(-1))
            targets = tgt_output.contiguous().view(-1)

            # Compute loss (which is KL divergence with label smoothing)
            loss = self.criterion(logits, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping and tracking
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # Update weights and learning rate
            self.optimizer.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm.item()
            num_batches += 1
            self.global_step += 1

        avg_loss = total_loss / num_batches
        avg_grad_norm = total_grad_norm / num_batches
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
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    is_best_loss = True
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

        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
