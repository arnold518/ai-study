"""Base configuration for all models."""

class BaseConfig:
    """Shared configuration across all models."""

    # Languages
    src_lang = "ko"
    tgt_lang = "en"

    # Data Sources (for download_data.py)
    datasets_to_download = ["moo", "tatoeba", "aihub"]  # Options: moo, tatoeba, aihub, all

    # Data Preprocessing (for split_data.py)
    min_length_chars = 1    # Minimum sentence length (characters, for initial check)
    length_ratio = 3.5      # Max length ratio between source and target (relaxed from 2.0)

    # Vocabulary (for train_tokenizer.py)
    use_shared_vocab = False  # False: separate vocabs for Korean/English (better for distant languages)
    vocab_size = 16000        # Size per language (or total if shared)
    character_coverage = 0.9995  # For SentencePiece (0.9995 for Korean, 1.0 for English)
    spm_model_type = "unigram"  # Options: unigram, bpe, char, word

    # Training Data
    max_seq_length = 150  # Maximum sequence length (TOKENS) - increased for longer Korean sentences
    min_freq = 2

    # Training
    batch_size = 128
    gradient_accumulation_steps = 2  # Effective batch size = 128 * 2 = 256
    num_epochs = 50        # Increased from 30 - with better regularization, can train longer
    learning_rate = 1e-4   # Not used (Transformer uses Noam scheduler)
    grad_clip = 1.0

    # Regularization
    dropout = 0.3          # Increased from 0.1 - prevent overfitting on 897k dataset
    label_smoothing = 0.05 # Reduced from 0.1 - less aggressive smoothing

    # Checkpointing
    save_every = 5
    eval_every = 1
    max_checkpoint_size_gb = 20.0  # Maximum total size of checkpoint directory (GB)
    keep_n_recent_checkpoints = 3  # Number of periodic checkpoints to keep

    # Early Stopping (overfitting prevention)
    early_stopping_patience = 10     # Increased from 5 - with strong regularization, can be more patient
    early_stopping_min_delta = 0.001 # Increased from 0.0001 - require larger improvement

    # Evaluation
    bleu_num_samples = 100  # Number of validation samples for BLEU computation
    inference_num_examples = 2  # Number of inference examples to display

    # Device
    device = "cuda"
    num_workers = 8

    # Mixed Precision Training (Automatic Mixed Precision)
    use_mixed_precision = True  # Enable AMP for 2-3x speedup and 40% memory savings

    # Gradient Monitoring (for debugging training issues)
    monitor_gradients = False  # Enable detailed gradient statistics and anomaly detection

    # Paths
    data_dir = "data"
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    vocab_dir = "data/vocab"
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    output_dir = "outputs"
