"""Base configuration for all models."""

class BaseConfig:
    """Shared configuration across all models."""

    # Languages
    src_lang = "ko"
    tgt_lang = "en"

    # Data Sources (for download_data.py)
    datasets_to_download = ["moo", "tatoeba", "aihub"]  # Options: moo, tatoeba, aihub, all

    # Data Preprocessing (for split_data.py)
    min_length = 1        # Minimum sentence length (words/tokens)
    max_length = 150      # Maximum sentence length
    length_ratio = 2.0    # Max length ratio between source and target (1.5 or 2.0)

    # Vocabulary (for train_tokenizer.py)
    use_shared_vocab = True  # True: shared vocab for src/tgt, False: separate vocabs
    vocab_size = 16000        # Size per language (or total if shared)
    character_coverage = 0.9995  # For SentencePiece (0.9995 for Korean, 1.0 for English)
    spm_model_type = "unigram"  # Options: unigram, bpe, char, word

    # Training Data
    max_seq_length = 128  # For model input (can be different from max_length)
    min_freq = 2

    # Training
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-4
    grad_clip = 1.0

    # Regularization
    dropout = 0.1
    label_smoothing = 0.1

    # Checkpointing
    save_every = 5
    eval_every = 1

    # Evaluation
    bleu_num_samples = 100  # Number of validation samples for BLEU computation
    inference_num_examples = 2  # Number of inference examples to display

    # Device
    device = "cpu"
    num_workers = 4

    # Paths
    data_dir = "data"
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    vocab_dir = "data/vocab"
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    output_dir = "outputs"
