  📚 Usage Instructions

  1. Training Script 🏋️

  The training script now automatically computes BLEU +
   chrF++ and logs them to CSV.

  Start New Training:

  /home/arnold/venv/bin/python scripts/train.py

  Resume from Checkpoint (recommended - resume from 
  epoch 15):

  /home/arnold/venv/bin/python scripts/train.py
  --resume checkpoints/best_model.pt

  Training with Small Dataset (for testing):

  /home/arnold/venv/bin/python scripts/train.py --small

  What You'll See During Training:

  Epoch 1/30
    Train Loss: 3.6386 | Train PPL: 38.04
    Val Loss:   3.1242 | Val PPL:   22.74
    Learning Rate: 0.001085
    BLEU Score: 11.16
    chrF++ Score: 35.42        ← NEW!

    Translation Examples:
      [1] Source:     안녕하세요
          Reference:  Hello
          Prediction: Hello

  Output Files:

  - logs/training_log_TIMESTAMP.csv: All metrics logged
   (now includes val_chrf column)
  - checkpoints/best_model.pt: Best model by validation
   loss
  - checkpoints/best_bleu_model.pt: Best model by BLEU
  score
  - checkpoints/checkpoint_epoch_N.pt: Periodic
  checkpoints

  ---
  2. Evaluation Script 📊

  Comprehensive evaluation on the full test set with
  all metrics, error analysis, and reports.

  Basic Evaluation (uses best model, beam search):

  /home/arnold/venv/bin/python scripts/evaluate.py

  Evaluate Specific Checkpoint:

  /home/arnold/venv/bin/python scripts/evaluate.py
  --checkpoint checkpoints/best_bleu_model.pt

  Use Greedy Search (faster):

  /home/arnold/venv/bin/python scripts/evaluate.py
  --method greedy

  Quick Test (first 100 samples):

  /home/arnold/venv/bin/python scripts/evaluate.py
  --max-samples 100

  With Advanced Metrics (COMET, BERTScore - requires 
  packages):

  # First install packages
  pip install unbabel-comet bert-score

  # Then run with advanced metrics
  /home/arnold/venv/bin/python scripts/evaluate.py
  --use-advanced-metrics

  Custom Output Directory:

  /home/arnold/venv/bin/python scripts/evaluate.py
  --output-dir my_evaluation

  All Options:

  /home/arnold/venv/bin/python scripts/evaluate.py \
      --checkpoint checkpoints/best_model.pt \
      --method beam \
      --max-samples 1000 \
      --output-dir outputs/my_eval \
      --device cuda \
      --use-advanced-metrics

  ---
  3. What the Evaluation Script Generates 📁

  After running, you'll get a directory like
  outputs/evaluation_20251212_123456/ containing:

  outputs/evaluation_20251212_123456/
  ├── EVALUATION_REPORT.md          # Comprehensive 
  markdown report
  ├── translations.txt               # All translations
   (source/pred/ref)
  ├── metrics.json                   # All metrics in 
  JSON
  ├── error_analysis.json            # Error statistics
  └── error_examples.txt             # Examples of each
   error type

  EVALUATION_REPORT.md - Main Report:

  # Evaluation Report

  **Date**: 2025-12-12 12:34:56
  **Model**: checkpoints/best_model.pt
  **Test Samples**: 4061
  **Inference Method**: beam

  ## Metrics

  | Metric | Score |
  |--------|-------|
  | BLEU | 27.78 |
  | BLEU_1 | 58.42 |
  | BLEU_2 | 37.15 |
  | BLEU_3 | 25.89 |
  | BLEU_4 | 19.23 |
  | CHRF | 56.34 |

  ## Error Analysis

  | Error Type | Count | Rate |
  |------------|-------|------|
  | Repetition Errors | 42 | 1.03% |
  | Number Mismatches | 123 | 3.03% |
  | Unknown Tokens | 8 | 0.20% |

  translations.txt - All Translations:

  Source:     안녕하세요, 만나서 반갑습니다.
  Prediction: Hello, nice to meet you.
  Reference:  Hello, pleased to meet you.
  -----------------------------------------------------
  ---------------------------
  Source:     오늘 날씨가 정말 좋네요.
  Prediction: The weather is really nice today.
  Reference:  The weather is really good today.
  -----------------------------------------------------
  ---------------------------
  ...

  error_examples.txt - Error Examples:

  REPETITION ERRORS
  =====================================================
  ===========================

  Example 1:
    Source:     저는 학생입니다.
    Hypothesis: I am a student student.
    Reference:  I am a student.

  NUMBER MISMATCH ERRORS
  =====================================================
  ===========================

  Example 1:
    Source:     나는 5개의 사과를 샀다.
    Source Numbers:     ['5']
    Hypothesis: I bought 3 apples.
    Hypothesis Numbers: ['3']
    Reference:  I bought 5 apples.

  ---
  📊 What's New in Training Logs

  The CSV log now includes a new column:

  | Column   | Description           |
  |----------|-----------------------|
  | val_bleu | BLEU score (existing) |
  | val_chrf | chrF++ score (NEW!)   |

  You can analyze the logs:
  # View latest training log
  cat logs/training_log_*.csv | column -t -s,

  # Or open in Excel/spreadsheet software

  ---
  🔧 Troubleshooting

  If evaluation crashes due to missing packages:

  # For basic evaluation (BLEU, chrF++)
  pip install sacrebleu

  # For advanced metrics
  pip install unbabel-comet bert-score

  If you run out of memory during evaluation:

  # Use greedy search instead of beam
  /home/arnold/venv/bin/python scripts/evaluate.py
  --method greedy

  # Or evaluate in batches
  /home/arnold/venv/bin/python scripts/evaluate.py
  --max-samples 1000

  If training still has NaN:

  The trainer now automatically:
  - ✅ Detects NaN in loss/gradients
  - ✅ Skips invalid batches
  - ✅ Prints warnings
  - ✅ Continues training

  You'll see warnings like:
  ⚠️  NaN LOSS DETECTED (Epoch 16, Batch 123/6634)
  Skipping this update to prevent model corruption

  This is expected behavior - the model will continue
  training safely.

  ---
  🚀 Recommended Workflow

  Full Training Run:

  # 1. Start training (or resume)
  /home/arnold/venv/bin/python scripts/train.py
  --resume checkpoints/best_model.pt

  # 2. Monitor logs during training
  tail -f logs/training_log_*.csv

  # 3. After training, run full evaluation
  /home/arnold/venv/bin/python scripts/evaluate.py
  --checkpoint checkpoints/best_bleu_model.pt

  # 4. Check the report
  cat outputs/evaluation_*/EVALUATION_REPORT.md

  Quick Test Run:

  # Quick evaluation on subset
  /home/arnold/venv/bin/python scripts/evaluate.py \
      --max-samples 500 \
      --method greedy \
      --output-dir outputs/quick_test