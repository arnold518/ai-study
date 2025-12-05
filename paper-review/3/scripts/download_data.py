#!/usr/bin/env python
"""
Download Korean-English parallel corpus from multiple sources.

Supported datasets:
1. moo: Moo/korean-parallel-corpora (Hugging Face) - ~96k train, 1k val, 2k test
2. tatoeba: Helsinki-NLP/tatoeba_mt (Hugging Face) - Only test/val sets
3. aihub: AI Hub via Korpora - Requires manual download first

Downloads are configured in config/base_config.py (datasets_to_download parameter).

Usage:
    /home/arnold/venv/bin/python scripts/download_data.py
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.base_config import BaseConfig


def download_moo_dataset(raw_dir):
    """Download Moo/korean-parallel-corpora dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found.")
        print("Please install it: pip install 'datasets<3'")
        return False

    print("\n" + "=" * 60)
    print("Downloading: Moo/korean-parallel-corpora")
    print("=" * 60)

    output_dir = raw_dir / "moo"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading dataset from Hugging Face...")
        ds = load_dataset("Moo/korean-parallel-corpora")

        print("\nDataset loaded successfully!")
        print(f"  Train:      {len(ds['train']):,} sentence pairs")
        print(f"  Validation: {len(ds['validation']):,} sentence pairs")
        print(f"  Test:       {len(ds['test']):,} sentence pairs")
        print("\nSample sentence pair:")
        print(f"  Korean:  {ds['train'][0]['ko']}")
        print(f"  English: {ds['train'][0]['en']}")

        # Save to text files
        print("\nSaving to text files...")
        for split in ['train', 'validation', 'test']:
            ko_path = output_dir / f"{split}.ko"
            en_path = output_dir / f"{split}.en"

            with open(ko_path, 'w', encoding='utf-8') as f_ko, \
                 open(en_path, 'w', encoding='utf-8') as f_en:
                for item in ds[split]:
                    f_ko.write(item['ko'].strip() + '\n')
                    f_en.write(item['en'].strip() + '\n')

            print(f"  ✓ Saved {split}.ko and {split}.en")

        print(f"\n✓ Moo dataset saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"\n✗ Error downloading Moo dataset: {e}")
        return False


def download_tatoeba_dataset(raw_dir):
    """Download Helsinki-NLP/tatoeba_mt dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found.")
        print("Please install it: pip install 'datasets<3'")
        return False

    print("\n" + "=" * 60)
    print("Downloading: Helsinki-NLP/tatoeba_mt (eng-kor)")
    print("=" * 60)

    output_dir = raw_dir / "tatoeba"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading dataset from Hugging Face...")
        ds = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-kor")

        print("\nDataset loaded successfully!")
        print("Note: Tatoeba only has test and validation sets (no training set)")
        print(f"  Test:       {len(ds['test']):,} sentence pairs")
        print(f"  Validation: {len(ds['validation']):,} sentence pairs")
        print("\nSample sentence pair:")
        print(f"  English: {ds['test'][0]['sourceString']}")
        print(f"  Korean:  {ds['test'][0]['targetString']}")

        # Save to text files
        print("\nSaving to text files...")
        for split in ['test', 'validation']:
            ko_path = output_dir / f"{split}.ko"
            en_path = output_dir / f"{split}.en"

            with open(ko_path, 'w', encoding='utf-8') as f_ko, \
                 open(en_path, 'w', encoding='utf-8') as f_en:
                for item in ds[split]:
                    # Note: Tatoeba has sourceString (English) and targetString (Korean)
                    f_en.write(item['sourceString'].strip() + '\n')
                    f_ko.write(item['targetString'].strip() + '\n')

            print(f"  ✓ Saved {split}.ko and {split}.en")

        print(f"\n✓ Tatoeba dataset saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"\n✗ Error downloading Tatoeba dataset: {e}")
        return False

def download_aihub_dataset(raw_dir):
    """Download AI Hub dataset via Korpora."""
    try:
        from Korpora import Korpora
        if not hasattr(ET.ElementTree, "getiterator"):
            ET.ElementTree.getiterator = ET.ElementTree.iter
    except ImportError:
        print("Error: 'Korpora' library not found.")
        print("Please install it: pip install Korpora")
        return False

    print("\n" + "=" * 60)
    print("Downloading: AI Hub Translation via Korpora")
    print("=" * 60)

    output_dir = raw_dir / "aihub"

    print("""
IMPORTANT: AI Hub dataset requires manual registration and download.

Steps:
1. Visit https://www.aihub.or.kr/
2. Click "AI데이터" and find the translation dataset
3. Apply for access (auto-approved)
4. Download the dataset
5. Extract and rename files as shown in Korpora documentation
6. Place files in: {}

After manual setup, Korpora will load the local files.
""".format(output_dir))

    try:
        print("Attempting to load AI Hub dataset from local files...")
        corpus = Korpora.load("aihub_translation", root_dir=str(output_dir))

        print(f"\n✓ AI Hub dataset loaded successfully!")
        print(corpus)
        print("\nDataset loaded successfully!")
        print("Note: AI Hub only has train set (no test and validation set)")
        print(f"  Train:      {len(corpus.train):,} sentence pairs")
        print("\nSample sentence pair:")
        print(f"  Korean:  {corpus.train[0].text}")
        print(f"  English: {corpus.train[0].pair}")

        # Extract and save parallel sentences
        print("\nExtracting and saving to text files...")

        # AIHub only has train split
        # Each item has 'text' (Korean) and 'pair' (English) attributes
        ko_path = output_dir / "train.ko"
        en_path = output_dir / "train.en"

        with open(ko_path, 'w', encoding='utf-8') as f_ko, \
             open(en_path, 'w', encoding='utf-8') as f_en:
            for item in corpus.train:
                f_ko.write(item.text.strip() + '\n')
                f_en.write(item.pair.strip() + '\n')

        print(f"  ✓ Saved train.ko and train.en ({len(corpus.train):,} pairs)")
        print(f"\n✓ AI Hub dataset saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"\n✗ Could not load AI Hub dataset: {e}")
        print("\nThis is expected if you haven't downloaded the dataset manually yet.")
        print("The AI Hub dataset requires manual download from the website.")
        return False


def main():
    """Main function to download datasets."""
    config = BaseConfig()

    # Setup paths
    script_dir = Path(__file__).parent
    raw_dir = script_dir.parent / config.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Korean-English Dataset Download")
    print("=" * 60)
    print(f"\nData directory: {raw_dir}")
    print(f"Datasets to download: {', '.join(config.datasets_to_download)}")
    print()

    # Download requested dataset(s)
    results = {}

    for dataset in config.datasets_to_download:
        if dataset == 'moo':
            results['moo'] = download_moo_dataset(raw_dir)
        elif dataset == 'tatoeba':
            results['tatoeba'] = download_tatoeba_dataset(raw_dir)
        elif dataset == 'aihub':
            results['aihub'] = download_aihub_dataset(raw_dir)
        else:
            print(f"\n✗ Unknown dataset: {dataset}")
            results[dataset] = False

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    for dataset_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {dataset_name:10s}: {status}")

    print()
    if any(results.values()):
        print("Next step: Preprocess the data")
        print("  /home/arnold/venv/bin/python scripts/split_data.py")
    else:
        print("No datasets were successfully downloaded.")
        print("Please check the error messages above.")


if __name__ == "__main__":
    main()
