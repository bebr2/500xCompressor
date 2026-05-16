"""
data_transform.py - Transform MS MARCO and HotpotQA datasets for 500xCompressor training

Usage:
    python data_transform.py --output_dir output/processed --samples 80000 --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd


class DataTransformer:
    """Transform MS MARCO and HotpotQA datasets into finetuning and pretraining formats."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def load_msmarco(self, path: str) -> pd.DataFrame:
        """Load MS MARCO parquet file."""
        return pd.read_parquet(path)

    def load_hotpotqa(self, paths: List[str]) -> pd.DataFrame:
        """Load and concatenate HotpotQA parquet files."""
        dfs = [pd.read_parquet(p) for p in paths]
        return pd.concat(dfs, ignore_index=True)

    def clean_text(self, text: str) -> Optional[str]:
        """Clean text by removing excess whitespace and control characters."""
        if not text or not isinstance(text, str):
            return None
        text = ' '.join(text.split())
        text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
        return text.strip() if text.strip() else None

    def extract_msmarco(self, row: pd.Series) -> Optional[Dict[str, str]]:
        """Extract QA pair from MS MARCO record."""
        try:
            query = self.clean_text(row['query'])
            answers = row['answers']
            passages = row['passages']

            if not query:
                return None

            # Handle empty answers
            if not answers or len(answers) == 0:
                return None
            answer = self.clean_text(answers[0])
            if not answer:
                return None

            # Concatenate all passages as context
            passage_texts = list(passages['passage_text'])
            all_contexts = [self.clean_text(p) for p in passage_texts]
            all_contexts = [c for c in all_contexts if c]  # Filter None

            if not all_contexts:
                return None

            context = '\n\n'.join(all_contexts)

            return {
                'context': context,
                'question': query,
                'answer': answer,
                'source': 'msmarco'
            }
        except Exception:
            return None

    def extract_hotpotqa(self, row: pd.Series) -> Optional[Dict[str, str]]:
        """Extract QA pair from HotpotQA record."""
        try:
            question = self.clean_text(row['question'])
            answer = self.clean_text(row['answer'])
            context_data = row['context']

            if not question or not answer:
                return None

            # Reconstruct and concatenate all documents
            all_contexts = []
            for title, sentences in zip(context_data['title'], context_data['sentences']):
                full_text = f"{title}: {' '.join(sentences)}"
                cleaned = self.clean_text(full_text)
                if cleaned:
                    all_contexts.append(cleaned)

            if not all_contexts:
                return None

            context = '\n\n'.join(all_contexts)

            return {
                'context': context,
                'question': question,
                'answer': answer,
                'source': 'hotpotqa',
                'type': row.get('type', 'unknown')
            }
        except Exception:
            return None

    def process_dataset(
        self,
        msmarco_path: str,
        hotpotqa_paths: List[str],
        n_samples: int = 80000
    ) -> List[Dict[str, str]]:
        """Process both datasets and return sampled records."""

        # Load data
        print("Loading MS MARCO...")
        msmarco_df = self.load_msmarco(msmarco_path)
        print(f"Loaded {len(msmarco_df)} MS MARCO records")

        print("Loading HotpotQA...")
        hotpotqa_df = self.load_hotpotqa(hotpotqa_paths)
        print(f"Loaded {len(hotpotqa_df)} HotpotQA records")

        # Extract records
        print("Extracting MS MARCO records...")
        msmarco_records = []
        for idx, row in msmarco_df.iterrows():
            record = self.extract_msmarco(row)
            if record:
                msmarco_records.append(record)
        print(f"Extracted {len(msmarco_records)} valid MS MARCO records")

        print("Extracting HotpotQA records...")
        hotpotqa_records = []
        for idx, row in hotpotqa_df.iterrows():
            record = self.extract_hotpotqa(row)
            if record:
                hotpotqa_records.append(record)
        print(f"Extracted {len(hotpotqa_records)} valid HotpotQA records")

        # Calculate sample sizes (proportional)
        total_valid = len(msmarco_records) + len(hotpotqa_records)
        msmarco_ratio = len(msmarco_records) / total_valid
        n_msmarco = int(n_samples * msmarco_ratio)
        n_hotpotqa = n_samples - n_msmarco

        # Stratified sampling for HotpotQA (by type)
        print("Sampling records...")
        sampled_msmarco = random.sample(
            msmarco_records,
            min(n_msmarco, len(msmarco_records))
        )

        # Stratify HotpotQA by question type
        bridge_records = [r for r in hotpotqa_records if r.get('type') == 'bridge']
        comparison_records = [r for r in hotpotqa_records if r.get('type') == 'comparison']

        bridge_ratio = len(bridge_records) / len(hotpotqa_records) if hotpotqa_records else 0
        n_bridge = int(n_hotpotqa * bridge_ratio)
        n_comparison = n_hotpotqa - n_bridge

        sampled_bridge = random.sample(bridge_records, min(n_bridge, len(bridge_records)))
        sampled_comparison = random.sample(comparison_records, min(n_comparison, len(comparison_records)))
        sampled_hotpotqa = sampled_bridge + sampled_comparison

        # Combine and shuffle
        all_records = sampled_msmarco + sampled_hotpotqa
        random.shuffle(all_records)

        print(f"Final sample: {len(all_records)} records")
        print(f"  - MS MARCO: {len(sampled_msmarco)}")
        print(f"  - HotpotQA: {len(sampled_hotpotqa)} (bridge: {len(sampled_bridge)}, comparison: {len(sampled_comparison)})")

        return all_records

    def save_finetune(self, records: List[Dict[str, str]], output_path: str):
        """Save records as JSONL for finetuning."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                json_line = json.dumps({
                    'context': record['context'],
                    'question': record['question'],
                    'answer': record['answer']
                }, ensure_ascii=False)
                f.write(json_line + '\n')
        print(f"Saved {len(records)} records to {output_path}")

    def save_pretrain(self, records: List[Dict[str, str]], output_path: str):
        """Save contexts as plain text for pretraining."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                # Write context as a single line (replace newlines with spaces)
                context_line = record['context'].replace('\n', ' ')
                f.write(context_line + '\n')
        print(f"Saved {len(records)} documents to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Transform datasets for 500xCompressor')
    parser.add_argument('--msmarco_path', type=str,
                        default='output/msmarco/train-00000-of-00001.parquet')
    parser.add_argument('--hotpotqa_paths', type=str, nargs='+',
                        default=['output/hotpotqa/train-00000-of-00002.parquet',
                                'output/hotpotqa/train-00001-of-00002.parquet'])
    parser.add_argument('--output_dir', type=str, default='output/processed')
    parser.add_argument('--samples', type=int, default=80000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transform data
    transformer = DataTransformer(seed=args.seed)
    records = transformer.process_dataset(
        msmarco_path=args.msmarco_path,
        hotpotqa_paths=args.hotpotqa_paths,
        n_samples=args.samples
    )

    # Save outputs
    transformer.save_finetune(records, str(output_dir / 'finetune.jsonl'))
    transformer.save_pretrain(records, str(output_dir / 'pretrain.txt'))


if __name__ == '__main__':
    main()