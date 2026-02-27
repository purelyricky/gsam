#!/usr/bin/env python3
"""
download_bc5cdr.py
==================
Download and convert the BC5CDR dataset from Hugging Face into the GSAM
JSONL format used by the medical DataProcessor.

The BC5CDR (BioCreative V Chemical Disease Relation) dataset contains:
  - 1,500 PubMed article titles + abstracts
  - 2 entity types: Chemical and Disease
  - Concept normalization to MeSH IDs

The original BC5CDR annotation format uses token-level BIO tags per document.
We convert it into a GSAM-compatible text-generation format:

  Per example (one entity mention per example):
    context  : The sentence containing the entity mention.
    question : "What is the medical entity type of '<SPAN>' in this text?"
    target   : "Chemical" or "Disease"

This mirrors the FiNER format where:
    context  : A financial text excerpt.
    question : An instruction identifying a numeric token.
    target   : One of 139 XBRL entity type labels.

Usage
-----
    python eval/medical/data/download_bc5cdr.py \
        --output_dir ./eval/medical/data \
        --max_train 1000 \
        --max_val   300  \
        --max_test  300

Requirements
------------
    pip install datasets

The `datasets` package is the only extra dependency; it is not in the main
requirements.txt to avoid forcing it on all users.
"""

import os
import sys
import json
import argparse
import random
from typing import List, Dict, Any

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: The 'datasets' package is required to download BC5CDR.")
    print("Install it with:  pip install datasets")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _extract_entity_examples(split_data) -> List[Dict[str, Any]]:
    """
    Convert BC5CDR documents into one GSAM example per entity mention.

    Each example asks the model to classify a single entity span as
    "Chemical" or "Disease" given the surrounding sentence context.

    The BC5CDR dataset from bigbio/bc5cdr uses the BigBio schema:
      document_id, passages (with entities inline), entities, relations
    """
    examples = []

    for doc in split_data:
        # Collect all entity annotations across all passages
        doc_entities = doc.get("entities", [])
        if not doc_entities:
            continue

        # Build a flat text map: passage_id -> text
        passages = doc.get("passages", [])
        passage_text = {}
        for p in passages:
            offsets = p.get("offsets", [[0, 0]])
            start = offsets[0][0] if offsets else 0
            passage_text[start] = p.get("text", "")

        # Find the full document text (concatenate passages for context)
        full_text = " ".join(p.get("text", "") for p in passages)

        for entity in doc_entities:
            entity_type = entity.get("type", "")
            if entity_type not in ("Chemical", "Disease"):
                continue

            # Use the first mention text
            mention_texts = entity.get("text", [])
            if not mention_texts:
                continue
            mention = mention_texts[0].strip()
            if not mention:
                continue

            # Use the sentence that most tightly contains the mention as context
            context_sentence = _find_sentence_with_mention(full_text, mention)

            question = (
                f"What is the medical entity type of '{mention}' in the following "
                f"biomedical text? Answer with exactly one of: Chemical, Disease."
            )

            examples.append({
                "context": context_sentence,
                "question": question,
                "target": entity_type,
                "others": {
                    "document_id": doc.get("document_id", ""),
                    "entity_id": entity.get("id", ""),
                    "mention": mention,
                    "normalized": entity.get("normalized", []),
                    "task": "bc5cdr",
                    "data_source": "bc5cdr",
                },
            })

    return examples


def _find_sentence_with_mention(full_text: str, mention: str) -> str:
    """
    Return the sentence in full_text that contains the mention.
    Falls back to full_text if no sentence boundary is found.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    for sent in sentences:
        if mention.lower() in sent.lower():
            return sent.strip()
    # Fallback: return first 512 characters of document text
    return full_text[:512]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert BC5CDR dataset to GSAM JSONL format"
    )
    parser.add_argument("--output_dir", type=str,
                        default="./eval/medical/data",
                        help="Directory to write bc5cdr_train/val/test.jsonl")
    parser.add_argument("--max_train", type=int, default=1000,
                        help="Max training examples (None = all)")
    parser.add_argument("--max_val", type=int, default=300,
                        help="Max validation examples")
    parser.add_argument("--max_test", type=int, default=300,
                        help="Max test examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Downloading BC5CDR from Hugging Face (bigbio/bc5cdr) ...")
    # bigbio schema gives us structured access to entities
    dataset = load_dataset("bigbio/bc5cdr", name="bc5cdr_bigbio_kb", trust_remote_code=True)

    splits = {
        "train": ("train", args.max_train, "bc5cdr_train.jsonl"),
        "val":   ("validation", args.max_val, "bc5cdr_val.jsonl"),
        "test":  ("test", args.max_test, "bc5cdr_test.jsonl"),
    }

    for split_name, (hf_split, max_n, filename) in splits.items():
        if hf_split not in dataset:
            print(f"  Split '{hf_split}' not found in dataset, skipping.")
            continue

        print(f"\nProcessing {hf_split} split ({len(dataset[hf_split])} documents) ...")
        examples = _extract_entity_examples(dataset[hf_split])

        # Shuffle and cap
        random.shuffle(examples)
        if max_n is not None:
            examples = examples[:max_n]

        out_path = os.path.join(args.output_dir, filename)
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        # Count entity type distribution
        counts = {}
        for ex in examples:
            t = ex["target"]
            counts[t] = counts.get(t, 0) + 1

        print(f"  Wrote {len(examples)} examples to {out_path}")
        print(f"  Distribution: {counts}")

    print("\nDone!  You can now run GSAM with:")
    print("    python -m eval.medical.run_gsam_medical \\")
    print("        --task_name bc5cdr \\")
    print("        --api_provider sambanova \\")
    print("        --mode online \\")
    print("        --save_path ./results/bc5cdr")


if __name__ == "__main__":
    main()
