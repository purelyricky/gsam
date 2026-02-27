"""
eval/medical/data_processor.py
================================
DataProcessor for medical NER benchmarks, currently supporting BC5CDR.

BC5CDR Task (analogous to FiNER):
  - Input  : A biomedical sentence containing a medical entity.
  - Output : Entity type — "Chemical" or "Disease".

The data format produced by download_bc5cdr.py is already in the standard
GSAM format {context, question, target, others}, so process_task_data just
validates/normalises fields.
"""

import re
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Supported tasks
# ---------------------------------------------------------------------------

SUPPORTED_TASKS = {"bc5cdr"}

# Canonical entity types for BC5CDR
BC5CDR_ENTITY_TYPES = {"chemical", "disease"}


# ---------------------------------------------------------------------------
# DataProcessor
# ---------------------------------------------------------------------------


class MedicalDataProcessor:
    """
    DataProcessor for medical NER tasks.

    Mirrors the interface of eval.finance.data_processor.DataProcessor so that
    the same GSAM orchestration code works without modification:

        processor.process_task_data(raw_data)  -> List[Dict]
        processor.answer_is_correct(pred, gt)  -> bool
        processor.evaluate_accuracy(preds, gts) -> float
    """

    def __init__(self, task_name: str):
        if task_name not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unknown medical task: {task_name!r}. "
                f"Supported tasks: {SUPPORTED_TASKS}"
            )
        self.task_name = task_name

    # ------------------------------------------------------------------
    # process_task_data
    # ------------------------------------------------------------------

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Validate and normalise raw JSONL records from the download script.

        The download script already produces {context, question, target, others},
        so this mainly validates required fields and provides a consistent
        hook for future preprocessing.

        Returns:
            List of standardised {context, question, target, others} dicts.
        """
        processed = []
        for item in raw_data:
            context = item.get("context", "")
            question = item.get("question", "")
            target = item.get("target", "")

            if not question:
                # Reconstruct question from mention if missing (robustness)
                mention = item.get("others", {}).get("mention", "")
                question = (
                    f"What is the medical entity type of '{mention}' in this text? "
                    "Answer with exactly one of: Chemical, Disease."
                    if mention
                    else "What type of medical entity is mentioned in this text?"
                )

            processed.append({
                "context": context,
                "question": question,
                "target": target,
                "others": item.get("others", {
                    "task": self.task_name,
                    "data_source": "bc5cdr",
                }),
            })

        return processed

    # ------------------------------------------------------------------
    # answer_is_correct
    # ------------------------------------------------------------------

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if the predicted entity type matches the ground truth.

        Comparison is case-insensitive and strips whitespace/punctuation.
        We also handle common paraphrase patterns such as:
          - "The entity is a Chemical" → "chemical"
          - "chemical compound"       → "chemical"
          - "it is a Disease"         → "disease"
        """
        if self.task_name == "bc5cdr":
            return self._bc5cdr_answer_is_correct(predicted, ground_truth)
        raise ValueError(f"Unknown task: {self.task_name}")

    def _bc5cdr_answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        pred_norm = self._normalise_entity_type(predicted)
        gt_norm = self._normalise_entity_type(ground_truth)
        return pred_norm == gt_norm

    @staticmethod
    def _normalise_entity_type(text: str) -> str:
        """
        Extract "chemical" or "disease" from a model response.

        Handles both direct answers ("Chemical") and embedded answers
        ("The entity type is Chemical.").
        """
        text = text.lower().strip()
        # Direct match first
        if text in BC5CDR_ENTITY_TYPES:
            return text
        # Search for the entity type word anywhere in the response
        for et in BC5CDR_ENTITY_TYPES:
            if et in text:
                return et
        # Fallback: return the cleaned text for comparison
        return re.sub(r'[^a-z]', '', text)

    # ------------------------------------------------------------------
    # evaluate_accuracy
    # ------------------------------------------------------------------

    def evaluate_accuracy(
        self, predictions: List[str], targets: List[str]
    ) -> float:
        """
        Compute exact-match accuracy over a list of predictions and targets.

        Mirrors the FiNER accuracy interface (returns float in [0, 1]).
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs "
                f"{len(targets)} targets"
            )
        if not predictions:
            return 0.0

        if self.task_name == "bc5cdr":
            return self._bc5cdr_evaluate_accuracy(predictions, targets)
        raise ValueError(f"Unknown task: {self.task_name}")

    def _bc5cdr_evaluate_accuracy(
        self, predictions: List[str], targets: List[str]
    ) -> float:
        correct = sum(
            1 for p, t in zip(predictions, targets)
            if self._bc5cdr_answer_is_correct(p, t)
        )
        return correct / len(predictions)
