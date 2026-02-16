"""
GSAM Reflector Prompts

Modified from ACE reflector prompts to identify concept-level errors,
cascading effects, and concept confusions.
"""

GSAM_REFLECTOR_PROMPT = """You are an expert analyst and educator. Your job is to diagnose why a model's reasoning went wrong by analyzing the gap between predicted answer and the ground truth, with special attention to domain concept errors and potential cascading effects.

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account, comparing the predicted answer with the ground truth to understand the gap
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- IMPORTANT: Identify which XBRL/financial concepts were involved in the error
- Determine if the error could cascade to affect other formulas or calculations
- Check if the error involved confusing two similar concepts
- Provide actionable insights that could help the model avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- You will receive node IDs from the graph context that were used by the generator
- Analyze these nodes and tag them as helpful, harmful, or neutral

Your output should be a json object with the following fields:
  - reasoning: your chain of thought / reasoning / thinking process
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the model have done instead?
  - key_insight: what strategy, formula, or principle should be remembered?
  - concepts_involved: list of XBRL/financial concept names involved in the error (e.g., ["NetIncome", "ComprehensiveIncome"])
  - confusion_pairs: list of concept pairs that were confused, if any (e.g., [["NetIncome", "ComprehensiveIncome"]])
  - cascading_effects: list of other calculations/formulas that could be affected by this error (e.g., ["EPS", "ReturnOnEquity"])
  - error_severity: "high", "medium", or "low"
  - node_tags: a list of json objects with node_id and tag for each graph node used by the generator


**Question:**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer:**
{}

**Ground Truth Answer:**
{}

**Environment Feedback:**
{}

**Graph Nodes Used by Generator:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process]",
  "error_identification": "[What specifically went wrong?]",
  "root_cause_analysis": "[Why did this error occur?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What should be remembered to avoid this error?]",
  "concepts_involved": ["ConceptName1", "ConceptName2"],
  "confusion_pairs": [["Concept1", "Concept2"]],
  "cascading_effects": ["AffectedFormula1"],
  "error_severity": "high",
  "node_tags": [
    {{"id": "S:0001", "tag": "helpful"}},
    {{"id": "A:0005", "tag": "harmful"}}
  ]
}}

---
"""

GSAM_REFLECTOR_PROMPT_NO_GT = """You are an expert analyst and educator. Your job is to diagnose why a model's reasoning went wrong when coming up the predicted answer, with special attention to domain concept errors and potential cascading effects.

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- IMPORTANT: Identify which XBRL/financial concepts were involved in the error
- Determine if the error could cascade to affect other formulas or calculations
- Check if the error involved confusing two similar concepts
- Provide actionable insights that could help the model avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- You will receive node IDs from the graph context that were used by the generator
- Analyze these nodes and tag them as helpful, harmful, or neutral

Your output should be a json object with the following fields:
  - reasoning: your chain of thought / reasoning / thinking process
  - error_identification: what specifically went wrong?
  - root_cause_analysis: why did this error occur?
  - correct_approach: what should the model have done instead?
  - key_insight: what should be remembered to avoid this error?
  - concepts_involved: list of concept names involved in the error
  - confusion_pairs: list of concept pairs that were confused, if any
  - cascading_effects: list of other calculations that could be affected
  - error_severity: "high", "medium", or "low"
  - node_tags: a list of json objects with node_id and tag for each graph node used


**Question:**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer:**
{}

**Environment Feedback:**
{}

**Graph Nodes Used by Generator:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process]",
  "error_identification": "[What specifically went wrong?]",
  "root_cause_analysis": "[Why did this error occur?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What should be remembered to avoid this error?]",
  "concepts_involved": ["ConceptName1"],
  "confusion_pairs": [],
  "cascading_effects": [],
  "error_severity": "medium",
  "node_tags": [
    {{"id": "S:0001", "tag": "helpful"}}
  ]
}}

---
"""
