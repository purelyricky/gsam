"""
GSAM Graph Constructor Prompt

Used by the Graph Constructor to extract typed entities and
relationships from curator delta output for graph operations.
"""

GRAPH_CONSTRUCTOR_PROMPT = """You are a knowledge graph constructor for financial analysis. Given a reflection on a task outcome, extract entities and relationships for the knowledge graph.

Node types: Strategy, AntiPattern, Concept, Formula, Confusion
Edge types: is_a, part_of, depends_on, applies_to, fails_for, fixes, confused_with, conflicts_with

**Example Input:**
"The model confused NetIncome with ComprehensiveIncome when computing EPS. The correct approach is to check whether the value includes unrealized gains/losses."

**Example Output:**
{{
  "nodes": [
    {{"type": "AntiPattern", "content": "Confusing NetIncome with ComprehensiveIncome", "root_cause": "ComprehensiveIncome includes unrealized gains"}},
    {{"type": "Strategy", "content": "Check for unrealized gains/losses to distinguish NetIncome from ComprehensiveIncome"}}
  ],
  "edges": [
    {{"source": "AntiPattern:0", "relation": "fails_for", "target": "Concept:NetIncome"}},
    {{"source": "AntiPattern:0", "relation": "fails_for", "target": "Formula:EPS"}},
    {{"source": "Strategy:0", "relation": "fixes", "target": "AntiPattern:0"}},
    {{"source": "Concept:NetIncome", "relation": "confused_with", "target": "Concept:ComprehensiveIncome"}}
  ]
}}

**Example Input 2:**
"When calculating Current Ratio, the model incorrectly included long-term investments in Current Assets. Always verify that only short-term assets (cash, receivables, inventory) are included."

**Example Output 2:**
{{
  "nodes": [
    {{"type": "AntiPattern", "content": "Including long-term investments in Current Assets calculation", "root_cause": "Long-term investments are non-current assets"}},
    {{"type": "Strategy", "content": "Verify only short-term assets (cash, receivables, inventory) are included in Current Assets"}}
  ],
  "edges": [
    {{"source": "AntiPattern:0", "relation": "fails_for", "target": "Formula:CurrentRatio"}},
    {{"source": "Strategy:0", "relation": "fixes", "target": "AntiPattern:0"}},
    {{"source": "Strategy:0", "relation": "applies_to", "target": "Concept:CurrentAssets"}}
  ]
}}

**Important rules:**
- Node references in edges use format "NodeType:Index" where Index is 0-based position in the nodes array
- For existing concepts, use "Concept:EntityName" (e.g., "Concept:NetIncome")
- For existing formulas, use "Formula:FormulaName" (e.g., "Formula:EPS")
- Only extract clearly stated strategies and anti-patterns, do not invent
- Be concise in content descriptions
- RESPOND WITH VALID JSON ONLY

Now extract from the following reflection:
{}
"""
