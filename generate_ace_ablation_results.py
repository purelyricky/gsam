"""
generate_ace_ablation_results.py
Generates synthetic reference result files for ACE and GSAM ablation experiments.
Every JSON includes "synthetic_reference": true.
"""

import json
import os
import random
import math
from datetime import datetime, timedelta

# ─── Seed for reproducibility ────────────────────────────────────────────────
random.seed(42)

BASE_DIR = "C:/Users/Window/Desktop/gsam-rsh/clean_results"

# ─── Helper utilities ─────────────────────────────────────────────────────────

def makedirs(path):
    os.makedirs(path, exist_ok=True)

def write_json(path, data, indent=2):
    makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

def write_jsonl(path, records):
    makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

def write_text(path, content):
    makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def ts(base_dt, delta_seconds):
    """Return ISO timestamp string offset from base_dt by delta_seconds."""
    return (base_dt + timedelta(seconds=delta_seconds)).isoformat()

def fmt_dt(base_dt, delta_seconds):
    """Return log-friendly timestamp."""
    dt = base_dt + timedelta(seconds=delta_seconds)
    return dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]

# ─── XBRL concept names used in logs ─────────────────────────────────────────
XBRL_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetIncomeLoss",
    "DebtInstrumentFaceAmount",
    "DebtInstrumentCarryingAmount",
    "CommonStockSharesAuthorized",
    "CommonStockSharesOutstanding",
    "AllocatedShareBasedCompensationExpense",
    "LineOfCreditFacilityMaximumBorrowingCapacity",
    "LineOfCreditFacilityCurrentBorrowingCapacity",
    "NumberOfOperatingSegments",
    "NumberOfReportableSegments",
    "Depreciation",
    "AmortizationOfIntangibleAssets",
    "PropertyPlantAndEquipmentNet",
    "PropertyPlantAndEquipmentGross",
    "InterestExpense",
    "InterestPaid",
    "OperatingLeaseExpense",
    "FinanceLeaseExpense",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "RetainedEarningsAccumulatedDeficit",
    "AdditionalPaidInCapital",
    "Goodwill",
    "GoodwillImpairmentLoss",
    "RestructuringCharges",
    "DeferredFinanceCostsNet",
    "IncomeTaxExpenseBenefit",
    "EffectiveIncomeTaxRateContinuingOperations",
]

BULLET_CONTENTS = [
    "When a sentence references 'principal amount of X' or 'face value of $X million', use DebtInstrumentFaceAmount. When it says 'carrying amount' or 'book value' or includes adjustments for discount/premium, use DebtInstrumentCarryingAmount.",
    "Do not confuse LineOfCreditFacilityMaximumBorrowingCapacity with LineOfCreditFacilityCurrentBorrowingCapacity: maximum is the total credit limit in the agreement; current/remaining is what's available to draw after existing draws.",
    "RevenueFromContractWithCustomerExcludingAssessedTax applies to revenue after excluding sales/VAT taxes explicitly mentioned. If the text says 'net revenue' without tax mention, still prefer this tag.",
    "NumberOfOperatingSegments counts business divisions reported internally. NumberOfReportableSegments is the external GAAP disclosure count. These frequently differ by 1-2 segments.",
    "EarningsPerShareBasic uses weighted-average shares outstanding (no dilution). EarningsPerShareDiluted adds in-the-money options and convertibles. When text says 'diluted EPS', always use diluted.",
    "For lease terms: LesseeOperatingLeaseTermOfContract is the initial fixed term. Use LesseeFinanceLeaseTermOfContract only when the lease transfers ownership or contains a bargain purchase option.",
    "Depreciation applies to tangible fixed assets (PP&E). AmortizationOfIntangibleAssets applies to patents, trademarks, customer lists. Depletion applies to natural resources — if all three are combined, use DepletionDepreciationAndAmortization.",
    "InterestExpense is accrual-based (income statement). InterestPaid is cash-based (cash flow statement). When context mentions 'paid' explicitly, use InterestPaid.",
    "AllocatedShareBasedCompensationExpense is the P&L charge. EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized is the unrecognized future cost.",
    "PropertyPlantAndEquipmentGross is before accumulated depreciation. PropertyPlantAndEquipmentNet is after. 'Net book value' always maps to the Net tag.",
    "Goodwill is only impaired downward, never amortized under US GAAP. GoodwillImpairmentLoss is the specific impairment charge recognized in that period.",
    "CommonStockSharesAuthorized is the charter maximum. CommonStockSharesOutstanding is actual shares in investor hands. CommonStockSharesIssued includes treasury shares.",
    "AdditionalPaidInCapital (APIC) captures amounts paid above par value. RetainedEarningsAccumulatedDeficit is cumulative net income minus dividends. Deficit = accumulated losses.",
    "IncomeTaxExpenseBenefit is the total tax provision (current + deferred). EffectiveIncomeTaxRateContinuingOperations is expressed as a percentage, not dollar amount.",
    "RestructuringCharges includes severance, facility closure costs. BusinessCombinationConsiderationTransferred1 is M&A deal value — do not confuse restructuring with acquisition costs.",
    "DeferredFinanceCostsNet (issuance costs) is netted against the debt balance on the balance sheet post-ASU 2015-03. Always check if 'deferred financing costs' are net or gross.",
    "OperatingLeaseExpense is the straight-line operating lease cost. FinanceLeaseExpense has two components: amortization + interest. Total finance lease cost = both combined.",
    "When text mentions 'effective tax rate' or 'ETR', use EffectiveIncomeTaxRateContinuingOperations (decimal, e.g. 0.21 for 21%). Dollar tax charges use IncomeTaxExpenseBenefit.",
    "EarningsPerShareBasic and Diluted are per-share amounts, typically small numbers ($0.xx to $xx.xx). Do not confuse with net income in millions.",
    "ContractWithCustomerLiabilityRevenueRecognized is revenue recognized FROM deferred revenue. ContractWithCustomerLiability is the remaining deferred balance. Track direction of flow.",
    "For stock repurchases: StockRepurchaseProgramAuthorizedAmount1 is the board-approved total. StockRepurchasedAndRetiredDuringPeriodValue or Shares is what was actually bought back.",
    "SaleOfStockPricePerShare is the offering price per share. SaleOfStockNumberOfSharesIssuedInTransaction is the quantity. Proceeds = price × shares.",
    "UnrecognizedTaxBenefits is the gross amount of uncertain tax positions. UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate is the subset that would affect ETR if resolved.",
    "ProceedsFromIssuanceOfCommonStock is cash inflow (financing activities). StockIssuedDuringPeriodSharesNewIssues is the number of shares. Both may appear together in equity offerings.",
    "DerivativeNotionalAmount is the contractual reference amount (not fair value). DebtInstrumentFairValue is the market value of a debt instrument. These are conceptually distinct.",
    "For concentration risk: ConcentrationRiskPercentage1 is the percentage. Always check the ConcentrationRiskBenchmarkDescription to understand what it's a percentage of.",
    "LossContingencyEstimateOfPossibleLoss is a range estimate (may need MinimumMaximum suffix). LettersOfCreditOutstandingAmount is a credit facility contingency, not a loss estimate.",
    "EquityMethodInvestmentOwnershipPercentage is for investments where the investor has significant influence (typically 20-50%). Below 20% may use cost method.",
    "DefinedBenefitPlanContributionsByEmployer is annual cash contributed to pension plan. DefinedContributionPlanCostRecognized is 401(k)-style expense.",
    "AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount: shares excluded because including them would increase EPS. Always in shares, not dollars.",
    "ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1: the exercise/strike price per warrant. Per-share amount, typically stated in dollars.",
    "DebtInstrumentInterestRateStatedPercentage is the coupon rate in the indenture. DebtInstrumentInterestRateEffectivePercentage accounts for discount/premium (yield).",
    "GoodwillImpairmentLoss is typically a large one-time charge. Do not confuse with amortization — US GAAP goodwill is NOT amortized, only tested for impairment annually.",
    "BusinessCombinationAcquisitionRelatedCosts are transaction costs (legal, advisory fees). BusinessCombinationConsiderationTransferred1 is the actual purchase price paid.",
    "MinorityInterestOwnershipPercentageByParent: parent's % ownership in consolidated subsidiary. NoncontrollingInterest is the NCI balance on the balance sheet.",
    "DefinedBenefitPlanNetPeriodicBenefitCost has multiple components (service cost, interest, expected return). Total pension expense in P&L uses this tag.",
    "CashAndCashEquivalentsAtCarryingValue is the balance sheet amount. CashAndCashEquivalentsFairValueDisclosure is from fair value footnote tables — usually equal.",
    "ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1 is the total vesting term (e.g., '4 years'). Vesting rights percentage is the annual cliff amount.",
    "DebtInstrumentUnamortizedDiscount is the original issue discount being amortized to interest expense. DebtInstrumentBasisSpreadOnVariableRate1 is the credit spread (e.g., LIBOR+150bps).",
    "For multi-period Revenue: use RevenueFromContractWithCustomerExcludingAssessedTax for the fiscal year amount. Quarterly amounts use the same tag — period context distinguishes them.",
]

SECTIONS = ["identification_strategies", "common_errors", "formulas_and_calculations", "general_tips"]

SECTION_BULLETS = {
    "identification_strategies": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    "common_errors": [1, 3, 7, 10, 13, 17, 18, 25, 26, 31, 32],
    "formulas_and_calculations": [5, 6, 8, 14, 15, 16, 20, 22, 23],
    "general_tips": [4, 11, 12, 19, 24, 27, 28, 29, 30],
}

MODAL_URL = "https://xxxx--gsam-deepseek-serve-deepseekserver-serve.modal.run/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-Q4"

# ─── Window accuracy progressions ─────────────────────────────────────────────

ACE_FINER_WINDOW_ACC = [
    0.433, 0.450, 0.467, 0.483, 0.497,
    0.503, 0.510, 0.517, 0.523, 0.527,
    0.527, 0.530, 0.533, 0.530, 0.530,
    0.530, 0.530, 0.530, 0.530, 0.530,
]

ACE_FORMULA_WINDOW_ACC = [
    0.430, 0.447, 0.463, 0.477, 0.490,
    0.497, 0.503, 0.510, 0.517, 0.523,
    0.527, 0.530, 0.535, 0.538, 0.540,
    0.540, 0.540, 0.540, 0.540, 0.540,
]

ACE_FINER_OFFLINE_EPOCH_ACC = [0.433, 0.493, 0.527, 0.550, 0.577, 0.600]
ACE_FORMULA_OFFLINE_EPOCH_ACC = [0.430, 0.533, 0.600, 0.647, 0.683, 0.700]

# ─── Bullet size progression for ACE ─────────────────────────────────────────
# Per-window bullet counts (cumulative at end of window)
ACE_FINER_BULLET_COUNTS = [12, 20, 28, 34, 38, 43, 49, 55, 60, 65, 70, 75, 80, 84, 88, 94, 100, 106, 112, 118]
# After deduplication passes:
ACE_FINER_BULLET_COUNTS_DEDUP = [12, 20, 28, 34, 38, 43, 49, 55, 60, 65, 70, 75, 80, 84, 88, 94, 100, 106, 112, 116]
# Actually let's use the final 148 from the spec; scale to reach 148
ACE_FINER_BULLET_COUNTS_FINAL = [12, 24, 36, 46, 54, 62, 71, 80, 88, 96, 104, 112, 118, 124, 128, 132, 136, 140, 144, 148]

ACE_FORMULA_BULLET_COUNTS = [10, 20, 30, 38, 44, 50, 58, 65, 72, 78, 84, 90, 96, 101, 106, 110, 114, 118, 122, 126]

# ─────────────────────────────────────────────────────────────────────────────
# ACE FINER ONLINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_ace_finer_online():
    run_id = "ace_run_SYNTHETIC_20260312_092000_finer_online"
    base_path = f"{BASE_DIR}/ace/ace_finer_online/{run_id}"
    base_dt = datetime(2026, 3, 12, 9, 20, 0)

    # run_config.json
    run_config = {
        "synthetic_reference": True,
        "run_id": run_id,
        "task": "finer",
        "mode": "online",
        "model": MODEL_NAME,
        "max_samples": 300,
        "num_windows": 20,
        "window_size": 15,
        "max_rounds": 1,
        "num_epochs": 1,
        "use_typed_edges": False,
        "use_ontology": False,
        "use_cascades": False,
        "system": "ACE",
        "modal_url": MODAL_URL,
        "started_at": base_dt.isoformat(),
        "completed_at": (base_dt + timedelta(hours=3, minutes=22)).isoformat(),
        "git_hash": "c99514f",
    }
    write_json(f"{base_path}/run_config.json", run_config)

    # final_results.json
    final_results = {
        "synthetic_reference": True,
        "accuracy": 0.530,
        "correct": 159,
        "total": 300,
        "skipped": 0,
        "initial_accuracy": 0.433,
        "initial_correct": 130,
        "window_accuracies": ACE_FINER_WINDOW_ACC,
        "final_playbook_size": 148,
        "playbook_sections": {
            "identification_strategies": 52,
            "common_errors": 38,
            "formulas_and_calculations": 35,
            "general_tips": 23,
        },
        "avg_latency_s": 12.8,
        "gen_latency_s": 5.4,
        "ref_latency_s": 3.9,
        "cur_latency_s": 3.1,
        "task": "finer",
        "mode": "online",
        "model": MODEL_NAME,
    }
    write_json(f"{base_path}/final_results.json", final_results)

    # partial_online_results.json
    partial = {
        "synthetic_reference": True,
        "windows_completed": 20,
        "window_results": [
            {
                "window": i + 1,
                "accuracy": ACE_FINER_WINDOW_ACC[i],
                "correct": int(round(ACE_FINER_WINDOW_ACC[i] * 300)),
                "playbook_size": ACE_FINER_BULLET_COUNTS_FINAL[i],
            }
            for i in range(20)
        ],
    }
    write_json(f"{base_path}/partial_online_results.json", partial)

    # progress.json
    progress = {
        "synthetic_reference": True,
        "total_steps": 300,
        "completed_steps": 300,
        "current_window": 20,
        "total_windows": 20,
        "status": "complete",
        "last_update": (base_dt + timedelta(hours=3, minutes=20)).isoformat(),
    }
    write_json(f"{base_path}/progress.json", progress)

    # bullet_usage_log.jsonl (300 entries)
    bullet_log = []
    elapsed = 90  # seconds offset from base_dt
    for step in range(1, 301):
        window = (step - 1) // 15 + 1
        window_idx = window - 1
        bullet_count = ACE_FINER_BULLET_COUNTS_FINAL[window_idx] if window_idx < 20 else 148
        # How many bullets used (grows over time)
        bullets_available = ACE_FINER_BULLET_COUNTS_FINAL[max(0, window_idx - 1)] if window_idx > 0 else 0
        n_used = min(bullets_available, max(0, int(bullets_available * 0.3) + random.randint(0, 3)))
        bullet_ids_used = [f"bullet_{i:04d}" for i in random.sample(range(1, bullets_available + 1), min(n_used, bullets_available))] if bullets_available > 0 else []
        is_correct = random.random() < ACE_FINER_WINDOW_ACC[window_idx]
        ref_summary = None
        if not is_correct and step > 1:
            tag = random.choice(XBRL_TAGS)
            ref_summary = f"Applied bullet about {tag}. Reflection identified confusion between similar concepts."
        entry = {
            "timestamp": ts(base_dt, elapsed),
            "epoch": 1,
            "step": step,
            "sample_id": f"epoch_1_step_{step}",
            "bullet_ids_used": bullet_ids_used,
            "bullets_with_content": bullet_ids_used,
            "is_correct": is_correct,
            "sample_context": "",
            "sample_question": f"You are XBRL expert. Here is a list of US GAAP tags options: {random.choice(XBRL_TAGS)}, {random.choice(XBRL_TAGS)}...",
            "reflection_summary": ref_summary,
            "bullet_count": bullet_count,
        }
        bullet_log.append(entry)
        elapsed += random.randint(35, 50)
    write_jsonl(f"{base_path}/bullet_usage_log.jsonl", bullet_log)

    # curator_operations_diff.jsonl (300 entries)
    op_types = ["ADD"] * 5 + ["UPDATE"] * 3 + ["MERGE"] * 1 + ["DELETE"] * 1
    curator_log = []
    elapsed2 = 120
    for step in range(1, 301):
        section = random.choice(SECTIONS)
        op = random.choice(op_types)
        content_idx = (step - 1) % len(BULLET_CONTENTS)
        content = BULLET_CONTENTS[content_idx]
        entry = {
            "timestamp": ts(base_dt, elapsed2),
            "operation_type": op,
            "reason": f"{'New pattern identified' if op == 'ADD' else 'Refinement of existing pattern' if op == 'UPDATE' else 'Near-duplicate merged' if op == 'MERGE' else 'Outdated heuristic removed'}",
            "section": section,
            "content": content,
            "content_length": len(content),
            "call_id": f"online_train_s_{step}",
        }
        curator_log.append(entry)
        elapsed2 += random.randint(30, 45)
    write_jsonl(f"{base_path}/curator_operations_diff.jsonl", curator_log)

    # Intermediate playbooks
    for window, step, bullet_count in [(5, 75, 54), (10, 150, 96), (20, 300, 148)]:
        bullets = []
        for i in range(bullet_count):
            content_idx = i % len(BULLET_CONTENTS)
            section = SECTIONS[i % len(SECTIONS)]
            bullets.append({
                "id": f"bullet_{i+1:04d}",
                "section": section,
                "content": BULLET_CONTENTS[content_idx],
                "helpful": random.randint(1, 12),
                "harmful": random.randint(0, 2),
            })
        playbook = {
            "synthetic_reference": True,
            "window": window,
            "step": step,
            "bullet_count": bullet_count,
            "bullets": bullets,
        }
        write_json(f"{base_path}/intermediate_playbooks/playbook_window_{window}.json", playbook)

    # detailed_llm_logs/ (30 representative files)
    makedirs(f"{base_path}/detailed_llm_logs")
    roles = ["generator", "reflector", "curator"]
    for i in range(30):
        step = i * 10 + 1
        role = roles[i % 3]
        suffix = "gen_initial" if role == "generator" else f"round_{i % 2}" if role == "reflector" else ""
        fname = f"{role}_online_train_s_{step}{'_' + suffix if suffix else ''}_{fmt_dt(base_dt, 90 + i*1100)}.json"
        log_entry = {
            "synthetic_reference": True,
            "call_id": f"online_train_s_{step}{'_' + suffix if suffix else ''}",
            "role": role,
            "step": step,
            "model": MODEL_NAME,
            "prompt_tokens": random.randint(800, 2400),
            "completion_tokens": random.randint(200, 600),
            "latency_s": round(random.uniform(4.5, 14.0), 2),
            "timestamp": ts(base_dt, 90 + i * 1100),
            "is_correct": random.random() < 0.53,
            "tag_predicted": random.choice(XBRL_TAGS),
            "tag_ground_truth": random.choice(XBRL_TAGS),
        }
        write_json(f"{base_path}/detailed_llm_logs/{fname}", log_entry)

    # ── Main log file ─────────────────────────────────────────────────────────
    log_lines = []
    log_lines.append("[INFO] Online mode requires num_epochs=1; overriding from 5 to 1")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("ACE SYSTEM")
    log_lines.append("=" * 60)
    log_lines.append("Task: finer")
    log_lines.append("Mode: ONLINE")
    log_lines.append(f"Generator Model: {MODEL_NAME}")
    log_lines.append("Max Samples: 300")
    log_lines.append("=" * 60)
    log_lines.append("")
    log_lines.append("Loaded 300 samples from ./eval/finance/data/finer_test.json")
    log_lines.append("Online mode: Training and testing on 300 examples")
    log_lines.append("Using empty playbook as initial playbook")
    log_lines.append("")
    log_lines.append(f"Using MODAL API  ({MODAL_URL})")
    log_lines.append("")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("ACE SYSTEM - ONLINE MODE")
    log_lines.append("=" * 60)
    log_lines.append("Task: finer")
    log_lines.append("Test samples (used for training and testing): 300")
    log_lines.append("=" * 60)
    log_lines.append("")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("INITIAL TEST (before training)")
    log_lines.append("=" * 60)
    log_lines.append("")
    log_lines.append("")
    log_lines.append("=" * 40)
    log_lines.append("EVALUATING TEST SET - 300 samples, 20 workers")
    log_lines.append("=" * 40)
    # Initial test calls
    for i in range(300):
        log_lines.append(f"[GENERATOR] Starting call test_eval_{i}...")
    for i in range(300):
        latency = round(random.uniform(4.5, 9.5), 2)
        log_lines.append(f"[GENERATOR] Call test_eval_{i} completed in {latency}s")
        log_lines.append(f"[LOG] generator call logged to generator_test_eval_{i}_{fmt_dt(base_dt, 60 + i*2)}.json")
    log_lines.append("Final Accuracy: 0.433 (130/300)")

    # Windows
    elapsed_s = 1200  # ~20 min after start
    for w in range(1, 21):
        window_idx = w - 1
        prev_bullets = ACE_FINER_BULLET_COUNTS_FINAL[window_idx - 1] if window_idx > 0 else 0
        curr_bullets = ACE_FINER_BULLET_COUNTS_FINAL[window_idx]
        acc = ACE_FINER_WINDOW_ACC[window_idx]
        correct = int(round(acc * 300))

        log_lines.append("")
        log_lines.append("=" * 60)
        log_lines.append(f"WINDOW {w}/20 (samples {(w-1)*15}-{w*15-1})")
        log_lines.append("=" * 60)
        log_lines.append("")
        log_lines.append("=" * 40)
        log_lines.append("EVALUATING TEST SET - 300 samples, 20 workers")
        log_lines.append("=" * 40)
        for i in range(300):
            log_lines.append(f"[GENERATOR] Starting call test_eval_{i}...")
        for i in range(300):
            latency = round(random.uniform(4.5, 9.5), 2)
            log_lines.append(f"[GENERATOR] Call test_eval_{i} completed in {latency}s")
            log_lines.append(f"[LOG] generator call logged to generator_test_eval_{i}_{fmt_dt(base_dt, elapsed_s + i*2)}.json")
        log_lines.append(f"Final Accuracy: {acc:.3f} ({correct}/300)")
        log_lines.append(f"Window {w} accuracy: {acc:.3f} | Cumulative: {acc:.3f}")

        elapsed_s += 600 + 300 * 2

        # Steps within window
        for step_in_w in range(1, 16):
            global_step = (w - 1) * 15 + step_in_w
            is_correct = random.random() < acc
            log_lines.append("")
            log_lines.append(f"--- Window {w}, Step {step_in_w}/15 (Global {global_step}) ---")
            log_lines.append(f"[GENERATOR] Starting call online_train_s_{global_step}_gen_initial...")
            latency = round(random.uniform(8.0, 14.0), 2)
            log_lines.append(f"[GENERATOR] Call online_train_s_{global_step}_gen_initial completed in {latency}s")
            log_lines.append(f"[LOG] generator call logged to generator_online_train_s_{global_step}_gen_initial_{fmt_dt(base_dt, elapsed_s)}.json")
            log_lines.append(f"Correct: {is_correct}")

            if not is_correct:
                log_lines.append("Reflection round 1/1")
                ref_latency = round(random.uniform(7.0, 12.0), 2)
                log_lines.append(f"[REFLECTOR] Starting call online_train_s_{global_step}_round_0...")
                log_lines.append(f"[REFLECTOR] Call online_train_s_{global_step}_round_0 completed in {ref_latency}s")
                log_lines.append(f"[LOG] reflector call logged to reflector_online_train_s_{global_step}_round_0_{fmt_dt(base_dt, elapsed_s + int(latency))}.json")
                post_latency = round(random.uniform(6.0, 10.0), 2)
                log_lines.append(f"[GENERATOR] Starting call online_train_s_{global_step}_post_reflect_0...")
                log_lines.append(f"[GENERATOR] Call online_train_s_{global_step}_post_reflect_0 completed in {post_latency}s")
                log_lines.append(f"[LOG] generator call logged to generator_online_train_s_{global_step}_post_reflect_0_{fmt_dt(base_dt, elapsed_s + int(latency) + int(ref_latency))}.json")
                elapsed_s += int(latency) + int(ref_latency) + int(post_latency)
            else:
                elapsed_s += int(latency)

            log_lines.append("")
            log_lines.append(f"--- Running Curator at step {global_step} ---")
            cur_latency = round(random.uniform(7.0, 13.0), 2)
            log_lines.append(f"[CURATOR] Starting call online_train_s_{global_step}...")
            log_lines.append(f"[CURATOR] Call online_train_s_{global_step} completed in {cur_latency}s")
            log_lines.append(f"[LOG] curator call logged to curator_online_train_s_{global_step}_{fmt_dt(base_dt, elapsed_s)}.json")
            added = random.randint(1, 3) if global_step <= 100 else random.randint(0, 2)
            log_lines.append(f"  Playbook: {prev_bullets} bullets \u2192 {prev_bullets + added} bullets (added {added})")
            prev_bullets += added
            post_cur_latency = round(random.uniform(5.0, 9.0), 2)
            log_lines.append(f"[GENERATOR] Starting call online_train_s_{global_step}_post_curate...")
            log_lines.append(f"[GENERATOR] Call online_train_s_{global_step}_post_curate completed in {post_cur_latency}s")
            log_lines.append(f"[LOG] generator call logged to generator_online_train_s_{global_step}_post_curate_{fmt_dt(base_dt, elapsed_s + int(cur_latency))}.json")
            elapsed_s += int(cur_latency) + int(post_cur_latency)

            # Crowding warning at windows 12-15
            if w in (12, 13, 14, 15) and step_in_w == 8:
                pb_now = ACE_FINER_BULLET_COUNTS_FINAL[window_idx]
                log_lines.append(f"[WARNING] Playbook size: {pb_now} bullets. Retrieval may suffer from embedding crowding.")
                log_lines.append("[CURATOR] Attempting deduplication pass...")
                merged = random.randint(1, 3)
                log_lines.append(f"[CURATOR] Merged {merged} near-duplicate bullets (cosine>0.91)")
                log_lines.append(f"  Playbook: {pb_now} \u2192 {pb_now - merged} bullets")

    log_lines.append("")
    log_lines.append(f"Final Online Test Accuracy: 0.530")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("ACE RUN COMPLETE")
    log_lines.append("=" * 60)
    log_lines.append("Final Playbook: 148 bullets")
    log_lines.append("Playbook sections: identification_strategies=52, common_errors=38, formulas_and_calculations=35, general_tips=23")
    log_lines.append("Avg latency: 12.8s/sample (gen=5.4s, ref=3.9s, cur=3.1s)")
    log_lines.append(f"Results saved to: {base_path}")
    log_lines.append("=" * 60)

    write_text(f"{base_path}/ace_finer_online_SYNTHETIC.log", "\n".join(log_lines))
    print(f"  [OK] ace_finer_online: {len(log_lines)} log lines")


# ─────────────────────────────────────────────────────────────────────────────
# ACE FORMULA ONLINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_ace_formula_online():
    run_id = "ace_run_SYNTHETIC_20260312_132000_formula_online"
    base_path = f"{BASE_DIR}/ace/ace_formula_online/{run_id}"
    base_dt = datetime(2026, 3, 12, 13, 20, 0)

    run_config = {
        "synthetic_reference": True,
        "run_id": run_id,
        "task": "formula",
        "mode": "online",
        "model": MODEL_NAME,
        "max_samples": 300,
        "num_windows": 20,
        "window_size": 15,
        "max_rounds": 1,
        "num_epochs": 1,
        "system": "ACE",
        "modal_url": MODAL_URL,
        "started_at": base_dt.isoformat(),
        "completed_at": (base_dt + timedelta(hours=3, minutes=15)).isoformat(),
        "git_hash": "c99514f",
    }
    write_json(f"{base_path}/run_config.json", run_config)

    final_results = {
        "synthetic_reference": True,
        "accuracy": 0.540,
        "correct": 162,
        "total": 300,
        "skipped": 0,
        "initial_accuracy": 0.430,
        "initial_correct": 129,
        "window_accuracies": ACE_FORMULA_WINDOW_ACC,
        "final_playbook_size": 126,
        "task": "formula",
        "mode": "online",
        "model": MODEL_NAME,
    }
    write_json(f"{base_path}/final_results.json", final_results)

    partial = {
        "synthetic_reference": True,
        "windows_completed": 20,
        "window_results": [
            {"window": i+1, "accuracy": ACE_FORMULA_WINDOW_ACC[i],
             "correct": int(round(ACE_FORMULA_WINDOW_ACC[i]*300)),
             "playbook_size": ACE_FORMULA_BULLET_COUNTS[i]}
            for i in range(20)
        ],
    }
    write_json(f"{base_path}/partial_online_results.json", partial)

    progress = {
        "synthetic_reference": True,
        "total_steps": 300,
        "completed_steps": 300,
        "current_window": 20,
        "total_windows": 20,
        "status": "complete",
    }
    write_json(f"{base_path}/progress.json", progress)

    # Minimal log for formula online
    log_lines = []
    log_lines.append("[INFO] Online mode requires num_epochs=1; overriding from 5 to 1")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("ACE SYSTEM")
    log_lines.append("=" * 60)
    log_lines.append("Task: formula")
    log_lines.append("Mode: ONLINE")
    log_lines.append(f"Generator Model: {MODEL_NAME}")
    log_lines.append("Max Samples: 300")
    log_lines.append("=" * 60)
    log_lines.append("")
    log_lines.append("Loaded 300 samples from ./eval/finance/data/formula_test.json")
    log_lines.append("Online mode: Training and testing on 300 examples")
    log_lines.append("Using empty playbook as initial playbook")
    log_lines.append(f"Using MODAL API  ({MODAL_URL})")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("INITIAL TEST (before training)")
    log_lines.append("=" * 60)
    log_lines.append("Final Accuracy: 0.430 (129/300)")
    log_lines.append("")
    for w in range(1, 21):
        acc = ACE_FORMULA_WINDOW_ACC[w-1]
        correct = int(round(acc * 300))
        bc = ACE_FORMULA_BULLET_COUNTS[w-1]
        log_lines.append(f"=== WINDOW {w}/20 ===")
        log_lines.append(f"Final Accuracy: {acc:.3f} ({correct}/300)")
        log_lines.append(f"Window {w} accuracy: {acc:.3f} | Cumulative: {acc:.3f}")
        log_lines.append(f"Playbook size after window: {bc} bullets")
        if w >= 13:
            log_lines.append(f"[WARNING] Playbook size: {bc} bullets. Retrieval may suffer from embedding crowding.")
            log_lines.append(f"[CURATOR] Merged {random.randint(1,2)} near-duplicate bullets (cosine>0.91)")
    log_lines.append("")
    log_lines.append("Final Online Test Accuracy: 0.540")
    log_lines.append("=" * 60)
    log_lines.append("ACE RUN COMPLETE")
    log_lines.append("=" * 60)
    log_lines.append("Final Playbook: 126 bullets")
    log_lines.append("Avg latency: 11.9s/sample (gen=5.1s, ref=3.7s, cur=2.9s)")
    log_lines.append(f"Results saved to: {base_path}")
    log_lines.append("=" * 60)

    write_text(f"{base_path}/ace_formula_online_SYNTHETIC.log", "\n".join(log_lines))
    print(f"  [OK] ace_formula_online")


# ─────────────────────────────────────────────────────────────────────────────
# ACE FINER OFFLINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_ace_finer_offline():
    run_id = "ace_run_SYNTHETIC_20260312_112000_finer_offline"
    base_path = f"{BASE_DIR}/ace/ace_finer_offline/{run_id}"
    base_dt = datetime(2026, 3, 12, 11, 20, 0)

    run_config = {
        "synthetic_reference": True,
        "run_id": run_id,
        "task": "finer",
        "mode": "offline",
        "model": MODEL_NAME,
        "max_train_samples": 1000,
        "max_test_samples": 300,
        "num_epochs": 5,
        "system": "ACE",
        "modal_url": MODAL_URL,
        "started_at": base_dt.isoformat(),
        "completed_at": (base_dt + timedelta(hours=5, minutes=40)).isoformat(),
        "git_hash": "c99514f",
    }
    write_json(f"{base_path}/run_config.json", run_config)

    final_results = {
        "synthetic_reference": True,
        "accuracy": 0.600,
        "correct": 180,
        "total": 300,
        "skipped": 0,
        "initial_accuracy": 0.433,
        "initial_correct": 130,
        "epoch_accuracies": ACE_FINER_OFFLINE_EPOCH_ACC,
        "final_playbook_size": 210,
        "task": "finer",
        "mode": "offline",
        "model": MODEL_NAME,
    }
    write_json(f"{base_path}/final_results.json", final_results)

    progress = {"synthetic_reference": True, "total_epochs": 5, "completed_epochs": 5, "status": "complete"}
    write_json(f"{base_path}/progress.json", progress)

    log_lines = []
    log_lines.append("=" * 60)
    log_lines.append("ACE SYSTEM")
    log_lines.append("=" * 60)
    log_lines.append("Task: finer")
    log_lines.append("Mode: OFFLINE")
    log_lines.append(f"Generator Model: {MODEL_NAME}")
    log_lines.append("Train samples: 1000 | Test samples: 300 | Epochs: 5")
    log_lines.append("=" * 60)
    log_lines.append(f"Using MODAL API  ({MODAL_URL})")
    log_lines.append("")
    log_lines.append("INITIAL TEST: Accuracy: 0.433 (130/300)")
    log_lines.append("")
    for epoch in range(1, 6):
        acc = ACE_FINER_OFFLINE_EPOCH_ACC[epoch]
        correct = int(round(acc * 300))
        log_lines.append(f"=== EPOCH {epoch}/5 ===")
        log_lines.append(f"Training on 1000 samples...")
        for step in range(1, 101):
            if step % 50 == 0:
                log_lines.append(f"  Step {step}/1000: playbook updated")
        log_lines.append(f"Epoch {epoch} Test Accuracy: {acc:.3f} ({correct}/300)")
        log_lines.append(f"Playbook size after epoch {epoch}: {50 + epoch * 32} bullets")
    log_lines.append("")
    log_lines.append("Final Test Accuracy: 0.600 (180/300)")
    log_lines.append("=" * 60)
    log_lines.append("ACE RUN COMPLETE (OFFLINE)")
    log_lines.append("=" * 60)

    write_text(f"{base_path}/ace_finer_offline_SYNTHETIC.log", "\n".join(log_lines))
    print(f"  [OK] ace_finer_offline")


# ─────────────────────────────────────────────────────────────────────────────
# ACE FORMULA OFFLINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_ace_formula_offline():
    run_id = "ace_run_SYNTHETIC_20260312_152000_formula_offline"
    base_path = f"{BASE_DIR}/ace/ace_formula_offline/{run_id}"
    base_dt = datetime(2026, 3, 12, 15, 20, 0)

    run_config = {
        "synthetic_reference": True,
        "run_id": run_id,
        "task": "formula",
        "mode": "offline",
        "model": MODEL_NAME,
        "max_train_samples": 1000,
        "max_test_samples": 300,
        "num_epochs": 5,
        "system": "ACE",
        "modal_url": MODAL_URL,
        "started_at": base_dt.isoformat(),
        "completed_at": (base_dt + timedelta(hours=5, minutes=55)).isoformat(),
        "git_hash": "c99514f",
    }
    write_json(f"{base_path}/run_config.json", run_config)

    final_results = {
        "synthetic_reference": True,
        "accuracy": 0.700,
        "correct": 210,
        "total": 300,
        "skipped": 0,
        "initial_accuracy": 0.430,
        "initial_correct": 129,
        "epoch_accuracies": ACE_FORMULA_OFFLINE_EPOCH_ACC,
        "final_playbook_size": 185,
        "task": "formula",
        "mode": "offline",
        "model": MODEL_NAME,
    }
    write_json(f"{base_path}/final_results.json", final_results)

    progress = {"synthetic_reference": True, "total_epochs": 5, "completed_epochs": 5, "status": "complete"}
    write_json(f"{base_path}/progress.json", progress)

    log_lines = []
    log_lines.append("=" * 60)
    log_lines.append("ACE SYSTEM")
    log_lines.append("=" * 60)
    log_lines.append("Task: formula")
    log_lines.append("Mode: OFFLINE")
    log_lines.append(f"Generator Model: {MODEL_NAME}")
    log_lines.append("Train samples: 1000 | Test samples: 300 | Epochs: 5")
    log_lines.append("=" * 60)
    log_lines.append(f"Using MODAL API  ({MODAL_URL})")
    log_lines.append("")
    log_lines.append("INITIAL TEST: Accuracy: 0.430 (129/300)")
    log_lines.append("")
    for epoch in range(1, 6):
        acc = ACE_FORMULA_OFFLINE_EPOCH_ACC[epoch]
        correct = int(round(acc * 300))
        log_lines.append(f"=== EPOCH {epoch}/5 ===")
        log_lines.append(f"Training on 1000 samples...")
        log_lines.append(f"Epoch {epoch} Test Accuracy: {acc:.3f} ({correct}/300)")
        log_lines.append(f"Playbook size after epoch {epoch}: {45 + epoch * 28} bullets")
    log_lines.append("")
    log_lines.append("Final Test Accuracy: 0.700 (210/300)")
    log_lines.append("=" * 60)
    log_lines.append("ACE RUN COMPLETE (OFFLINE)")
    log_lines.append("=" * 60)

    write_text(f"{base_path}/ace_formula_offline_SYNTHETIC.log", "\n".join(log_lines))
    print(f"  [OK] ace_formula_offline")


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "gsam_no_ontology": {
        "finer_acc": 0.570, "finer_correct": 171,
        "formula_acc": 0.610, "formula_correct": 183,
        "use_ontology": False,
        "use_typed_edges": True,
        "use_cascades": True,
        "flag_label": "Ontology: DISABLED (ablation: no_ontology)",
        "finer_ts": "20260312_160000",
        "formula_ts": "20260312_161000",
        "finer_dt": datetime(2026, 3, 12, 16, 0, 0),
        "formula_dt": datetime(2026, 3, 12, 16, 10, 0),
        "description": "No taxonomy backbone; Stage 3 expansion disabled. BFS still works but concept disambiguation relies on learned strategies only.",
        "finer_window_acc": [0.433, 0.450, 0.463, 0.477, 0.487, 0.493, 0.497, 0.503, 0.507, 0.513, 0.517, 0.520, 0.523, 0.527, 0.537, 0.547, 0.553, 0.560, 0.563, 0.570],
        "formula_window_acc": [0.430, 0.447, 0.460, 0.473, 0.483, 0.490, 0.497, 0.503, 0.510, 0.517, 0.523, 0.530, 0.537, 0.543, 0.550, 0.557, 0.563, 0.573, 0.580, 0.610],
        "graph_note": "No is_a edges in graph. Graph contains only experiential nodes (Strategy, AntiPattern, Confusion) and concept nodes without taxonomy linkage.",
        "retrieval_precision_start": 0.12,
        "retrieval_precision_end": 0.18,
    },
    "gsam_no_cascades": {
        "finer_acc": 0.620, "finer_correct": 186,
        "formula_acc": 0.610, "formula_correct": 183,
        "use_ontology": True,
        "use_typed_edges": True,
        "use_cascades": False,
        "flag_label": "Failure Cascades: DISABLED (ablation: no_cascades)",
        "finer_ts": "20260312_162000",
        "formula_ts": "20260312_163000",
        "finer_dt": datetime(2026, 3, 12, 16, 20, 0),
        "formula_dt": datetime(2026, 3, 12, 16, 30, 0),
        "description": "Cascade propagation of failure signals across formula dependencies disabled. Anti-patterns created but not linked via fails_for->depends_on.",
        "finer_window_acc": [0.433, 0.453, 0.470, 0.487, 0.500, 0.510, 0.517, 0.523, 0.530, 0.537, 0.543, 0.550, 0.557, 0.563, 0.573, 0.583, 0.590, 0.600, 0.610, 0.620],
        "formula_window_acc": [0.430, 0.447, 0.460, 0.470, 0.480, 0.487, 0.493, 0.500, 0.507, 0.513, 0.520, 0.527, 0.533, 0.540, 0.550, 0.560, 0.573, 0.583, 0.600, 0.610],
        "graph_note": "Cascade edges (fails_for->depends_on propagation) absent. Graph has standard typed edges; formula dependency chains not propagated.",
        "retrieval_precision_start": 0.19,
        "retrieval_precision_end": 0.28,
    },
    "gsam_embedding_only": {
        "finer_acc": 0.550, "finer_correct": 165,
        "formula_acc": 0.590, "formula_correct": 177,
        "use_ontology": False,
        "use_typed_edges": False,
        "use_cascades": False,
        "flag_label": "Retrieval: EMBEDDING ONLY (ablation: embedding_only_retrieval)",
        "finer_ts": "20260312_164000",
        "formula_ts": "20260312_165000",
        "finer_dt": datetime(2026, 3, 12, 16, 40, 0),
        "formula_dt": datetime(2026, 3, 12, 16, 50, 0),
        "description": "Pure embedding similarity retrieval only. No BFS traversal, no taxonomy expansion. Embedding space crowds as graph grows.",
        "finer_window_acc": [0.433, 0.450, 0.463, 0.473, 0.483, 0.490, 0.497, 0.503, 0.510, 0.517, 0.523, 0.530, 0.535, 0.537, 0.540, 0.545, 0.548, 0.547, 0.548, 0.550],
        "formula_window_acc": [0.430, 0.447, 0.460, 0.470, 0.480, 0.490, 0.500, 0.510, 0.517, 0.523, 0.530, 0.537, 0.545, 0.550, 0.555, 0.558, 0.559, 0.557, 0.558, 0.590],
        "graph_note": "Retrieval uses cosine similarity only. Precision degrades as embedding space crowds with more nodes. BFS and Stage 3 taxonomy expansion completely bypassed.",
        "retrieval_precision_start": 0.15,
        "retrieval_precision_end": 0.08,  # degrades!
    },
    "gsam_untyped_edges": {
        "finer_acc": 0.600, "finer_correct": 180,
        "formula_acc": 0.630, "formula_correct": 189,
        "use_ontology": True,
        "use_typed_edges": False,
        "use_cascades": True,
        "flag_label": "Edge Types: UNTYPED (ablation: untyped_edges)",
        "finer_ts": "20260312_166000",
        "formula_ts": "20260312_167000",
        "finer_dt": datetime(2026, 3, 12, 16, 30, 0),
        "formula_dt": datetime(2026, 3, 12, 16, 45, 0),
        "description": "All edges treated as same type. BFS traversal works but is_a edges cause some sibling flooding. Less severe than Bug 10+11 combined.",
        "finer_window_acc": [0.433, 0.450, 0.467, 0.480, 0.493, 0.500, 0.507, 0.513, 0.520, 0.527, 0.533, 0.540, 0.547, 0.553, 0.563, 0.570, 0.577, 0.583, 0.590, 0.600],
        "formula_window_acc": [0.430, 0.447, 0.463, 0.477, 0.490, 0.497, 0.503, 0.510, 0.520, 0.527, 0.533, 0.540, 0.550, 0.557, 0.567, 0.573, 0.583, 0.597, 0.610, 0.630],
        "graph_note": "All edges stored as generic type. BFS cannot filter by edge type; occasional sibling flooding via is_a edges reduces retrieval precision.",
        "retrieval_precision_start": 0.16,
        "retrieval_precision_end": 0.20,
    },
    "gsam_no_multi_epoch": {
        "finer_acc": 0.625, "finer_correct": 188,
        "formula_acc": 0.635, "formula_correct": 191,
        "use_ontology": True,
        "use_typed_edges": True,
        "use_cascades": True,
        "flag_label": "Multi-Epoch Refinement: DISABLED (ablation: no_multi_epoch)",
        "finer_ts": "20260312_170000",
        "formula_ts": "20260312_171000",
        "finer_dt": datetime(2026, 3, 12, 17, 0, 0),
        "formula_dt": datetime(2026, 3, 12, 17, 10, 0),
        "description": "Online mode; multi-epoch refinement not applicable to online setting. Minimal impact as online mode is inherently single-pass.",
        "finer_window_acc": [0.433, 0.453, 0.473, 0.490, 0.503, 0.513, 0.520, 0.527, 0.537, 0.543, 0.550, 0.557, 0.563, 0.573, 0.583, 0.590, 0.600, 0.607, 0.617, 0.625],
        "formula_window_acc": [0.430, 0.450, 0.467, 0.483, 0.497, 0.507, 0.517, 0.527, 0.537, 0.543, 0.550, 0.560, 0.570, 0.577, 0.587, 0.597, 0.607, 0.613, 0.620, 0.635],
        "graph_note": "Full GSAM architecture. Multi-epoch not applicable in online mode; difference from Full GSAM arises from random sampling variation only.",
        "retrieval_precision_start": 0.21,
        "retrieval_precision_end": 0.31,
    },
}

GSAM_FULL_FINER_WINDOW_ACC = [
    0.433, 0.460, 0.480, 0.500, 0.517,
    0.527, 0.537, 0.547, 0.557, 0.567,
    0.577, 0.587, 0.597, 0.607, 0.617,
    0.623, 0.630, 0.633, 0.637, 0.640,
]
GSAM_FULL_FORMULA_WINDOW_ACC = [
    0.430, 0.457, 0.480, 0.500, 0.517,
    0.530, 0.540, 0.550, 0.560, 0.570,
    0.577, 0.587, 0.597, 0.607, 0.617,
    0.623, 0.630, 0.637, 0.643, 0.650,
]

def generate_ablation(ablation_name, task, run_ts, run_dt, config):
    run_id = f"gsam_run_SYNTHETIC_{run_ts}_{task}_online"
    base_path = f"{BASE_DIR}/ablations/{ablation_name}/{run_id}"

    is_finer = task == "finer"
    final_acc = config["finer_acc"] if is_finer else config["formula_acc"]
    final_correct = config["finer_correct"] if is_finer else config["formula_correct"]
    window_accs = config["finer_window_acc"] if is_finer else config["formula_window_acc"]
    prec_start = config["retrieval_precision_start"]
    prec_end = config["retrieval_precision_end"]

    # run_config.json
    run_config = {
        "synthetic_reference": True,
        "run_id": run_id,
        "task": task,
        "mode": "online",
        "model": MODEL_NAME,
        "max_samples": 300,
        "num_windows": 20,
        "window_size": 15,
        "max_rounds": 1,
        "num_epochs": 1,
        "use_ontology": config["use_ontology"],
        "use_typed_edges": config["use_typed_edges"],
        "use_cascades": config["use_cascades"],
        "ablation": ablation_name,
        "system": "GSAM",
        "modal_url": MODAL_URL,
        "started_at": run_dt.isoformat(),
        "completed_at": (run_dt + timedelta(hours=2, minutes=45)).isoformat(),
        "git_hash": "c99514f",
    }
    write_json(f"{base_path}/run_config.json", run_config)

    # final_results.json
    final_results = {
        "synthetic_reference": True,
        "online_test_results": {
            "accuracy": final_acc,
            "correct": final_correct,
            "total": 300,
            "skipped": 0,
        },
        "window_accuracies": window_accs,
        "initial_accuracy": 0.433 if is_finer else 0.430,
        "ablation": ablation_name,
        "ablation_description": config["description"],
        "task": task,
        "mode": "online",
        "model": MODEL_NAME,
    }
    write_json(f"{base_path}/final_results.json", final_results)

    # graph_stats.json
    graph_node_counts = {
        "Strategy": random.randint(55, 75),
        "AntiPattern": random.randint(30, 50),
        "Confusion": random.randint(40, 60),
        "Formula": random.randint(20, 40) if task == "formula" else 0,
        "Concept": 139 if config["use_ontology"] else 0,
    }
    graph_stats = {
        "synthetic_reference": True,
        "final_node_counts": graph_node_counts,
        "total_nodes": sum(graph_node_counts.values()),
        "total_edges": random.randint(200, 400),
        "is_a_edges": random.randint(100, 150) if config["use_ontology"] else 0,
        "applies_to_edges": random.randint(40, 80),
        "fails_for_edges": random.randint(20, 50) if config["use_cascades"] else 0,
        "confused_with_edges": random.randint(15, 35),
        "avg_retrieval_precision_window_1": prec_start,
        "avg_retrieval_precision_window_20": prec_end,
        "ablation": ablation_name,
        "note": config["graph_note"],
    }
    write_json(f"{base_path}/graph_stats.json", graph_stats)

    # partial results
    partial = {
        "synthetic_reference": True,
        "windows_completed": 20,
        "window_results": [
            {"window": i+1, "accuracy": window_accs[i],
             "correct": int(round(window_accs[i]*300)),
             "retrieval_precision": round(prec_start + (prec_end - prec_start) * i / 19, 3)}
            for i in range(20)
        ],
    }
    write_json(f"{base_path}/partial_online_results.json", partial)

    # Log file (~800 lines)
    log_lines = []
    log_lines.append("[INFO] Online mode requires num_epochs=1; overriding from 5 to 1")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("GSAM SYSTEM")
    log_lines.append("=" * 60)
    log_lines.append(f"Task: {task}")
    log_lines.append("Mode: ONLINE")
    log_lines.append(f"Generator Model: {MODEL_NAME}")
    log_lines.append("Max Samples: 300")
    log_lines.append(f"Ablation: {ablation_name}")
    log_lines.append(f"  {config['flag_label']}")
    log_lines.append("=" * 60)
    log_lines.append("")
    log_lines.append(f"Loaded 300 samples from ./eval/finance/data/{task}_test.json")
    log_lines.append("Online mode: Training and testing on 300 examples")
    if config["use_ontology"]:
        log_lines.append("Loading XBRL ontology... 139 concept nodes, 118 is_a edges")
    else:
        log_lines.append("Ontology loading SKIPPED (ablation: no_ontology)")
    log_lines.append(f"Using MODAL API  ({MODAL_URL})")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("INITIAL TEST (before training)")
    log_lines.append("=" * 60)
    init_acc = 0.433 if is_finer else 0.430
    init_correct = 130 if is_finer else 129
    log_lines.append(f"Final Accuracy: {init_acc:.3f} ({init_correct}/300)")
    log_lines.append("")

    elapsed_s = 800
    for w in range(1, 21):
        window_idx = w - 1
        acc = window_accs[window_idx]
        correct = int(round(acc * 300))
        prec = round(prec_start + (prec_end - prec_start) * window_idx / 19, 3)

        log_lines.append("=" * 60)
        log_lines.append(f"WINDOW {w}/20 (samples {(w-1)*15}-{w*15-1})")
        log_lines.append("=" * 60)
        log_lines.append("")
        log_lines.append("=" * 40)
        log_lines.append("EVALUATING TEST SET - 300 samples, 20 workers")
        log_lines.append("=" * 40)
        log_lines.append(f"Final Accuracy: {acc:.3f} ({correct}/300)")
        log_lines.append(f"Window {w} accuracy: {acc:.3f} | Cumulative: {acc:.3f}")
        log_lines.append(f"Retrieved: 30 nodes | Precision: {prec:.3f}")

        # Embedding only warning
        if ablation_name == "gsam_embedding_only" and w >= 12:
            log_lines.append(f"[WARNING] Embedding space crowding detected. {graph_node_counts['Strategy']+graph_node_counts['AntiPattern']} experiential nodes in graph.")
            log_lines.append(f"[INFO] Retrieval precision degraded from {prec_start:.2f} to {prec:.2f} as graph grew.")
        if ablation_name == "gsam_no_ontology" and w == 5:
            log_lines.append("[INFO] Stage 3 (taxonomy expansion) skipped — ontology disabled.")
        if ablation_name == "gsam_untyped_edges" and w >= 8:
            n_siblings = random.randint(3, 8)
            log_lines.append(f"[INFO] Untyped BFS retrieved {n_siblings} is_a-sibling nodes (mild sibling flooding).")

        # Steps
        for step_in_w in range(1, 16):
            global_step = (w - 1) * 15 + step_in_w
            is_correct = random.random() < acc

            log_lines.append(f"--- Window {w}, Step {step_in_w}/15 (Global {global_step}) ---")
            log_lines.append(f"[GENERATOR] Starting call online_train_s_{global_step}_gen_initial...")
            latency = round(random.uniform(8.0, 14.0), 2)
            log_lines.append(f"[GENERATOR] Call online_train_s_{global_step}_gen_initial completed in {latency}s")
            log_lines.append(f"Correct: {is_correct}")

            if not is_correct:
                log_lines.append("Reflection round 1/1")
                log_lines.append(f"[REFLECTOR] Starting call online_train_s_{global_step}_round_0...")
                ref_lat = round(random.uniform(7.0, 12.0), 2)
                log_lines.append(f"[REFLECTOR] Call online_train_s_{global_step}_round_0 completed in {ref_lat}s")
                log_lines.append(f"[GRAPH_CONSTRUCTOR] Starting call online_train_s_{global_step}...")
                gc_lat = round(random.uniform(6.0, 10.0), 2)
                log_lines.append(f"[GRAPH_CONSTRUCTOR] Call online_train_s_{global_step} completed in {gc_lat}s")
                n_ops = random.randint(1, 3)
                log_lines.append(f"Applied {n_ops} graph operation{'s' if n_ops != 1 else ''} (ADD_STRATEGY={n_ops})")

        n_strat = graph_node_counts['Strategy']
        n_anti = graph_node_counts['AntiPattern']
        n_conf = graph_node_counts['Confusion']
        n_concept = graph_node_counts['Concept']
        log_lines.append(f"Graph state: Strategy={n_strat}, AntiPattern={n_anti}, Confusion={n_conf}, Concept={n_concept}")

        if ablation_name == "gsam_no_multi_epoch" and w == 10:
            log_lines.append("[INFO] Multi-epoch refinement not applicable in online mode. Skipping epoch consolidation.")

    log_lines.append("")
    log_lines.append(f"Final Online Test Accuracy: {final_acc:.3f}")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("GSAM RUN COMPLETE")
    log_lines.append("=" * 60)
    log_lines.append(f"Ablation: {ablation_name}")
    log_lines.append(f"  {config['flag_label']}")
    log_lines.append(f"Final accuracy: {final_acc:.3f} ({final_correct}/300)")
    log_lines.append(f"Graph: {sum(graph_node_counts.values())} nodes")
    log_lines.append(f"Avg retrieval precision: {round((prec_start + prec_end) / 2, 3)}")
    log_lines.append(f"Results saved to: {base_path}")
    log_lines.append("=" * 60)

    log_name = f"gsam_{ablation_name}_{task}_online_SYNTHETIC.log"
    write_text(f"{base_path}/{log_name}", "\n".join(log_lines))
    print(f"  [OK] {ablation_name}/{task}: {len(log_lines)} log lines")


def generate_all_ablations():
    for abl_name, config in ABLATION_CONFIGS.items():
        generate_ablation(abl_name, "finer", config["finer_ts"], config["finer_dt"], config)
        generate_ablation(abl_name, "formula", config["formula_ts"], config["formula_dt"], config)


# ─────────────────────────────────────────────────────────────────────────────
# FiNER-Transfer
# ─────────────────────────────────────────────────────────────────────────────

SIBLING_PAIRS = [
    ("RevenueFromContractWithCustomerExcludingAssessedTax", "ContractWithCustomerLiabilityRevenueRecognized", 2, 0.40, 0.53, 0.133, True),
    ("NetIncomeLoss", "ComprehensiveIncomeNetOfTax", 2, 0.35, 0.47, 0.120, True),
    ("DebtInstrumentFaceAmount", "DebtInstrumentCarryingAmount", 2, 0.45, 0.57, 0.120, True),
    ("CommonStockSharesAuthorized", "CommonStockSharesOutstanding", 2, 0.50, 0.57, 0.067, True),
    ("AllocatedShareBasedCompensationExpense", "EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized", 3, 0.30, 0.47, 0.167, True),
    ("LineOfCreditFacilityMaximumBorrowingCapacity", "LineOfCreditFacilityCurrentBorrowingCapacity", 2, 0.55, 0.67, 0.120, True),
    ("NumberOfOperatingSegments", "NumberOfReportableSegments", 2, 0.60, 0.73, 0.133, True),
    ("Depreciation", "AmortizationOfIntangibleAssets", 3, 0.42, 0.50, 0.083, True),
    ("PropertyPlantAndEquipmentNet", "PropertyPlantAndEquipmentGross", 2, 0.47, 0.57, 0.100, True),
    ("InterestExpense", "InterestPaid", 2, 0.52, 0.60, 0.080, True),
    ("OperatingLeaseExpense", "FinanceLeaseExpense", 3, 0.38, 0.45, 0.067, True),
    ("LesseeOperatingLeaseTermOfContract", "LesseeFinanceLeaseTermOfContract", 3, 0.42, 0.50, 0.080, True),
    ("EarningsPerShareBasic", "EarningsPerShareDiluted", 2, 0.48, 0.57, 0.090, True),
    ("RetainedEarningsAccumulatedDeficit", "AdditionalPaidInCapital", 3, 0.43, 0.50, 0.067, True),
    ("Goodwill", "GoodwillImpairmentLoss", 2, 0.37, 0.43, 0.060, True),
    ("RestructuringCharges", "BusinessCombinationConsiderationTransferred1", 4, 0.33, 0.40, 0.067, True),
    ("DeferredFinanceCostsNet", "DeferredFinanceCostsGross", 2, 0.48, 0.57, 0.087, True),
    ("DebtInstrumentInterestRateStatedPercentage", "DebtInstrumentInterestRateEffectivePercentage", 2, 0.45, 0.53, 0.080, True),
    ("ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1", "SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage", 2, 0.40, 0.47, 0.067, True),
    ("EquityMethodInvestmentOwnershipPercentage", "MinorityInterestOwnershipPercentageByParent", 3, 0.37, 0.43, 0.067, True),
    ("DefinedBenefitPlanContributionsByEmployer", "DefinedContributionPlanCostRecognized", 3, 0.33, 0.40, 0.067, True),
    ("AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount", "ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1", 4, 0.28, 0.33, 0.050, True),
    ("ConcentrationRiskPercentage1", "NumberOfRealEstateProperties", 5, 0.32, 0.37, 0.050, True),
    ("LossContingencyEstimateOfPossibleLoss", "LettersOfCreditOutstandingAmount", 4, 0.30, 0.35, 0.050, True),
    ("StockRepurchaseProgramAuthorizedAmount1", "StockRepurchasedAndRetiredDuringPeriodShares", 3, 0.35, 0.40, 0.050, True),
    ("SaleOfStockPricePerShare", "SaleOfStockNumberOfSharesIssuedInTransaction", 2, 0.40, 0.45, 0.050, True),
    ("PublicUtilitiesRequestedRateIncreaseDecreaseAmount", "GuaranteeObligationsMaximumExposure", 5, 0.27, 0.32, 0.050, True),
    ("OperatingLeaseWeightedAverageRemainingLeaseTerm1", "OperatingLeaseWeightedAverageDiscountRatePercent", 2, 0.38, 0.38, 0.000, False),
    ("DebtInstrumentUnamortizedDiscount", "DebtInstrumentBasisSpreadOnVariableRate1", 3, 0.42, 0.42, 0.000, False),
    ("RevenueRemainingPerformanceObligation", "ContractWithCustomerAsset1Net", 3, 0.30, 0.30, 0.000, False),
    ("CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalentsFairValueDisclosure", 2, 0.53, 0.53, 0.000, False),
    ("CommonStockDividendsPerShareDeclared", "PreferredStockDividendsPerShareDeclared", 2, 0.43, 0.43, 0.000, False),
    ("IncomeTaxExpenseBenefit", "EffectiveIncomeTaxRateContinuingOperations", 2, 0.47, 0.47, 0.000, False),
    ("BusinessCombinationAcquisitionRelatedCosts", "BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill", 2, 0.28, 0.28, 0.000, False),
    ("UnrecognizedTaxBenefits", "UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate", 2, 0.35, 0.35, 0.000, False),
    ("ProceedsFromIssuanceOfCommonStock", "StockIssuedDuringPeriodSharesNewIssues", 2, 0.45, 0.45, 0.000, False),
    ("ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber", "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue", 2, 0.32, 0.32, 0.000, False),
    ("DebtInstrumentRedemptionPricePercentage", "DebtInstrumentConvertibleConversionPrice1", 3, 0.38, 0.35, -0.033, False),
    ("DerivativeNotionalAmount", "DebtInstrumentFairValue", 4, 0.30, 0.27, -0.033, False),
    ("AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife", "FiniteLivedIntangibleAssetUsefulLife", 2, 0.40, 0.37, -0.033, False),
    # Pairs 41-42: fill to 42
    ("RevenueFromContractWithCustomerIncludingAssessedTax", "RevenueFromContractWithCustomerExcludingAssessedTax", 2, 0.37, 0.43, 0.060, True),
    ("NetIncomeLossAttributableToNoncontrollingInterest", "NetIncomeLoss", 2, 0.32, 0.38, 0.060, True),
]

def build_sibling_pair_results_gsam():
    results = []
    for src, tgt, lca, base_acc, transfer_acc, gain, positive in SIBLING_PAIRS:
        results.append({
            "pair_type": "sibling",
            "source_concept": src,
            "target_concept": tgt,
            "lca_depth": lca,
            "base_acc": base_acc,
            "transfer_acc": transfer_acc,
            "transfer_gain": gain,
            "positive": positive,
        })
    return results

def build_distant_pair_results_gsam():
    results = []
    # 9 positive, 29 zero, 4 negative = 42 total
    concept_pool = [s[0] for s in SIBLING_PAIRS[:20]]
    pairs_used = set()
    positive_distant = [
        ("NetIncomeLoss", "LineOfCreditFacilityMaximumBorrowingCapacity", 0.43, 0.45, 0.017, True),
        ("AllocatedShareBasedCompensationExpense", "Goodwill", 0.38, 0.40, 0.017, True),
        ("Depreciation", "StockRepurchaseProgramAuthorizedAmount1", 0.42, 0.43, 0.013, True),
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "PropertyPlantAndEquipmentNet", 0.45, 0.47, 0.017, True),
        ("DebtInstrumentFaceAmount", "NumberOfOperatingSegments", 0.60, 0.62, 0.017, True),
        ("CommonStockSharesAuthorized", "Goodwill", 0.35, 0.37, 0.017, True),
        ("InterestExpense", "ConcentrationRiskPercentage1", 0.30, 0.32, 0.017, True),
        ("EarningsPerShareBasic", "RestructuringCharges", 0.35, 0.37, 0.017, True),
        ("LossContingencyEstimateOfPossibleLoss", "ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1", 0.28, 0.30, 0.017, True),
    ]
    for src, tgt, base, trans, gain, pos in positive_distant:
        results.append({"pair_type": "distant", "source_concept": src, "target_concept": tgt,
                        "subtree_distance": "high", "base_acc": base, "transfer_acc": trans,
                        "transfer_gain": gain, "positive": pos})
    # 29 zero-transfer
    zero_pairs = [
        ("RetainedEarningsAccumulatedDeficit", "OperatingLeaseExpense"),
        ("EarningsPerShareDiluted", "Depreciation"),
        ("GoodwillImpairmentLoss", "InterestPaid"),
        ("DeferredFinanceCostsNet", "AllocatedShareBasedCompensationExpense"),
        ("SaleOfStockPricePerShare", "NetIncomeLoss"),
        ("EquityMethodInvestmentOwnershipPercentage", "PropertyPlantAndEquipmentGross"),
        ("IncomeTaxExpenseBenefit", "Goodwill"),
        ("UnrecognizedTaxBenefits", "RestructuringCharges"),
        ("ProceedsFromIssuanceOfCommonStock", "DebtInstrumentFaceAmount"),
        ("NumberOfReportableSegments", "RetainedEarningsAccumulatedDeficit"),
        ("LesseeFinanceLeaseTermOfContract", "CommonStockSharesOutstanding"),
        ("OperatingLeaseWeightedAverageRemainingLeaseTerm1", "AllocatedShareBasedCompensationExpense"),
        ("DebtInstrumentBasisSpreadOnVariableRate1", "EarningsPerShareBasic"),
        ("AmortizationOfIntangibleAssets", "SaleOfStockPricePerShare"),
        ("AdditionalPaidInCapital", "LossContingencyEstimateOfPossibleLoss"),
        ("FinanceLeaseExpense", "NumberOfOperatingSegments"),
        ("StockRepurchasedAndRetiredDuringPeriodShares", "DeferredFinanceCostsNet"),
        ("DebtInstrumentInterestRateEffectivePercentage", "OperatingLeaseExpense"),
        ("MinorityInterestOwnershipPercentageByParent", "InterestExpense"),
        ("DefinedContributionPlanCostRecognized", "Goodwill"),
        ("AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount", "PropertyPlantAndEquipmentNet"),
        ("NumberOfRealEstateProperties", "DebtInstrumentFaceAmount"),
        ("GuaranteeObligationsMaximumExposure", "EarningsPerShareDiluted"),
        ("PublicUtilitiesRequestedRateIncreaseDecreaseAmount", "Depreciation"),
        ("LettersOfCreditOutstandingAmount", "CommonStockSharesAuthorized"),
        ("StockRepurchaseProgramAuthorizedAmount1", "InterestPaid"),
        ("SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage", "NetIncomeLoss"),
        ("ConcentrationRiskPercentage1", "RetainedEarningsAccumulatedDeficit"),
        ("ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1", "AllocatedShareBasedCompensationExpense"),
    ]
    for src, tgt in zero_pairs:
        base = round(random.uniform(0.30, 0.55), 2)
        results.append({"pair_type": "distant", "source_concept": src, "target_concept": tgt,
                        "subtree_distance": "high", "base_acc": base, "transfer_acc": base,
                        "transfer_gain": 0.000, "positive": False})
    # 4 negative
    neg_pairs = [
        ("GoodwillImpairmentLoss", "RevenueFromContractWithCustomerExcludingAssessedTax", 0.32, 0.28, -0.033),
        ("DebtInstrumentConvertibleConversionPrice1", "NumberOfOperatingSegments", 0.38, 0.35, -0.033),
        ("FiniteLivedIntangibleAssetUsefulLife", "CommonStockSharesAuthorized", 0.40, 0.37, -0.033),
        ("AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife", "NetIncomeLoss", 0.35, 0.32, -0.033),
    ]
    for src, tgt, base, trans, gain in neg_pairs:
        results.append({"pair_type": "distant", "source_concept": src, "target_concept": tgt,
                        "subtree_distance": "high", "base_acc": base, "transfer_acc": trans,
                        "transfer_gain": gain, "positive": False})
    return results

def build_sibling_pair_results_ace():
    """ACE version: fewer positive (11/42), more negative (8/42), smaller gains."""
    results = []
    positive_count = 0
    negative_count = 0
    for i, (src, tgt, lca, base_acc, gsam_transfer_acc, gsam_gain, gsam_positive) in enumerate(SIBLING_PAIRS):
        if gsam_positive and positive_count < 11:
            # ACE positive with smaller gain
            gain = round(gsam_gain * 0.35, 3)
            trans = round(base_acc + gain, 3)
            results.append({
                "pair_type": "sibling", "source_concept": src, "target_concept": tgt,
                "lca_depth": lca, "base_acc": base_acc, "transfer_acc": trans,
                "transfer_gain": gain, "positive": True,
            })
            positive_count += 1
        elif not gsam_positive and negative_count < 8:
            # ACE makes more mistakes -> negative transfer
            gain = round(-random.uniform(0.020, 0.067), 3)
            trans = round(base_acc + gain, 3)
            results.append({
                "pair_type": "sibling", "source_concept": src, "target_concept": tgt,
                "lca_depth": lca, "base_acc": base_acc, "transfer_acc": trans,
                "transfer_gain": gain, "positive": False,
            })
            negative_count += 1
        else:
            # zero transfer
            results.append({
                "pair_type": "sibling", "source_concept": src, "target_concept": tgt,
                "lca_depth": lca, "base_acc": base_acc, "transfer_acc": base_acc,
                "transfer_gain": 0.000, "positive": False,
            })
    return results

def build_distant_pair_results_ace():
    """ACE: 6/42 positive near-transfer, mostly zero."""
    gsam_distant = build_distant_pair_results_gsam()
    results = []
    pos_count = 0
    for entry in gsam_distant:
        e = dict(entry)
        if e["positive"] and pos_count < 6:
            e["transfer_gain"] = round(e["transfer_gain"] * 0.5, 3)
            e["transfer_acc"] = round(e["base_acc"] + e["transfer_gain"], 3)
            pos_count += 1
        else:
            e["transfer_gain"] = 0.000
            e["transfer_acc"] = e["base_acc"]
            e["positive"] = False
        results.append(e)
    return results

def generate_finer_transfer():
    transfer_dir = f"{BASE_DIR}/finer_transfer"

    # GSAM transfer results
    gsam_results = {
        "synthetic_reference": True,
        "method": "GSAM",
        "n_sibling_pairs": 42,
        "n_distant_pairs": 42,
        "near_transfer_rate": 0.643,
        "far_transfer_rate": 0.214,
        "transfer_precision": 0.062,
        "negative_transfer_rate": 0.071,
        "pairs_with_positive_near_transfer": 27,
        "pairs_with_zero_near_transfer": 12,
        "pairs_with_negative_near_transfer": 3,
        "pairs_with_positive_far_transfer": 9,
        "pairs_with_zero_far_transfer": 29,
        "pairs_with_negative_far_transfer": 4,
        "sibling_pair_results": build_sibling_pair_results_gsam(),
        "distant_pair_results": build_distant_pair_results_gsam(),
    }
    write_json(f"{transfer_dir}/gsam_transfer_results.json", gsam_results)

    # ACE transfer results
    ace_sib = build_sibling_pair_results_ace()
    ace_distant = build_distant_pair_results_ace()
    ace_pos_near = sum(1 for r in ace_sib if r["positive"])
    ace_neg_near = sum(1 for r in ace_sib if r["transfer_gain"] < 0)
    ace_zero_near = 42 - ace_pos_near - ace_neg_near
    ace_pos_far = sum(1 for r in ace_distant if r["positive"])
    ace_neg_far = sum(1 for r in ace_distant if r["transfer_gain"] < 0)
    ace_zero_far = 42 - ace_pos_far - ace_neg_far

    ace_results = {
        "synthetic_reference": True,
        "method": "ACE",
        "n_sibling_pairs": 42,
        "n_distant_pairs": 42,
        "near_transfer_rate": round(ace_pos_near / 42, 3),
        "far_transfer_rate": round(ace_pos_far / 42, 3),
        "transfer_precision": 0.038,
        "negative_transfer_rate": round(ace_neg_near / 42, 3),
        "pairs_with_positive_near_transfer": ace_pos_near,
        "pairs_with_zero_near_transfer": ace_zero_near,
        "pairs_with_negative_near_transfer": ace_neg_near,
        "pairs_with_positive_far_transfer": ace_pos_far,
        "pairs_with_zero_far_transfer": ace_zero_far,
        "pairs_with_negative_far_transfer": ace_neg_far,
        "sibling_pair_results": ace_sib,
        "distant_pair_results": ace_distant,
    }
    write_json(f"{transfer_dir}/ace_transfer_results.json", ace_results)

    # transfer_summary.json
    summary = {
        "synthetic_reference": True,
        "comparison": {
            "near_transfer_rate": {"ACE": round(ace_pos_near / 42, 3), "GSAM": 0.643, "improvement": round(0.643 / max(ace_pos_near / 42, 0.001), 2)},
            "far_transfer_rate": {"ACE": round(ace_pos_far / 42, 3), "GSAM": 0.214},
            "transfer_precision": {"ACE": 0.038, "GSAM": 0.062},
            "negative_transfer_rate": {"ACE": round(ace_neg_near / 42, 3), "GSAM": 0.071,
                                        "reduction": round(1 - 0.071 / max(ace_neg_near / 42, 0.001), 3)},
        }
    }
    write_json(f"{transfer_dir}/transfer_summary.json", summary)

    # finer_transfer_SYNTHETIC.log
    log_lines = []
    log_lines.append("=" * 60)
    log_lines.append("FiNER-Transfer Benchmark")
    log_lines.append("=" * 60)
    log_lines.append("Method: GSAM")
    log_lines.append("Sibling pairs (n=42): defined by XBRL parent-child taxonomy structure")
    log_lines.append("Distant pairs (n=42): cross-subtree pairs (e.g., IncomeStatement <-> BalanceSheet concepts)")
    log_lines.append("Protocol: Adapt on source concept examples -> eval on target concept (no additional adaptation)")
    log_lines.append("=" * 60)
    log_lines.append("")

    for i, row in enumerate(gsam_results["sibling_pair_results"]):
        src = row["source_concept"]
        tgt = row["target_concept"]
        gain = row["transfer_gain"]
        sign = "+POSITIVE" if row["positive"] else ("NEGATIVE" if gain < 0 else "ZERO")
        n_strats = random.randint(1, 4)
        n_examples = random.randint(5, 10)
        base_10 = int(round(row["base_acc"] * 10))
        trans_15 = int(round(row["transfer_acc"] * 15))
        log_lines.append(f"Processing sibling pair {i+1}/42: {src} -> {tgt}")
        log_lines.append(f"  Source adaptation: {n_examples} examples -> {n_strats} strategies learned")
        log_lines.append(f"  Target base accuracy: {row['base_acc']*100:.1f}% ({base_10}/10)")
        log_lines.append(f"  Target transfer accuracy: {row['transfer_acc']*100:.1f}% ({trans_15}/15)")
        log_lines.append(f"  Transfer gain: {gain*100:+.1f}pp [{sign}]")
        log_lines.append("")

    log_lines.append("=" * 60)
    log_lines.append("GSAM Near-Transfer Summary (Sibling Pairs)")
    log_lines.append("=" * 60)
    log_lines.append(f"Positive transfer: 27/42 = 64.3%")
    log_lines.append(f"Zero transfer:     12/42 = 28.6%")
    log_lines.append(f"Negative transfer:  3/42 =  7.1%")
    log_lines.append("")
    log_lines.append("Processing 42 distant pairs...")
    for i, row in enumerate(gsam_results["distant_pair_results"][:10]):
        src = row["source_concept"]
        tgt = row["target_concept"]
        gain = row["transfer_gain"]
        log_lines.append(f"Processing distant pair {i+1}/42: {src} -> {tgt}")
        log_lines.append(f"  Transfer gain: {gain*100:+.1f}pp")
    log_lines.append("  ... (remaining 32 distant pairs processed)")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("GSAM Far-Transfer Summary (Distant Pairs)")
    log_lines.append("=" * 60)
    log_lines.append("Positive transfer:  9/42 = 21.4%")
    log_lines.append("Zero transfer:     29/42 = 69.0%")
    log_lines.append("Negative transfer:  4/42 =  9.5%")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("ACE Comparison")
    log_lines.append("=" * 60)
    log_lines.append(f"ACE Near-Transfer Rate: {round(ace_pos_near/42*100, 1)}%  vs  GSAM: 64.3%")
    log_lines.append(f"ACE Far-Transfer Rate:  {round(ace_pos_far/42*100, 1)}%  vs  GSAM: 21.4%")
    log_lines.append(f"ACE Negative Rate:      {round(ace_neg_near/42*100, 1)}%  vs  GSAM:  7.1%")
    log_lines.append("")
    log_lines.append("GSAM achieves 2.45x better near-transfer rate vs ACE.")
    log_lines.append("Key mechanism: graph topology preserves concept relationships;")
    log_lines.append("strategies learned for source concept propagate via is_a edges to target.")

    write_text(f"{transfer_dir}/finer_transfer_SYNTHETIC.log", "\n".join(log_lines))
    print(f"  [OK] finer_transfer: {len(log_lines)} log lines")


# ─────────────────────────────────────────────────────────────────────────────
# README
# ─────────────────────────────────────────────────────────────────────────────

def generate_readme():
    readme = """# Synthetic Reference Results — clean_results/

> **WARNING: These are SYNTHETIC REFERENCE results, NOT real experimental data.**
> Every JSON file contains `"synthetic_reference": true`.
> These files exist to:
> 1. Provide thesis table scaffolding (Tables 2-6) with realistic expected values
> 2. Serve as regression targets to verify real runs are in the right ballpark
> 3. Demonstrate the exact file structure produced by real ACE/GSAM runs

---

## Summary Tables

### Table 2: Main Results (FiNER, Online)

| System        | FiNER Acc | Formula Acc | Notes |
|---------------|-----------|-------------|-------|
| Base LLM      | 43.3%     | 43.0%       | 7B Q4 DeepSeek, no memory |
| ACE           | 53.0%     | 54.0%       | Flat playbook, 148 bullets |
| GSAM (Full)   | 64.0%     | 65.0%       | Graph memory, all features |

### Table 3: Offline Results (5 epochs, 1000 train samples)

| System        | FiNER Acc | Formula Acc |
|---------------|-----------|-------------|
| Base LLM      | 43.3%     | 43.0%       |
| ACE           | 60.0%     | 70.0%       |

### Table 4: GSAM Ablations (Online, 300 samples)

| Ablation          | FiNER  | Formula | Avg    | Drop vs Full |
|-------------------|--------|---------|--------|--------------|
| GSAM Full         | 64.0%  | 65.0%   | 64.5%  | —            |
| no_ontology       | 57.0%  | 61.0%   | 59.0%  | -5.5pp       |
| no_cascades       | 62.0%  | 61.0%   | 61.5%  | -3.0pp       |
| embedding_only    | 55.0%  | 59.0%   | 57.0%  | -7.5pp       |
| untyped_edges     | 60.0%  | 63.0%   | 61.5%  | -3.0pp       |
| no_multi_epoch    | 62.5%  | 63.5%   | 63.0%  | -1.5pp       |

### Table 5: FiNER-Transfer (Near and Far Transfer)

| Method | Near-Transfer Rate | Far-Transfer Rate | Neg Transfer | Transfer Precision |
|--------|--------------------|-------------------|--------------|--------------------|
| ACE    | ~26.2%             | ~14.3%            | ~19.0%       | 3.8%               |
| GSAM   | 64.3%              | 21.4%             | 7.1%         | 6.2%               |

### Table 6: Playbook Growth vs Graph Growth (ACE vs GSAM)

| Window | ACE Bullets | ACE Acc | GSAM Nodes | GSAM Acc |
|--------|-------------|---------|------------|----------|
| 1      | 12          | 43.3%   | ~15        | 43.3%    |
| 5      | 54          | 49.7%   | ~45        | 51.7%    |
| 10     | 96          | 52.7%   | ~120       | 56.7%    |
| 15     | 128         | 53.0%   | ~180       | 61.7%    |
| 20     | 148         | 53.0%   | ~220       | 64.0%    |

---

## Directory Structure

```
clean_results/
  ace/
    ace_finer_online/       ace_run_SYNTHETIC_20260312_092000_finer_online/
    ace_formula_online/     ace_run_SYNTHETIC_20260312_132000_formula_online/
    ace_finer_offline/      ace_run_SYNTHETIC_20260312_112000_finer_offline/
    ace_formula_offline/    ace_run_SYNTHETIC_20260312_152000_formula_offline/
  ablations/
    gsam_no_ontology/       (finer + formula runs)
    gsam_no_cascades/       (finer + formula runs)
    gsam_embedding_only/    (finer + formula runs)
    gsam_untyped_edges/     (finer + formula runs)
    gsam_no_multi_epoch/    (finer + formula runs)
  finer_transfer/
    gsam_transfer_results.json
    ace_transfer_results.json
    transfer_summary.json
    finer_transfer_SYNTHETIC.log
  README.md  (this file)
```

---

## Key Files in Each ACE Run Directory

| File | Description |
|------|-------------|
| `run_config.json` | All hyperparameters and settings |
| `final_results.json` | Top-level accuracy, window accuracies, playbook size |
| `partial_online_results.json` | Per-window accuracy + playbook size progression |
| `progress.json` | Run completion status |
| `bullet_usage_log.jsonl` | 300 entries; shows which bullets were retrieved each step |
| `curator_operations_diff.jsonl` | 300 entries; ADD/UPDATE/MERGE/DELETE operations |
| `intermediate_playbooks/` | Playbook snapshots at windows 5, 10, 20 |
| `detailed_llm_logs/` | 30 representative LLM call logs |
| `*.log` | Full ~1800-line run log |

---

## Key Files in Each GSAM Ablation Run Directory

| File | Description |
|------|-------------|
| `run_config.json` | Hyperparameters + ablation flags |
| `final_results.json` | `online_test_results.accuracy` (nested structure, NOT flat) |
| `partial_online_results.json` | Per-window accuracy + retrieval precision |
| `graph_stats.json` | Node counts by type, edge counts, retrieval precision start/end |
| `*.log` | ~800-line run log |

> **CRITICAL**: GSAM `final_results.json` uses NESTED structure:
> - Online: `result["online_test_results"]["accuracy"]`
> - ACE uses flat: `result["accuracy"]`

---

## What to Look for in the Logs

### ACE Log Signature
- `Playbook: X bullets -> Y bullets (added Z)` — grows each step
- `[WARNING] Playbook size: N bullets. Retrieval may suffer from embedding crowding.` — appears at windows 12-15
- `[CURATOR] Merged N near-duplicate bullets (cosine>0.91)` — deduplication passes
- NO `Retrieved: N nodes | Precision: X.XX` line (ACE has no graph retrieval)
- NO `[GRAPH_CONSTRUCTOR]` calls

### GSAM Log Signature (full or ablation)
- `Retrieved: 30 nodes | Precision: X.XXX` — per-window retrieval quality
- `[GRAPH_CONSTRUCTOR] Starting call ...` — graph update after reflection
- `Applied N graph operations (ADD_STRATEGY=N)` — graph growth
- `Graph state: Strategy=N, AntiPattern=N, Confusion=N, Concept=N` — graph snapshot
- NO `Playbook: X bullets` line (GSAM has no playbook)

### Ablation-Specific Log Markers
- `no_ontology`: `Ontology loading SKIPPED (ablation: no_ontology)`, `Stage 3 (taxonomy expansion) skipped`
- `no_cascades`: `Failure Cascades: DISABLED (ablation: no_cascades)`, `Cascade edges (fails_for->depends_on propagation) absent`
- `embedding_only`: `Retrieval: EMBEDDING ONLY`, `Embedding space crowding detected`, precision degrades window 1->20
- `untyped_edges`: `Edge Types: UNTYPED`, `Untyped BFS retrieved N is_a-sibling nodes (mild sibling flooding)`
- `no_multi_epoch`: `Multi-epoch refinement not applicable in online mode. Skipping epoch consolidation.`

---

## GSAM Improvements Explained

### 1. Graph Growth vs Playbook Crowding
ACE stores all knowledge in a flat list of bullets. As the playbook grows beyond ~100 bullets,
embedding-based retrieval degrades (cosine space crowding) — the system must retrieve K bullets
from N>>K bullets, and similar-looking but semantically different bullets crowd each other out.

GSAM uses a graph structure where each node has typed edges. Retrieval uses 3 stages:
1. Concept ID matching (exact node lookup)
2. BFS over experiential edges only (no is_a flooding)
3. Taxonomy expansion via is_a edges (Stage 3)

This means related knowledge is retrieved structurally, not just by embedding similarity.

### 2. Retrieval Precision
ACE: embedding cosine similarity -> precision ~5-8% (out of 148 bullets, ~8 are referenced by generator)
GSAM Full: 3-stage retrieval -> precision ~21-31% (out of 30 nodes, ~7-9 are referenced)

### 3. Knowledge Transfer
GSAM's graph topology enables transfer: a strategy learned for `DebtInstrumentFaceAmount`
is linked via `is_a` edges to the parent concept `DebtInstrument`, which also links to
`DebtInstrumentCarryingAmount`. When queried about carrying amount, the strategy propagates.

ACE has no such structure; strategies are stored as flat bullets with no relational links.

---

## Comparing Real Results to These References

```python
import json, os

def load_results(path):
    with open(path) as f:
        return json.load(f)

# Compare real ACE FiNER online result
real = load_results("results/ace_finer_online/final_results.json")
ref  = load_results("clean_results/ace/ace_finer_online/ace_run_SYNTHETIC_20260312_092000_finer_online/final_results.json")

real_acc = real.get("accuracy")  # ACE uses flat structure
ref_acc  = ref.get("accuracy")
print(f"Real: {real_acc:.3f} | Reference: {ref_acc:.3f} | Delta: {real_acc - ref_acc:+.3f}")

# Compare real GSAM ablation
real_gsam = load_results("results/ablations/gsam_no_ontology/final_results.json")
ref_gsam  = load_results("clean_results/ablations/gsam_no_ontology/gsam_run_SYNTHETIC_20260312_160000_finer_online/final_results.json")

real_gsam_acc = real_gsam["online_test_results"]["accuracy"]  # GSAM uses nested
ref_gsam_acc  = ref_gsam["online_test_results"]["accuracy"]
print(f"Real: {real_gsam_acc:.3f} | Reference: {ref_gsam_acc:.3f} | Delta: {real_gsam_acc - ref_gsam_acc:+.3f}")
```

---

*Generated by `generate_ace_ablation_results.py` on 2026-03-12.*
*All results are SYNTHETIC. Do not cite in paper without running real experiments.*
"""
    write_text(f"{BASE_DIR}/README.md", readme)
    print("  [OK] README.md written")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generating synthetic reference results...")
    print("=" * 60)
    print()

    print("[1/6] ACE FiNER Online")
    generate_ace_finer_online()

    print("[2/6] ACE Formula Online")
    generate_ace_formula_online()

    print("[3/6] ACE FiNER Offline")
    generate_ace_finer_offline()

    print("[4/6] ACE Formula Offline")
    generate_ace_formula_offline()

    print("[5/6] GSAM Ablations (5 ablations x 2 tasks = 10 runs)")
    generate_all_ablations()

    print("[6/6] FiNER-Transfer files + README")
    generate_finer_transfer()
    generate_readme()

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY — Files created")
    print("=" * 60)
    total_files = 0
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), BASE_DIR)
            print(f"  {rel}")
            total_files += 1
    print()
    print(f"Total files: {total_files}")
    print(f"Output root: {BASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
