"""
ai_augmented_task.py
Part D: AI-Augmented Task
Day 14 | AM Take-Home Assignment

Prompt used:
  "Write a Python function that takes a Pandas DataFrame and generates
   an automated data quality report including: shape, dtypes, missing
   values percentage, duplicate rows, unique value counts per column,
   and basic stats. Return the report as a dict and also print a
   formatted summary."

AI output below (tested, reviewed, and improved).
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════
# AI-generated function (tested and improved)
# ══════════════════════════════════════════════════════════════

def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate an automated data quality report for a Pandas DataFrame.

    Improvements over raw AI output:
    - Handles empty DataFrames gracefully (early return with warning).
    - Detects all-null columns (useless features) explicitly.
    - Flags single-unique-value columns (zero-variance, useless features).
    - Uses df.memory_usage(deep=True) for accurate memory measurement.
    - Adds duplicate row percentage alongside raw count.
    - Prints a clean, section-based formatted summary.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict — structured quality report
    """

    # ── Edge case: empty DataFrame ─────────────────────────────
    if df.empty:
        print("⚠️  DataFrame is empty — no quality report generated.")
        return {"error": "empty_dataframe"}

    report = {}

    # ── Shape ──────────────────────────────────────────────────
    report["shape"] = {"rows": df.shape[0], "cols": df.shape[1]}

    # ── Data types ─────────────────────────────────────────────
    report["dtypes"] = df.dtypes.astype(str).to_dict()

    # ── Missing values ─────────────────────────────────────────
    null_counts   = df.isnull().sum()
    null_pct      = (null_counts / len(df) * 100).round(2)
    all_null_cols = null_counts[null_counts == len(df)].index.tolist()
    report["missing"] = {
        "counts":        null_counts.to_dict(),
        "percentage":    null_pct.to_dict(),
        "all_null_cols": all_null_cols,
    }

    # ── Duplicates ─────────────────────────────────────────────
    dup_count = df.duplicated().sum()
    report["duplicates"] = {
        "count":      int(dup_count),
        "percentage": round(dup_count / len(df) * 100, 2),
    }

    # ── Unique value counts per column ─────────────────────────
    unique_counts      = df.nunique().to_dict()
    single_value_cols  = [col for col, n in unique_counts.items() if n <= 1]
    report["unique_counts"]     = unique_counts
    report["single_value_cols"] = single_value_cols

    # ── Basic stats (numeric columns only) ─────────────────────
    numeric_df = df.select_dtypes(include="number")
    report["basic_stats"] = (
        numeric_df.describe().to_dict() if not numeric_df.empty else {}
    )

    # ── Memory ─────────────────────────────────────────────────
    report["memory_mb"] = round(
        df.memory_usage(deep=True).sum() / 1_048_576, 4
    )

    # ── Formatted summary ──────────────────────────────────────
    print("\n" + "═" * 55)
    print("  DATA QUALITY REPORT")
    print("═" * 55)

    print(f"\n📐 Shape          : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"💾 Memory         : {report['memory_mb']} MB")
    print(f"🔁 Duplicate rows : {dup_count} ({report['duplicates']['percentage']}%)")

    print("\n📋 Data Types:")
    for col, dtype in report["dtypes"].items():
        print(f"   {col:<25} {dtype}")

    print("\n❓ Missing Values (columns with any nulls):")
    missing_cols = {c: v for c, v in null_pct.items() if v > 0}
    if missing_cols:
        for col, pct in missing_cols.items():
            marker = " ⚠️  ALL NULL" if col in all_null_cols else ""
            print(f"   {col:<25} {pct}%{marker}")
    else:
        print("   ✅ No missing values!")

    print("\n🔢 Unique Values per Column:")
    for col, n in unique_counts.items():
        flag = " ⚠️  SINGLE VALUE" if col in single_value_cols else ""
        print(f"   {col:<25} {n}{flag}")

    if report["basic_stats"]:
        print("\n📈 Numeric Summary:")
        print(numeric_df.describe().to_string())

    print("\n" + "═" * 55)
    return report


# ══════════════════════════════════════════════════════════════
# Test 1: Clean DataFrame
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 1: Clean DataFrame")
print("=" * 60)

clean_df = pd.DataFrame({
    "product":  ["Laptop", "Phone", "Tablet", "Watch", "Earbuds"],
    "price":    [75000, 25000, 40000, 15000, 5000],
    "stock":    [50, 120, 80, 200, 350],
    "rating":   [4.5, 4.2, 4.7, 4.0, 4.3],
})

clean_report = data_quality_report(clean_df)

# ══════════════════════════════════════════════════════════════
# Test 2: Messy DataFrame (nulls, duplicates, single-value col)
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 2: Messy DataFrame")
print("=" * 60)

messy_df = pd.DataFrame({
    "product":     ["Laptop", "Phone", None, "Laptop", "Watch"],
    "price":       [75000, None, 40000, 75000, None],
    "stock":       [50, 120, None, 50, 200],
    "rating":      [None, None, None, None, None],   # all-null column
    "status":      ["active"] * 5,                   # single-value column
})

messy_report = data_quality_report(messy_df)

# ══════════════════════════════════════════════════════════════
# Test 3: Edge case — Empty DataFrame
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 3: Empty DataFrame (edge case)")
print("=" * 60)

empty_df = pd.DataFrame()
empty_report = data_quality_report(empty_df)

# ══════════════════════════════════════════════════════════════
# Critical Evaluation (200 words)
# ══════════════════════════════════════════════════════════════

CRITICAL_EVALUATION = """
CRITICAL EVALUATION OF AI OUTPUT
==================================
The AI-generated function produced a solid baseline covering shape, dtypes,
missing value counts, duplicate rows, unique value counts, and basic stats.
However, several gaps and weaknesses were identified:

1. EMPTY DATAFRAME: The original AI output would crash on df.empty because
   df.isnull().sum() / len(df) causes a ZeroDivisionError. Fixed by adding
   an early return guard.

2. MEMORY ACCURACY: The AI used df.memory_usage().sum() (shallow), which
   underestimates object-column memory. Fixed with deep=True for accurate MB.

3. ALL-NULL COLUMNS: Not detected explicitly. These represent completely
   useless features and should be flagged for dropping. Added detection.

4. SINGLE UNIQUE VALUE COLUMNS: The AI listed unique counts but did not flag
   zero-variance columns (e.g., a "status" column always = "active"). Added
   explicit detection as these are useless for ML/analysis.

5. MISSING VALUE PERCENTAGE: The AI returned raw counts only; percentage is
   more actionable (e.g., 80% missing → drop the column). Added.

6. FORMATTING: The AI's print output was flat. Restructured with sections,
   emoji markers, and alignment for readability.

Overall the AI accelerated ~60% of the work but required careful testing,
edge-case hardening, and enrichment to be production-ready.
"""

print(CRITICAL_EVALUATION)
print("✅ ai_augmented_task.py complete!")
