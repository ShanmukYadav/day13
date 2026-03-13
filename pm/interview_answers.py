"""
interview_answers.py
Part C: Interview-Ready Questions
Day 13 | PM Take-Home Assignment
"""

import pandas as pd
import numpy as np
import re

# ══════════════════════════════════════════════════════════════
# Q1 — Missing 'income' column: drop vs fill decision framework
# ══════════════════════════════════════════════════════════════

Q1_ANSWER = """
Q1: 1M rows, 40% of 'income' missing — decision framework
===========================================================

STEP 1 — Understand WHY data is missing (missingness mechanism)
  • MCAR (Missing Completely At Random): safe to drop or fill with mean/median
  • MAR  (Missing At Random): depends on other columns → fill using related cols
  • MNAR (Missing Not At Random): e.g., high earners skip the field →
    dropping biases the dataset; must impute carefully

STEP 2 — Should I DROP?
  Drop rows when:
  - Missingness is MCAR and the 40% represents a random minority
  - Rows with missing income have other issues too (nulls in many columns)
  - The downstream model/analysis doesn't require income for every row
  With 1M rows, dropping 400K rows is a significant loss — avoid unless
  the missing rows are demonstrably useless.

  Drop the COLUMN when:
  - Income is not a required feature for the model
  - >80% missing (40% here is not that extreme)

STEP 3 — Fill strategy (if keeping rows)
  a) Median fill:
     Income is right-skewed (outliers from top earners inflate the mean).
     Use median — it's robust to skew and doesn't inflate imputed values.
     df["income"].fillna(df["income"].median())

  b) Group-wise median (better):
     Fill with median income for the respondent's age-group or city —
     makes imputed values contextually realistic.
     df["income"] = df.groupby("age_group")["income"].transform(
         lambda x: x.fillna(x.median())
     )

  c) Model-based imputation (best for production):
     Train a regression model on other features (age, education, city)
     to predict income and fill. scikit-learn's IterativeImputer does this.

STEP 4 — Flag imputed rows
  Always add an indicator column so the model knows which values were
  imputed vs observed:
     df["income_was_missing"] = df["income"].isna().astype(int)

CONCLUSION:
  40% missing is substantial. For a 1M-row survey dataset I would:
  1. Check the missingness mechanism (MCAR vs MNAR)
  2. Fill using group-wise median (by age_group or region)
  3. Add an indicator column
  4. NOT drop — losing 400K rows wastes data and may introduce bias
"""

print(Q1_ANSWER)


# ══════════════════════════════════════════════════════════════
# Q2 — standardize_column() function
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("Q2: standardize_column()")
print("=" * 60)


def standardize_column(series: pd.Series) -> pd.Series:
    """
    Takes a Pandas Series of messy text and returns a cleaned version:
    - Stripped leading/trailing whitespace
    - Lowercased
    - Multiple spaces collapsed to single space
    - Special characters removed (keeps letters, digits, spaces)

    Parameters
    ----------
    series : pd.Series (dtype object / string)

    Returns
    -------
    pd.Series — cleaned text
    """
    return (
        series
        .astype(str)
        .str.strip()                                    # remove edge whitespace
        .str.lower()                                    # normalise casing
        .str.replace(r"[^a-z0-9\s]", "", regex=True)  # remove special chars
        .str.replace(r"\s+", " ", regex=True)          # collapse multiple spaces
        .str.strip()                                    # strip again after replacements
    )


# Test cases
test_series = pd.Series([" Hello World!! ", " NEW YORK ", "san--francisco", " MUMBAI "])
cleaned     = standardize_column(test_series)

print("\nInput  → Output")
print("-" * 40)
for raw, clean in zip(test_series, cleaned):
    print(f"  {repr(raw):<25} → {repr(clean)}")


# ══════════════════════════════════════════════════════════════
# Q3 — Debug: find and fix 4 bugs
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Q3: Debugging — 4 bugs found and fixed")
print("=" * 60)

df = pd.DataFrame({
    "price":    ["1,500", "2000", "N/A", "3,200", "abc"],
    "category": [" Electronics ", "CLOTHING", "electronics", " Books", ""],
    "date":     ["15/03/2024", "2024-07-01", "22-Nov-2024", "01/10/2024", None],
})

# ── Bug 1 ─────────────────────────────────────────────────────
# ORIGINAL (broken):
#   df["price"] = pd.to_numeric(df["price"], errors="coerce")
#   Problem: "1,500" has a comma — to_numeric can't parse it → becomes NaN
#            "N/A" is a hidden missing marker — should be NaN anyway,
#            but "1,500" is valid data that silently gets dropped.
#
# FIX: remove commas FIRST, then replace hidden NaN markers, then convert
df["price"] = (
    df["price"]
    .str.replace(",", "", regex=False)   # remove thousands separator
    .replace({"N/A": None, "": None})    # replace hidden NaN markers
)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
print("\nBug 1 FIXED — strip commas & hidden NaN before to_numeric:")
print(df["price"])

# ── Bug 2 ─────────────────────────────────────────────────────
# ORIGINAL (broken):
#   clean = df[df["price"] > 1000 and df["category"] != ""]
#   Problem: Python 'and' on Series raises ValueError (ambiguous truth value)
#
# FIX: use bitwise & with parentheses around each condition
clean = df[(df["price"] > 1000) & (df["category"] != "")]
print("\nBug 2 FIXED — use & with parentheses instead of 'and':")
print(clean[["price", "category"]])

# ── Bug 3 ─────────────────────────────────────────────────────
# ORIGINAL (broken):
#   electronics = df[df["category"].str.contains("electronics")]
#   Problem 1: NaN in category → str.contains returns NaN → TypeError when used as mask
#   Problem 2: casing inconsistency ('Electronics' won't match 'electronics')
#
# FIX: add na=False and case=False
electronics = df[df["category"].str.contains("electronics", case=False, na=False)]
print("\nBug 3 FIXED — str.contains with na=False and case=False:")
print(electronics[["category"]])

# ── Bug 4 ─────────────────────────────────────────────────────
# ORIGINAL (broken):
#   df["date"] = pd.to_datetime(df["date"])
#   Problem: mixed formats ("15/03/2024", "2024-07-01", "22-Nov-2024")
#            causes ParserError; None/NaN causes crash without errors='coerce'
#
# FIX: use dayfirst=True for dd/mm/yyyy + errors='coerce' for unparseable values
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
print("\nBug 4 FIXED — to_datetime with dayfirst=True and errors='coerce':")
print(df["date"])

print("\n✅ interview_answers.py complete!")
