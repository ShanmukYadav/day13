"""
interview_answers.py
Part C: Interview-Ready Questions
Day 14 | AM Take-Home Assignment
"""

import pandas as pd

# ══════════════════════════════════════════════════════════════
# Q1 — .loc[] vs .iloc[]: Conceptual Explanation
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("Q1: .loc[] vs .iloc[]")
print("=" * 60)

# --- Scenario 1: Integer index 0,1,2,3,4 ---
df_int = pd.DataFrame({
    "name":   ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "salary": [50000, 60000, 70000, 80000, 90000]
})

print("\n--- Integer index (0, 1, 2, 3, 4) ---")
print("df_int:\n", df_int)

# .loc[] with integer index is LABEL-based → 0:3 includes 0,1,2,3 (INCLUSIVE end)
print("\ndf.loc[0:3]  → label-based, END IS INCLUSIVE:")
print(df_int.loc[0:3])

# .iloc[] is POSITION-based → 0:3 means positions 0,1,2 (EXCLUSIVE end)
print("\ndf.iloc[0:3] → position-based, END IS EXCLUSIVE:")
print(df_int.iloc[0:3])

# --- Scenario 2: String index 'a','b','c','d','e' ---
df_str = pd.DataFrame(
    {"name":   ["Alice", "Bob", "Charlie", "Diana", "Eve"],
     "salary": [50000, 60000, 70000, 80000, 90000]},
    index=["a", "b", "c", "d", "e"]
)

print("\n--- String index ('a','b','c','d','e') ---")
print("df_str:\n", df_str)

# .loc[] uses the actual string labels 'a':'d' → INCLUSIVE
print("\ndf_str.loc['a':'d']  → includes 'a','b','c','d':")
print(df_str.loc["a":"d"])

# .iloc[] ignores labels, uses positions 0:3 → EXCLUSIVE → 0,1,2
print("\ndf_str.iloc[0:3]     → positions 0,1,2 → 'a','b','c':")
print(df_str.iloc[0:3])

print("""
SUMMARY
-------
| Feature       | .loc[]                        | .iloc[]                       |
|---------------|-------------------------------|-------------------------------|
| Basis         | LABELS (index values)         | INTEGER POSITIONS (0-based)   |
| Slice end     | INCLUSIVE                     | EXCLUSIVE                     |
| Works with    | Any index type                | Always 0,1,2,3...             |
| Use when      | You know the row/column NAME  | You know the row/column ORDER |
""")

# ══════════════════════════════════════════════════════════════
# Q2 — analyze_csv() function
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("Q2: analyze_csv() function")
print("=" * 60)


def analyze_csv(filepath: str) -> dict:
    """
    Load a CSV, print the 'First 5 Minutes' checklist, and return a
    summary dict with key metadata about the DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    dict with keys:
        num_rows, num_cols, numeric_cols, categorical_cols,
        null_counts, memory_mb
    """
    df = pd.read_csv(filepath)

    # ── First 5 Minutes checklist ──────────────────────────────
    print(f"\n📂 File: {filepath}")
    print(f"📐 Shape: {df.shape}")
    print(f"\n📋 Columns:\n{df.columns.tolist()}")
    print(f"\n🔍 Head:\n{df.head()}")
    print(f"\n📊 Info:")
    df.info()
    print(f"\n📈 Describe:\n{df.describe()}")
    print(f"\n❓ Null counts:\n{df.isnull().sum()}")

    # ── Build result dict ─────────────────────────────────────
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    null_counts      = df.isnull().sum().to_dict()
    memory_mb        = round(df.memory_usage(deep=True).sum() / 1_048_576, 4)

    result = {
        "num_rows":        df.shape[0],
        "num_cols":        df.shape[1],
        "numeric_cols":    numeric_cols,
        "categorical_cols": categorical_cols,
        "null_counts":     null_counts,
        "memory_mb":       memory_mb,
    }

    print(f"\n✅ Summary dict:\n{result}")
    return result


# Quick test with the CSV we generated in Part A
result = analyze_csv("budget_products.csv")

# ══════════════════════════════════════════════════════════════
# Q3 — Debug: find and fix 3 bugs
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Q3: Debugging")
print("=" * 60)

df_debug = pd.DataFrame({
    "name":   ["Alice", "Bob", "Charlie"],
    "age":    [25, 30, 35],
    "salary": [50000, 60000, 70000]
})

# ── Bug 1 ─────────────────────────────────────────────────────
# ORIGINAL (broken): uses Python 'and' — raises ValueError with Series
#   high_earners = df[df["age"] > 25 and df["salary"] > 55000]
#
# FIX: use bitwise '&' and wrap each condition in parentheses
high_earners = df_debug[(df_debug["age"] > 25) & (df_debug["salary"] > 55000)]
print("\nBug 1 FIXED — use & instead of 'and':")
print(high_earners)

# ── Bug 2 ─────────────────────────────────────────────────────
# ORIGINAL (broken): chained indexing — raises SettingWithCopyWarning
#   and may not actually update the original DataFrame
#   df["age"][0] = 26
#
# FIX: use .loc[] for safe, unambiguous assignment
df_debug.loc[0, "age"] = 26
print("\nBug 2 FIXED — use .loc[] for assignment:")
print(df_debug)

# ── Bug 3 ─────────────────────────────────────────────────────
# ORIGINAL (broken): comment says "expecting 3 rows (0,1,2)" but
#   .iloc[0:2] only returns 2 rows because iloc end is EXCLUSIVE
#   first_two = df.iloc[0:2]   # gets rows 0 and 1 only
#
# FIX: use .iloc[0:3] to get rows 0, 1, 2
first_three = df_debug.iloc[0:3]  # end is exclusive → gives 3 rows
print(f"\nBug 3 FIXED — use iloc[0:3] to get 3 rows:")
print(f"Got {len(first_three)} rows, expected 3")
print(first_three)

print("\n✅ interview_answers.py complete!")
