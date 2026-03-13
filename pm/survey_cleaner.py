"""
survey_cleaner.py
Part A: End-to-End Data Cleaning Pipeline
Day 13 | PM Take-Home Assignment
"""

import pandas as pd
import numpy as np
import json

# ══════════════════════════════════════════════════════════════
# 1. Create messy survey dataset (50+ rows, 8+ columns)
#    with 8+ deliberate data quality issues embedded
# ══════════════════════════════════════════════════════════════

def create_messy_survey() -> pd.DataFrame:
    """
    Generates a realistic but intentionally messy customer satisfaction
    survey dataset containing the following 8+ data quality issues:

    Issue 1  — NaN / None values scattered across columns
    Issue 2  — Empty string "" used as a missing value marker
    Issue 3  — Hidden NaN strings: 'N/A', 'null', 'None', 'n/a'
    Issue 4  — Duplicate rows
    Issue 5  — Wrong dtypes: age/score stored as strings
    Issue 6  — Inconsistent casing: 'electronics', 'ELECTRONICS', 'Electronics'
    Issue 7  — Extra whitespace in text: '  New York  ', ' Female'
    Issue 8  — Invalid numeric values: age = -5, 200; score = 0, 11
    Issue 9  — Inconsistent date formats: '15/03/2024', '2024-03-15'
    Issue 10 — Mixed category labels: 'Yes'/'yes'/'YES', 'No'/'NO'
    """
    np.random.seed(42)
    n = 55

    names = [
        "Alice Johnson", "Bob Smith", "  Charlie Brown  ", "Diana Prince",
        "Eve Adams", "Frank Castle", "Grace Hopper", " Henry Ford",
        "Iris West", "Jack Ryan", "Karen Page", "Leo Messi",
        "Mia Khalifa", "Nina Simone", "Oscar Wilde", "Pam Beesly",
        "Quinn Hughes", "Rachel Green", "Steve Rogers", "Tina Turner",
        "Uma Thurman", "Vince Lombardi", "Wendy Williams", "Xander Cage",
        "Yara Greyjoy", "Zoe Saldana", "Aaron Burr", "Beth March",
        "Carl Sagan", "Dorothy Gale", "Edward Norton", "Fiona Apple",
        "George Weasley", "Hannah Abbott", "Ivan Drago", "Jane Eyre",
        "Kurt Cobain", "Lara Croft", "Miles Morales", "Nancy Wheeler",
        "Oliver Queen", "Penny Lane", "Quinn Fabray", "Ron Weasley",
        "Sara Lance", "Ted Lasso", "Uma Fox", "Victor Stone",
        "Wanda Maximoff", "Xena Warrior", "Yogi Bear", "Zack Morris",
        "Alice Johnson",   # duplicate of row 0
        "Bob Smith",       # duplicate of row 1
        "N/A",             # hidden NaN in name
    ]

    ages_raw = (
        [str(np.random.randint(18, 65)) for _ in range(40)]
        + ["-5", "200", "abc", "N/A", ""]   # invalid / hidden NaN
        + [str(np.random.randint(18, 65)) for _ in range(10)]
    )

    genders = (
        ["Male", "Female", "FEMALE", " Male", "female", "MALE",
         "Non-binary", "male", "Female", "Male"] * 5
        + ["", None, "N/A", "male", "Female"]
    )

    cities = (
        ["  New York  ", "Los Angeles", "CHICAGO", "houston", " Miami  ",
         "Seattle", "BOSTON", "san francisco", "Denver", "Phoenix"] * 5
        + ["null", "", "New York", "Chicago", "Los Angeles"]
    )

    categories = (
        ["Electronics", "ELECTRONICS", "electronics", "Clothing",
         "CLOTHING", "clothing", "Books", "BOOKS", "books", "Home"] * 5
        + ["Electronics", "Clothing", "Books", "Home", "electronics"]
    )

    satisfaction_raw = (
        [str(np.random.randint(1, 11)) for _ in range(40)]
        + ["0", "11", "N/A", "", "10"]    # invalid / hidden NaN
        + [str(np.random.randint(1, 10)) for _ in range(10)]
    )

    purchase_amounts = (
        [round(np.random.uniform(100, 50000), 2) for _ in range(45)]
        + [None, np.nan, "N/A", "", -999.0]   # various missing markers
        + [round(np.random.uniform(100, 50000), 2) for _ in range(5)]
    )

    would_recommend = (
        ["Yes", "No", "yes", "NO", "YES", "no", "Yes", "No", "yes", "NO"] * 5
        + ["Yes", "no", "N/A", "YES", ""]
    )

    dates = (
        ["15/03/2024", "2024-04-01", "22-May-2024", "01/06/2024",
         "2024-07-15", "30/08/2024", "2024-09-10", "11/10/2024",
         "2024-11-05", "20/12/2024"] * 5
        + ["N/A", None, "2024-01-01", "15/03/2024", "2024-06-30"]
    )

    df = pd.DataFrame({
        "respondent_name":   names[:n],
        "age":               ages_raw[:n],
        "gender":            genders[:n],
        "city":              cities[:n],
        "product_category":  categories[:n],
        "satisfaction_score": satisfaction_raw[:n],
        "purchase_amount":   purchase_amounts[:n],
        "would_recommend":   would_recommend[:n],
        "survey_date":       dates[:n],
    })

    return df


# ══════════════════════════════════════════════════════════════
# 2. detect_issues() — comprehensive data quality report
# ══════════════════════════════════════════════════════════════

HIDDEN_NAN_MARKERS = {"N/A", "n/a", "null", "None", "none", "NA", "", "nan"}

def detect_issues(df: pd.DataFrame) -> dict:
    """
    Returns a structured data quality report as a dict covering:
    total_rows, total_missing, missing_per_column, duplicate_count,
    wrong_types, invalid_values.
    """
    report = {}

    report["total_rows"]    = len(df)
    report["total_columns"] = len(df.columns)

    # Missing (real NaN + hidden markers)
    def count_effective_nulls(series: pd.Series) -> int:
        return series.isna().sum() + series.astype(str).isin(HIDDEN_NAN_MARKERS).sum()

    missing_per_col = {col: int(count_effective_nulls(df[col])) for col in df.columns}
    report["total_missing"]      = sum(missing_per_col.values())
    report["missing_per_column"] = missing_per_col

    # Duplicates
    report["duplicate_count"] = int(df.duplicated().sum())

    # Wrong types (columns that look numeric but are stored as object)
    wrong_types = {}
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null  = df[col].dropna()
        if len(non_null) > 0 and converted.notna().sum() / len(non_null) > 0.6:
            wrong_types[col] = f"object → should be numeric"
    report["wrong_types"] = wrong_types

    # Invalid values
    invalid = {}
    age_num = pd.to_numeric(df.get("age", pd.Series(dtype=float)), errors="coerce")
    invalid["age_out_of_range"]  = int(((age_num < 0) | (age_num > 120)).sum())

    score_num = pd.to_numeric(df.get("satisfaction_score", pd.Series(dtype=float)), errors="coerce")
    invalid["score_out_of_range"] = int(((score_num < 1) | (score_num > 10)).sum())

    amt = pd.to_numeric(df.get("purchase_amount", pd.Series(dtype=float)), errors="coerce")
    invalid["negative_purchase"]  = int((amt < 0).sum())
    report["invalid_values"] = invalid

    return report


# ══════════════════════════════════════════════════════════════
# 3. clean_data() — full cleaning pipeline
# ══════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a structured cleaning pipeline:
      1. Replace hidden NaN markers with real NaN
      2. Fix data types
      3. Standardize text columns
      4. Handle missing values (strategy justified per column)
      5. Remove invalid rows
      6. Drop duplicates
    Returns the cleaned DataFrame.
    """
    df = df.copy()

    # ── Step 1: Replace hidden NaN markers ──────────────────────
    for col in df.columns:
        df[col] = df[col].replace(list(HIDDEN_NAN_MARKERS), np.nan)

    # ── Step 2: Fix data types ───────────────────────────────────
    # Use to_numeric(errors='coerce') — safer than astype() because
    # astype() raises on unconvertible values; coerce turns them to NaN.
    df["age"]                = pd.to_numeric(df["age"],                errors="coerce")
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    df["purchase_amount"]    = pd.to_numeric(df["purchase_amount"],    errors="coerce")

    # to_datetime with dayfirst=True handles dd/mm/yyyy; errors='coerce'
    # handles truly unparseable entries rather than crashing.
    df["survey_date"] = pd.to_datetime(df["survey_date"], dayfirst=True, errors="coerce")

    # ── Step 3: Standardize text columns ────────────────────────
    text_cols = ["respondent_name", "gender", "city", "product_category", "would_recommend"]
    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()         # remove leading/trailing whitespace
            .str.lower()         # normalise casing
            .str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces
        )
    # Re-mark 'nan' strings (artefact of astype(str) on NaN) as real NaN
    for col in text_cols:
        df[col] = df[col].replace("nan", np.nan)

    # ── Step 4: Fill missing values ─────────────────────────────
    # age → median: skewed distributions make mean misleading; median is robust
    df["age"] = df["age"].fillna(df["age"].median())

    # satisfaction_score → median: ordinal 1-10 scale; median preserves rank
    df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())

    # purchase_amount → median: right-skewed money distributions; mean inflated by outliers
    df["purchase_amount"] = df["purchase_amount"].fillna(df["purchase_amount"].median())

    # gender → mode: categorical, no ordering; most frequent label is the best guess
    df["gender"] = df["gender"].fillna(df["gender"].mode()[0])

    # city → mode: same reasoning as gender
    df["city"] = df["city"].fillna(df["city"].mode()[0])

    # product_category → mode: categorical, fill with most common category
    df["product_category"] = df["product_category"].fillna(df["product_category"].mode()[0])

    # would_recommend → mode: binary categorical; fill with majority class
    df["would_recommend"] = df["would_recommend"].fillna(df["would_recommend"].mode()[0])

    # respondent_name → drop: a missing name makes the row unidentifiable
    df = df.dropna(subset=["respondent_name"])

    # survey_date → drop: date is critical for time-series analysis; imputing is risky
    df = df.dropna(subset=["survey_date"])

    # ── Step 5: Remove invalid rows ─────────────────────────────
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    df = df[(df["satisfaction_score"] >= 1) & (df["satisfaction_score"] <= 10)]
    df = df[df["purchase_amount"] >= 0]

    # ── Step 6: Remove duplicates ────────────────────────────────
    df = df.drop_duplicates()

    df = df.reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════
# 4. Before / after comparison
# ══════════════════════════════════════════════════════════════

def compare_before_after(raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    print("\n" + "═" * 60)
    print("  BEFORE vs AFTER COMPARISON")
    print("═" * 60)

    print(f"\n{'Metric':<30} {'Before':>10} {'After':>10}")
    print("-" * 52)
    print(f"{'Rows':<30} {len(raw):>10} {len(cleaned):>10}")
    print(f"{'Columns':<30} {raw.shape[1]:>10} {cleaned.shape[1]:>10}")
    print(f"{'Total nulls':<30} {int(raw.isnull().sum().sum()):>10} {int(cleaned.isnull().sum().sum()):>10}")
    print(f"{'Duplicate rows':<30} {int(raw.duplicated().sum()):>10} {int(cleaned.duplicated().sum()):>10}")

    mem_before = raw.memory_usage(deep=True).sum() / 1024
    mem_after  = cleaned.memory_usage(deep=True).sum() / 1024
    print(f"{'Memory (KB)':<30} {mem_before:>10.1f} {mem_after:>10.1f}")

    print("\n📋 Dtype comparison:")
    print(f"  {'Column':<25} {'Before':<15} {'After':<15}")
    print("  " + "-" * 55)
    for col in raw.columns:
        before_dtype = str(raw[col].dtype)
        after_dtype  = str(cleaned[col].dtype) if col in cleaned.columns else "dropped"
        changed = " ✅" if before_dtype != after_dtype else ""
        print(f"  {col:<25} {before_dtype:<15} {after_dtype:<15}{changed}")


# ══════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create messy data & save as CSV
    raw_df = create_messy_survey()
    raw_df.to_csv("survey_results.csv", index=False)
    print(f"✅ Created survey_results.csv ({len(raw_df)} rows)")

    # Detect issues
    print("\n" + "═" * 60)
    print("  DATA QUALITY REPORT (before cleaning)")
    print("═" * 60)
    report = detect_issues(raw_df)
    for k, v in report.items():
        print(f"\n{k}:")
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                print(f"   {sub_k:<30} {sub_v}")
        else:
            print(f"   {v}")

    # Clean
    cleaned_df = clean_data(raw_df)

    # Compare
    compare_before_after(raw_df, cleaned_df)

    # Export
    cleaned_df.to_csv("cleaned_survey.csv", index=False)
    with open("data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Exported cleaned_survey.csv     ({len(cleaned_df)} rows)")
    print("✅ Exported data_quality_report.json")
    print("\n✅ survey_cleaner.py complete!")
