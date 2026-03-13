"""
data_profiler.py
Part B: Automated Data Profiler
Day 13 | PM Take-Home Assignment
"""

import pandas as pd
import numpy as np

# scipy is optional — gracefully degrade if not installed
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# Core profiler
# ══════════════════════════════════════════════════════════════

def profile_dataframe(df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Generate a complete profile for any DataFrame.

    For every column reports:
      - dtype, unique count, null count/percentage,
        top-N most frequent values

    For numeric columns additionally reports:
      - min, max, mean, median, std, skewness

    For string/object columns additionally reports:
      - avg length, min/max length, common patterns

    Also identifies potential data issues:
      - Single-value columns (zero variance)
      - High-cardinality string columns (potential free-text / ID cols)
      - Suspicious numeric outliers (> 3 std from mean)

    Parameters
    ----------
    df    : any pandas DataFrame
    top_n : number of top frequent values to show per column (default 5)

    Returns
    -------
    dict  : nested profile dict
    """

    if df.empty:
        print("⚠️  DataFrame is empty.")
        return {"error": "empty_dataframe"}

    profile = {
        "overview": {},
        "columns":  {},
        "issues":   {},
    }

    # ── Overview ─────────────────────────────────────────────────
    profile["overview"] = {
        "rows":       df.shape[0],
        "cols":       df.shape[1],
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb":  round(df.memory_usage(deep=True).sum() / 1_048_576, 4),
    }

    # ── Per-column profile ───────────────────────────────────────
    issues_single_value  = []
    issues_high_card     = []
    issues_outliers      = {}

    for col in df.columns:
        series      = df[col]
        null_count  = int(series.isnull().sum())
        null_pct    = round(null_count / len(df) * 100, 2)
        unique_count= int(series.nunique(dropna=True))

        col_profile = {
            "dtype":        str(series.dtype),
            "unique_count": unique_count,
            "null_count":   null_count,
            "null_pct":     null_pct,
            "top_values":   series.value_counts(dropna=True).head(top_n).to_dict(),
        }

        # ── Numeric extras ────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                skewness = (
                    float(scipy_stats.skew(clean)) if SCIPY_AVAILABLE
                    else float(clean.skew())        # pandas fallback
                )
                col_profile["numeric"] = {
                    "min":    float(clean.min()),
                    "max":    float(clean.max()),
                    "mean":   round(float(clean.mean()), 4),
                    "median": float(clean.median()),
                    "std":    round(float(clean.std()), 4),
                    "skewness": round(skewness, 4),
                }

                # Outlier detection: values > 3 std from mean
                mean, std = clean.mean(), clean.std()
                if std > 0:
                    outlier_mask  = (clean - mean).abs() > 3 * std
                    outlier_count = int(outlier_mask.sum())
                    if outlier_count > 0:
                        issues_outliers[col] = {
                            "outlier_count": outlier_count,
                            "examples": clean[outlier_mask].head(3).tolist(),
                        }

        # ── String / object extras ────────────────────────────────
        elif series.dtype == object:
            str_series = series.dropna().astype(str)
            lengths    = str_series.str.len()
            if len(lengths) > 0:
                col_profile["string"] = {
                    "avg_length": round(float(lengths.mean()), 2),
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                    "has_whitespace_issues": bool(
                        (str_series != str_series.str.strip()).any()
                    ),
                    "has_mixed_case": bool(
                        (str_series != str_series.str.lower()).any()
                        and (str_series != str_series.str.upper()).any()
                    ),
                }

            # High-cardinality check: >50% unique values in a string col
            if unique_count / max(len(df), 1) > 0.5:
                issues_high_card.append(col)

        # ── Single-value check (any dtype) ────────────────────────
        if unique_count <= 1:
            issues_single_value.append(col)

        profile["columns"][col] = col_profile

    # ── Issues summary ────────────────────────────────────────────
    profile["issues"] = {
        "single_value_cols":    issues_single_value,
        "high_cardinality_cols": issues_high_card,
        "outlier_cols":         issues_outliers,
    }

    # ── Formatted print summary ───────────────────────────────────
    _print_profile_summary(profile, df)

    return profile


def _print_profile_summary(profile: dict, df: pd.DataFrame) -> None:
    ov = profile["overview"]
    print("\n" + "═" * 65)
    print("  DATAFRAME PROFILE SUMMARY")
    print("═" * 65)
    print(f"\n📐 Shape          : {ov['rows']} rows × {ov['cols']} cols")
    print(f"💾 Memory         : {ov['memory_mb']} MB")
    print(f"❓ Total Nulls    : {ov['total_nulls']}")
    print(f"🔁 Duplicate Rows : {ov['duplicate_rows']}")

    print("\n" + "─" * 65)
    print(f"  {'Column':<22} {'Dtype':<12} {'Nulls':>8} {'Null%':>7} {'Uniq':>6}")
    print("─" * 65)
    for col, cp in profile["columns"].items():
        print(
            f"  {col:<22} {cp['dtype']:<12} "
            f"{cp['null_count']:>8} {cp['null_pct']:>6.1f}% {cp['unique_count']:>6}"
        )

    # Numeric details
    num_cols = [c for c, cp in profile["columns"].items() if "numeric" in cp]
    if num_cols:
        print("\n📈 Numeric Column Details:")
        print(f"  {'Column':<22} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10} {'Skew':>8}")
        print("  " + "─" * 75)
        for col in num_cols:
            n = profile["columns"][col]["numeric"]
            print(
                f"  {col:<22} {n['min']:>10.2f} {n['max']:>10.2f} "
                f"{n['mean']:>10.2f} {n['median']:>10.2f} {n['skewness']:>8.2f}"
            )

    # String details
    str_cols = [c for c, cp in profile["columns"].items() if "string" in cp]
    if str_cols:
        print("\n📝 String Column Details:")
        print(f"  {'Column':<22} {'AvgLen':>8} {'MinLen':>8} {'MaxLen':>8} {'Whitespace':>12} {'MixedCase':>12}")
        print("  " + "─" * 75)
        for col in str_cols:
            s = profile["columns"][col]["string"]
            print(
                f"  {col:<22} {s['avg_length']:>8.1f} {s['min_length']:>8} "
                f"{s['max_length']:>8} {str(s['has_whitespace_issues']):>12} "
                f"{str(s['has_mixed_case']):>12}"
            )

    # Issues
    issues = profile["issues"]
    print("\n⚠️  Potential Issues:")
    if issues["single_value_cols"]:
        print(f"   Single-value columns     : {issues['single_value_cols']}")
    if issues["high_cardinality_cols"]:
        print(f"   High-cardinality strings : {issues['high_cardinality_cols']}")
    if issues["outlier_cols"]:
        for col, info in issues["outlier_cols"].items():
            print(f"   Outliers in '{col}': {info['outlier_count']} values — e.g. {info['examples']}")
    if not any([issues["single_value_cols"], issues["high_cardinality_cols"], issues["outlier_cols"]]):
        print("   ✅ None detected")

    print("═" * 65)


# ══════════════════════════════════════════════════════════════
# Demo: run profiler on clean & messy DataFrames
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # --- Demo 1: clean e-commerce data ---
    print("\n" + "=" * 65)
    print("DEMO 1: Clean E-Commerce DataFrame")
    print("=" * 65)

    clean_df = pd.DataFrame({
        "product":  ["Laptop", "Phone", "Tablet", "Watch", "Earbuds",
                     "Monitor", "Keyboard", "Mouse", "Speaker", "Charger"],
        "category": ["Electronics"] * 8 + ["Accessories", "Accessories"],
        "price":    [75000, 25000, 40000, 15000, 5000,
                     22000, 3500, 1200, 8000, 999],
        "stock":    [50, 120, 80, 200, 350, 45, 300, 400, 150, 600],
        "rating":   [4.5, 4.2, 4.7, 4.0, 4.3, 4.6, 4.1, 4.4, 4.3, 4.0],
        "reviews":  [1500, 3200, 800, 4500, 9000, 600, 2100, 3800, 700, 12000],
    })

    profile_clean = profile_dataframe(clean_df)

    # --- Demo 2: messy survey data from Part A ---
    print("\n" + "=" * 65)
    print("DEMO 2: Messy Survey Data (from survey_cleaner.py)")
    print("=" * 65)

    try:
        survey_df = pd.read_csv("survey_results.csv")
        profile_survey = profile_dataframe(survey_df)
    except FileNotFoundError:
        print("⚠️  survey_results.csv not found. Run survey_cleaner.py first.")

    print("\n✅ data_profiler.py complete!")
