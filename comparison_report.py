"""
comparison_report.py
Part B: Multi-DataFrame Comparison Report
Day 14 | AM Take-Home Assignment
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. Create 3 DataFrames — one per month
# ─────────────────────────────────────────────

np.random.seed(42)

products = [
    "Samsung Galaxy S24", "Apple MacBook Air", "Sony Headphones",
    "Nike Air Max", "Levi Jeans", "Atomic Habits",
    "Instant Pot", "Philips Air Fryer", "Boat Rockerz",
    "OnePlus Nord"
]
categories = [
    "Electronics", "Electronics", "Electronics",
    "Clothing", "Clothing", "Books",
    "Home", "Home", "Electronics", "Electronics"
]
base_prices = [74999, 114900, 29990, 8995, 3499, 399, 8999, 10499, 1799, 24999]


def generate_month_df(month_name: str, seed_offset: int) -> pd.DataFrame:
    """Generate a realistic sales DataFrame for one month."""
    np.random.seed(42 + seed_offset)
    units_sold = np.random.randint(10, 150, size=len(products))
    prices = [p * np.random.uniform(0.9, 1.1) for p in base_prices]
    revenue = [int(u * p) for u, p in zip(units_sold, prices)]
    return pd.DataFrame({
        "product":    products,
        "category":   categories,
        "units_sold": units_sold,
        "price":      [round(p, 2) for p in prices],
        "revenue":    revenue,
        "month":      month_name,
    })


jan = generate_month_df("January",  0)
feb = generate_month_df("February", 1)
mar = generate_month_df("March",    2)

print("=" * 60)
print("MONTHLY SALES DATA (sample)")
print("=" * 60)
for month_df, label in [(jan, "January"), (feb, "February"), (mar, "March")]:
    print(f"\n{label} (first 3 rows):")
    print(month_df[["product", "units_sold", "revenue"]].head(3))

# ─────────────────────────────────────────────
# 2. Calculate metrics for each month
# ─────────────────────────────────────────────

def monthly_metrics(month_df: pd.DataFrame) -> dict:
    """Return total_revenue, avg_order_value, top_selling_product."""
    total_revenue     = month_df["revenue"].sum()
    avg_order_value   = month_df["revenue"].mean()
    top_product       = month_df.loc[month_df["revenue"].idxmax(), "product"]
    return {
        "total_revenue":    total_revenue,
        "avg_order_value":  round(avg_order_value, 2),
        "top_product":      top_product,
    }

jan_metrics = monthly_metrics(jan)
feb_metrics = monthly_metrics(feb)
mar_metrics = monthly_metrics(mar)

print("\n" + "=" * 60)
print("MONTHLY METRICS")
print("=" * 60)
for label, m in [("January", jan_metrics), ("February", feb_metrics), ("March", mar_metrics)]:
    print(f"\n{label}:")
    print(f"  Total Revenue   : ₹{m['total_revenue']:,.0f}")
    print(f"  Avg Order Value : ₹{m['avg_order_value']:,.2f}")
    print(f"  Top Product     : {m['top_product']}")

# ─────────────────────────────────────────────
# 3. Summary comparison DataFrame
# ─────────────────────────────────────────────

summary = pd.DataFrame(
    [jan_metrics, feb_metrics, mar_metrics],
    index=["January", "February", "March"]
)
summary.index.name = "month"

print("\n" + "=" * 60)
print("SUMMARY COMPARISON DATAFRAME")
print("=" * 60)
print(summary)

# ─────────────────────────────────────────────
# 4. .query() filtering
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("QUERY() EXAMPLES")
print("=" * 60)

# Combine all three months for easier querying
all_months = pd.concat([jan, feb, mar], ignore_index=True)

high_revenue_electronics = all_months.query("category == 'Electronics' and revenue > 500000")
print("\nHigh-revenue Electronics (revenue > 500000):")
print(high_revenue_electronics[["product", "month", "units_sold", "revenue"]])

jan_top = jan.query("units_sold > 80")
print("\nJanuary products with units_sold > 80:")
print(jan_top[["product", "units_sold", "revenue"]])

# ─────────────────────────────────────────────
# 5. nlargest() / nsmallest() for outliers
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("OUTLIER DETECTION (nlargest / nsmallest)")
print("=" * 60)

print("\nTop 3 revenue products across all months:")
print(all_months.nlargest(3, "revenue")[["product", "month", "revenue"]])

print("\nBottom 3 revenue products across all months:")
print(all_months.nsmallest(3, "revenue")[["product", "month", "revenue"]])

print("\nTop 3 units sold in March:")
print(mar.nlargest(3, "units_sold")[["product", "units_sold"]])

# Export summary
summary.to_csv("monthly_summary.csv")
print("\n✅ Exported monthly_summary.csv")
print("✅ comparison_report.py complete!")
