"""
product_analyzer.py
Part A: E-Commerce Product Analyzer
Day 14 | AM Take-Home Assignment
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. Create DataFrame with 20+ products
# ─────────────────────────────────────────────

data = {
    "name": [
        "Samsung Galaxy S24", "Apple MacBook Air", "Sony WH-1000XM5",
        "Dell Monitor 27\"", "Logitech MX Master 3", "Levi's Jeans 501",
        "Nike Air Max 270", "Adidas Ultraboost 22", "Zara Blazer",
        "H&M Casual Shirt", "Python Crash Course", "Atomic Habits",
        "The Pragmatic Programmer", "Clean Code", "Harry Potter Box Set",
        "Instant Pot Duo 7-in-1", "Philips Air Fryer", "Dyson V15 Detect",
        "IKEA KALLAX Shelf", "Prestige Pressure Cooker", "Boat Rockerz 450",
        "OnePlus Nord CE 3"
    ],
    "category": [
        "Electronics", "Electronics", "Electronics",
        "Electronics", "Electronics", "Clothing",
        "Clothing", "Clothing", "Clothing",
        "Clothing", "Books", "Books",
        "Books", "Books", "Books",
        "Home", "Home", "Home",
        "Home", "Home", "Electronics",
        "Electronics"
    ],
    "price": [
        74999, 114900, 29990,
        22999, 9995, 3499,
        8995, 12999, 5499,
        1299, 499, 399,
        1899, 1799, 3999,
        8999, 10499, 43900,
        6999, 1299, 1799,
        24999
    ],
    "stock": [
        120, 45, 200,
        80, 300, 500,
        250, 180, 90,
        420, 600, 750,
        310, 280, 150,
        95, 130, 60,
        40, 350, 400,
        110
    ],
    "rating": [
        4.5, 4.7, 4.6,
        4.3, 4.8, 4.1,
        4.4, 4.6, 3.9,
        3.7, 4.8, 4.9,
        4.7, 4.6, 4.8,
        4.5, 4.4, 4.7,
        4.0, 4.2, 4.3,
        4.4
    ],
    "num_reviews": [
        1520, 860, 3400,
        540, 2100, 320,
        980, 670, 210,
        85, 4500, 8900,
        1200, 2300, 3100,
        750, 610, 420,
        190, 280, 1100,
        730
    ],
}

df = pd.DataFrame(data)

# ─────────────────────────────────────────────
# 2. "First 5 Minutes" Checklist
# ─────────────────────────────────────────────

print("=" * 60)
print("FIRST 5 MINUTES CHECKLIST")
print("=" * 60)

print("\n📐 Shape:", df.shape)
print("\n📋 Columns:", df.columns.tolist())
print("\n🔍 First 5 rows:")
print(df.head())
print("\n📊 Info:")
df.info()
print("\n📈 Describe:")
print(df.describe())
print("\n❓ Null values:")
print(df.isnull().sum())
print("\n📦 Data types:")
print(df.dtypes)

# ─────────────────────────────────────────────
# 3. .loc[] operations
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print(".LOC[] OPERATIONS")
print("=" * 60)

# (a) Select all Electronics
electronics = df.loc[df["category"] == "Electronics"]
print("\n(a) All Electronics:")
print(electronics[["name", "price", "rating"]])

# (b) Products rated > 4.0 with price < 5000
good_value = df.loc[(df["rating"] > 4.0) & (df["price"] < 5000)]
print("\n(b) Rated > 4.0 AND Price < 5000:")
print(good_value[["name", "price", "rating"]])

# (c) Update stock for a specific product (Atomic Habits)
df.loc[df["name"] == "Atomic Habits", "stock"] = 900
print("\n(c) Updated stock for 'Atomic Habits':")
print(df.loc[df["name"] == "Atomic Habits", ["name", "stock"]])

# ─────────────────────────────────────────────
# 4. .iloc[] operations
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print(".ILOC[] OPERATIONS")
print("=" * 60)

# (a) First 5 and last 5 products
print("\n(a) First 5 products:")
print(df.iloc[:5][["name", "category", "price"]])

print("\n(a) Last 5 products:")
print(df.iloc[-5:][["name", "category", "price"]])

# (b) Every other row
print("\n(b) Every other row:")
print(df.iloc[::2][["name", "price"]])

# (c) Rows 10-15, columns 0-3
print("\n(c) Rows 10-15, columns 0-3:")
print(df.iloc[10:16, 0:4])

# ─────────────────────────────────────────────
# 5. Create 3 filtered DataFrames
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("FILTERED DATAFRAMES")
print("=" * 60)

budget_products   = df[df["price"] < 1000]
premium_products  = df[df["price"] > 10000]
popular_products  = df[(df["num_reviews"] > 100) & (df["rating"] > 4.0)]

print(f"\nBudget products  (price < 1000)           : {len(budget_products)} products")
print(f"Premium products (price > 10000)          : {len(premium_products)} products")
print(f"Popular products (reviews > 100, rating > 4.0): {len(popular_products)} products")

# ─────────────────────────────────────────────
# 6. Export each filtered DataFrame to CSV
# ─────────────────────────────────────────────

filtered_dfs = {
    "budget_products": budget_products,
    "premium_products": premium_products,
    "popular_products": popular_products,
}

for filename, filtered_df in filtered_dfs.items():
    path = f"{filename}.csv"
    filtered_df.to_csv(path, index=False)
    print(f"✅ Exported {filename}.csv  ({len(filtered_df)} rows)")

print("\n✅ product_analyzer.py complete!")
