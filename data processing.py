"""
data_processing.py
Data processing & preparation for the Mobile Price Classification dataset.

Put this file in the SAME folder as train.csv, then run it.
Output goes to a new folder called "processed_data" (train.csv is never modified).
It creates 3 files: train_cleaned.csv, X.csv, y.csv

Requires: pip install pandas numpy
"""
import os
import pandas as pd

# ---------- Paths (relative to this file, so no editing needed) ----------
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "train.csv")
OUT = os.path.join(BASE, "processed_data")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(SRC)

# ---------- 1. Inspect ----------
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nTarget distribution:\n", df["price_range"].value_counts().sort_index())

# ---------- 2. Invalid values ----------
print("\nInvalid values found:")
print("  px_height == 0 :", (df["px_height"] == 0).sum(), "rows (impossible)")
print("  sc_w == 0      :", (df["sc_w"] == 0).sum(), "rows (impossible)")

# ---------- 3. Clean ----------
df = df[df["px_height"] > 0].copy()                  # drop 2 invalid rows
sc_median = df.loc[df["sc_w"] > 0, "sc_w"].median()  # median of valid widths
df["sc_w"] = df["sc_w"].replace(0, sc_median)        # replace 0 with median
df = df.reset_index(drop=True)

assert df.isna().sum().sum() == 0
assert df.duplicated().sum() == 0
assert (df["px_height"] > 0).all() and (df["sc_w"] > 0).all()
print("\nCleaned shape:", df.shape)

# ---------- 4. Separate specifications (X) from price_range (y) ----------
X = df.drop(columns="price_range")
y = df["price_range"]

# ---------- 5. Save ----------
df.to_csv(os.path.join(OUT, "train_cleaned.csv"), index=False)
X.to_csv(os.path.join(OUT, "X.csv"), index=False)
y.to_csv(os.path.join(OUT, "y.csv"), index=False)

print("X:", X.shape, "| y:", y.shape)
print("Saved to:", OUT)