import pandas as pd
import numpy as np
import os

INTERIM_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'interim')
FINAL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'final')
os.makedirs(FINAL_PATH, exist_ok=True)

# ─── Load engineered dataset from Week 6 ──────────────────────────────────────
sold = pd.read_csv(os.path.join(INTERIM_PATH, 'sold_engineered.csv'), low_memory=False)
print(f"Loaded sold_engineered: {len(sold):,} rows x {sold.shape[1]} columns")

for col in ['ClosePrice', 'LivingArea', 'DaysOnMarket']:
    if col in sold.columns:
        sold[col] = pd.to_numeric(sold[col], errors='coerce')

# ─── IQR Outlier Detection ─────────────────────────────────────────────────────
TARGET_FIELDS = ['ClosePrice', 'LivingArea', 'DaysOnMarket']

def compute_iqr_bounds(series):
    """Return (lower, upper) IQR bounds for a numeric series."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper

print("\n" + "=" * 60)
print("IQR BOUNDS")
print("=" * 60)

bounds = {}
for col in TARGET_FIELDS:
    if col not in sold.columns:
        continue
    lower, upper = compute_iqr_bounds(sold[col].dropna())
    bounds[col] = (lower, upper)
    print(f"{col:25s}  lower={lower:>12,.2f}  upper={upper:>12,.2f}")

# ─── Add outlier flag columns (tiered approach: flag, don't delete) ───────────
print("\n[1] Adding outlier flag columns...")

# Always-invalid business rules (ClosePrice <= 0 is never valid)
if 'ClosePrice' in sold.columns:
    sold['outlier_flag_close_price_invalid'] = sold['ClosePrice'] <= 0
    lower, upper = bounds.get('ClosePrice', (None, None))
    if lower is not None:
        sold['outlier_flag_close_price_iqr'] = (
            (sold['ClosePrice'] < lower) | (sold['ClosePrice'] > upper)
        )
        print(f"  outlier_flag_close_price_invalid: {sold['outlier_flag_close_price_invalid'].sum():,}")
        print(f"  outlier_flag_close_price_iqr:     {sold['outlier_flag_close_price_iqr'].sum():,}")

if 'LivingArea' in sold.columns:
    sold['outlier_flag_living_area_invalid'] = sold['LivingArea'] <= 0
    lower, upper = bounds.get('LivingArea', (None, None))
    if lower is not None:
        sold['outlier_flag_living_area_iqr'] = (
            (sold['LivingArea'] < lower) | (sold['LivingArea'] > upper)
        )
        print(f"  outlier_flag_living_area_invalid: {sold['outlier_flag_living_area_invalid'].sum():,}")
        print(f"  outlier_flag_living_area_iqr:     {sold['outlier_flag_living_area_iqr'].sum():,}")

if 'DaysOnMarket' in sold.columns:
    sold['outlier_flag_dom_invalid'] = sold['DaysOnMarket'] < 0
    lower, upper = bounds.get('DaysOnMarket', (None, None))
    if lower is not None:
        sold['outlier_flag_dom_iqr'] = (
            (sold['DaysOnMarket'] < lower) | (sold['DaysOnMarket'] > upper)
        )
        print(f"  outlier_flag_dom_invalid:         {sold['outlier_flag_dom_invalid'].sum():,}")
        print(f"  outlier_flag_dom_iqr:             {sold['outlier_flag_dom_iqr'].sum():,}")

# Composite: record is flagged if any IQR or invalid flag is set
iqr_flag_cols = [c for c in sold.columns if c.startswith('outlier_flag_')]
sold['any_outlier_flag'] = sold[iqr_flag_cols].any(axis=1)
print(f"\n  Records with any outlier flag: {sold['any_outlier_flag'].sum():,}"
      f"  ({sold['any_outlier_flag'].mean()*100:.1f}% of total)")

# ─── Comparison: before vs. after filtering ───────────────────────────────────
print("\n" + "=" * 60)
print("BEFORE vs. AFTER IQR FILTERING")
print("=" * 60)

# Build clean filtered dataset: remove records with any IQR or invalid flag
sold_filtered = sold[~sold['any_outlier_flag']].copy()

comparison_rows = []
for col in TARGET_FIELDS:
    if col not in sold.columns:
        continue
    row = {
        'field': col,
        'rows_before': sold[col].count(),
        'median_before': sold[col].median(),
        'mean_before': sold[col].mean(),
        'rows_after': sold_filtered[col].count(),
        'median_after': sold_filtered[col].median(),
        'mean_after': sold_filtered[col].mean(),
    }
    comparison_rows.append(row)

comparison = pd.DataFrame(comparison_rows).set_index('field')
print(comparison.to_string())

print(f"\nTotal rows before filtering: {len(sold):,}")
print(f"Total rows after filtering:  {len(sold_filtered):,}")
print(f"Rows removed:                {len(sold) - len(sold_filtered):,}"
      f"  ({(len(sold) - len(sold_filtered)) / len(sold) * 100:.1f}%)")

# ─── Save both datasets ───────────────────────────────────────────────────────
flagged_out = os.path.join(INTERIM_PATH, 'sold_flagged.csv')
filtered_out = os.path.join(FINAL_PATH, 'sold_clean_filtered.csv')

sold.to_csv(flagged_out, index=False)
sold_filtered.to_csv(filtered_out, index=False)

print(f"\nSaved full flagged dataset:   {flagged_out}")
print(f"Saved clean filtered dataset: {filtered_out}")
