import pandas as pd
import numpy as np
import os

INTERIM_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'interim')

# ─── Load cleaned sold dataset from Week 4-5 ──────────────────────────────────
sold = pd.read_csv(os.path.join(INTERIM_PATH, 'sold_cleaned.csv'), low_memory=False)
print(f"Loaded sold: {len(sold):,} rows x {sold.shape[1]} columns")

# Re-cast dates (they come back as strings after CSV round-trip)
DATE_FIELDS = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate']
for col in DATE_FIELDS:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors='coerce')

for col in ['ClosePrice', 'OriginalListPrice', 'ListPrice', 'LivingArea', 'DaysOnMarket']:
    if col in sold.columns:
        sold[col] = pd.to_numeric(sold[col], errors='coerce')

# ─── 1. Price Ratio (ClosePrice / OriginalListPrice) ─────────────────────────
#    Measures negotiation strength; > 1.0 means buyer paid over original list
if 'ClosePrice' in sold.columns and 'OriginalListPrice' in sold.columns:
    sold['price_ratio'] = sold['ClosePrice'] / sold['OriginalListPrice']
    print(f"\nprice_ratio — median: {sold['price_ratio'].median():.4f}  "
          f"mean: {sold['price_ratio'].mean():.4f}")

# ─── 2. Close to Original List Ratio ─────────────────────────────────────────
#    Same as price_ratio — captures the full price reduction from original list
#    Aliased here as a named metric per the handbook
sold['close_to_orig_list_ratio'] = sold.get('price_ratio', np.nan)

# ─── 3. Price Per Square Foot (ClosePrice / LivingArea) ──────────────────────
if 'ClosePrice' in sold.columns and 'LivingArea' in sold.columns:
    sold['price_per_sqft'] = sold['ClosePrice'] / sold['LivingArea']
    # Zero or negative LivingArea produces invalid PPSF — set to NaN
    sold.loc[sold['LivingArea'] <= 0, 'price_per_sqft'] = np.nan
    print(f"price_per_sqft — median: ${sold['price_per_sqft'].median():,.2f}  "
          f"mean: ${sold['price_per_sqft'].mean():,.2f}")

# ─── 4. Time-series variables from CloseDate ─────────────────────────────────
if 'CloseDate' in sold.columns:
    sold['close_year'] = sold['CloseDate'].dt.year
    sold['close_month'] = sold['CloseDate'].dt.month
    sold['close_yrmo'] = sold['CloseDate'].dt.to_period('M').astype(str)
    print(f"\nDate range: {sold['CloseDate'].min().date()} to {sold['CloseDate'].max().date()}")
    print(f"Unique year-months: {sold['close_yrmo'].nunique()}")

# ─── 5. Listing to Contract Days ─────────────────────────────────────────────
#    Time from listing to accepted offer
if 'PurchaseContractDate' in sold.columns and 'ListingContractDate' in sold.columns:
    sold['listing_to_contract_days'] = (
        (sold['PurchaseContractDate'] - sold['ListingContractDate']).dt.days
    )
    print(f"\nlisting_to_contract_days — median: {sold['listing_to_contract_days'].median():.0f}  "
          f"mean: {sold['listing_to_contract_days'].mean():.0f}")

# ─── 6. Contract to Close Days ────────────────────────────────────────────────
#    Escrow and closing period duration
if 'CloseDate' in sold.columns and 'PurchaseContractDate' in sold.columns:
    sold['contract_to_close_days'] = (
        (sold['CloseDate'] - sold['PurchaseContractDate']).dt.days
    )
    print(f"contract_to_close_days — median: {sold['contract_to_close_days'].median():.0f}  "
          f"mean: {sold['contract_to_close_days'].mean():.0f}")

# ─── 7. Sample output of new columns ──────────────────────────────────────────
new_cols = ['CloseDate', 'ClosePrice', 'OriginalListPrice', 'LivingArea',
            'price_ratio', 'close_to_orig_list_ratio', 'price_per_sqft',
            'close_year', 'close_month', 'close_yrmo',
            'listing_to_contract_days', 'contract_to_close_days']
available = [c for c in new_cols if c in sold.columns]
print("\nSample output (first 5 rows, engineered columns):")
print(sold[available].head(5).to_string())

# ─── 8. Segment Analysis ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SEGMENT ANALYSIS")
print("=" * 60)

agg_metrics = {
    'ClosePrice': ['median', 'mean', 'count'],
    'price_per_sqft': ['median', 'mean'],
    'DaysOnMarket': ['median', 'mean'],
    'price_ratio': ['median', 'mean'],
}
# Only keep metrics whose columns exist
agg_metrics = {k: v for k, v in agg_metrics.items() if k in sold.columns}

# By PropertySubType
if 'PropertySubType' in sold.columns:
    seg_subtype = sold.groupby('PropertySubType').agg(agg_metrics)
    seg_subtype.columns = ['_'.join(c) for c in seg_subtype.columns]
    seg_subtype = seg_subtype.sort_values('ClosePrice_median', ascending=False)
    print("\nBy PropertySubType:")
    print(seg_subtype.to_string())

# By CountyOrParish
if 'CountyOrParish' in sold.columns:
    seg_county = sold.groupby('CountyOrParish').agg(agg_metrics)
    seg_county.columns = ['_'.join(c) for c in seg_county.columns]
    seg_county = seg_county.sort_values('ClosePrice_median', ascending=False)
    print("\nBy CountyOrParish (top 15 by median close price):")
    print(seg_county.head(15).to_string())

# By MLSAreaMajor
if 'MLSAreaMajor' in sold.columns:
    seg_area = sold.groupby('MLSAreaMajor').agg(agg_metrics)
    seg_area.columns = ['_'.join(c) for c in seg_area.columns]
    seg_area = seg_area.sort_values('ClosePrice_median', ascending=False)
    print("\nBy MLSAreaMajor (top 10 by median close price):")
    print(seg_area.head(10).to_string())

# By ListOfficeName (competitive intelligence — top 20 by volume)
if 'ListOfficeName' in sold.columns:
    seg_office = (sold.groupby('ListOfficeName')
                  .agg(units=('ClosePrice', 'count'),
                       total_volume=('ClosePrice', 'sum'),
                       median_price=('ClosePrice', 'median'))
                  .sort_values('total_volume', ascending=False))
    print("\nTop 20 Listing Offices by Sales Volume:")
    print(seg_office.head(20).to_string())

# ─── Save engineered dataset ──────────────────────────────────────────────────
out_path = os.path.join(INTERIM_PATH, 'sold_engineered.csv')
sold.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Final shape: {sold.shape[0]:,} rows x {sold.shape[1]} columns")
