import pandas as pd
import glob
import os

# Paths relative to this script's location
RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'raw')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'interim')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─── Load and concatenate all monthly sold files ───────────────────────────────
sold_files = sorted(glob.glob(os.path.join(RAW_PATH, 'CRMLSSold*.csv')))
print(f"Found {len(sold_files)} sold files")

sold_dfs = []
for f in sold_files:
    df = pd.read_csv(f, low_memory=False)
    # Drop duplicate columns produced by crmls_sold.py (e.g., Latitude.1)
    df = df.loc[:, ~df.columns.duplicated()]
    sold_dfs.append(df)

sold = pd.concat(sold_dfs, ignore_index=True)
print(f"Combined sold rows before Residential filter: {len(sold)}")

# ─── Load and concatenate all monthly listing files ────────────────────────────
listing_files = sorted(glob.glob(os.path.join(RAW_PATH, 'CRMLSListing*.csv')))
print(f"\nFound {len(listing_files)} listing files")

listing_dfs = []
for f in listing_files:
    df = pd.read_csv(f, low_memory=False)
    # Drop duplicate columns produced by crmls_listed.py
    df = df.loc[:, ~df.columns.duplicated()]
    listing_dfs.append(df)

listings = pd.concat(listing_dfs, ignore_index=True)
print(f"Combined listings rows before Residential filter: {len(listings)}")

# ─── Filter to Residential PropertyType only ───────────────────────────────────
sold_residential = sold[sold['PropertyType'] == 'Residential'].copy()
listings_residential = listings[listings['PropertyType'] == 'Residential'].copy()

print(f"\nSold rows after Residential filter:     {len(sold_residential)}"
      f"  (removed {len(sold) - len(sold_residential)})")
print(f"Listings rows after Residential filter: {len(listings_residential)}"
      f"  (removed {len(listings) - len(listings_residential)})")

# ─── Save combined datasets ────────────────────────────────────────────────────
sold_out = os.path.join(OUTPUT_PATH, 'sold_combined.csv')
listings_out = os.path.join(OUTPUT_PATH, 'listings_combined.csv')

sold_residential.to_csv(sold_out, index=False)
listings_residential.to_csv(listings_out, index=False)

print(f"\nSaved: {sold_out}")
print(f"Saved: {listings_out}")
