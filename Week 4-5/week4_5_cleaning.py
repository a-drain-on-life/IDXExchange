import pandas as pd
import numpy as np
import os

INTERIM_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'interim')

# ─── Load enriched datasets from Week 2-3 ─────────────────────────────────────
sold = pd.read_csv(os.path.join(INTERIM_PATH, 'sold_with_rates.csv'), low_memory=False)
listings = pd.read_csv(os.path.join(INTERIM_PATH, 'listings_with_rates.csv'), low_memory=False)

print(f"Loaded sold:     {len(sold):,} rows x {sold.shape[1]} columns")
print(f"Loaded listings: {len(listings):,} rows x {listings.shape[1]} columns")
rows_before_sold = len(sold)
rows_before_listings = len(listings)

# ─── 1. Convert date fields to datetime ───────────────────────────────────────
DATE_FIELDS = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate',
               'ContractStatusChangeDate']

print("\n[1] Converting date fields to datetime...")
for col in DATE_FIELDS:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors='coerce')
    if col in listings.columns:
        listings[col] = pd.to_datetime(listings[col], errors='coerce')

print("    Date field dtypes (sold):")
for col in DATE_FIELDS:
    if col in sold.columns:
        print(f"      {col}: {sold[col].dtype}")

# ─── 2. Ensure numeric fields are properly typed ──────────────────────────────
NUMERIC_FIELDS = ['ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
                  'LotSizeAcres', 'LotSizeSquareFeet', 'BedroomsTotal',
                  'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt',
                  'GarageSpaces', 'ParkingTotal', 'AssociationFee',
                  'TaxAnnualAmount', 'AboveGradeFinishedArea']

print("\n[2] Ensuring numeric fields are properly typed...")
for col in NUMERIC_FIELDS:
    if col in sold.columns:
        sold[col] = pd.to_numeric(sold[col], errors='coerce')
    if col in listings.columns:
        listings[col] = pd.to_numeric(listings[col], errors='coerce')

# ─── 3. Remove high-missing columns (>90% null) ───────────────────────────────
print("\n[3] Removing columns with >90% missing values...")

def drop_high_missing(df, threshold=0.90, label=''):
    null_pct = df.isnull().mean()
    high_missing = null_pct[null_pct > threshold].index.tolist()
    if high_missing:
        print(f"    Dropping from {label} (>90% null): {high_missing}")
        df = df.drop(columns=high_missing)
    else:
        print(f"    No columns above 90% null threshold in {label}")
    return df

sold = drop_high_missing(sold, label='sold')
listings = drop_high_missing(listings, label='listings')

# ─── 4. Flag and remove invalid numeric values ────────────────────────────────
print("\n[4] Flagging invalid numeric values...")

# ClosePrice <= 0
if 'ClosePrice' in sold.columns:
    sold['flag_invalid_close_price'] = sold['ClosePrice'] <= 0
    print(f"    flag_invalid_close_price (sold):    {sold['flag_invalid_close_price'].sum():,}")

# LivingArea <= 0
if 'LivingArea' in sold.columns:
    sold['flag_invalid_living_area'] = sold['LivingArea'] <= 0
    print(f"    flag_invalid_living_area (sold):    {sold['flag_invalid_living_area'].sum():,}")
    if 'LivingArea' in listings.columns:
        listings['flag_invalid_living_area'] = listings['LivingArea'] <= 0

# DaysOnMarket < 0
if 'DaysOnMarket' in sold.columns:
    sold['flag_negative_dom'] = sold['DaysOnMarket'] < 0
    print(f"    flag_negative_dom (sold):           {sold['flag_negative_dom'].sum():,}")

# Negative bedrooms or bathrooms
if 'BedroomsTotal' in sold.columns:
    sold['flag_negative_beds'] = sold['BedroomsTotal'] < 0
    print(f"    flag_negative_beds (sold):          {sold['flag_negative_beds'].sum():,}")
if 'BathroomsTotalInteger' in sold.columns:
    sold['flag_negative_baths'] = sold['BathroomsTotalInteger'] < 0
    print(f"    flag_negative_baths (sold):         {sold['flag_negative_baths'].sum():,}")

# ─── 5. Date consistency checks ───────────────────────────────────────────────
print("\n[5] Date consistency checks...")

if all(c in sold.columns for c in ['ListingContractDate', 'CloseDate']):
    sold['listing_after_close_flag'] = sold['ListingContractDate'] > sold['CloseDate']
    print(f"    listing_after_close_flag:           {sold['listing_after_close_flag'].sum():,}")

if all(c in sold.columns for c in ['PurchaseContractDate', 'CloseDate']):
    sold['purchase_after_close_flag'] = sold['PurchaseContractDate'] > sold['CloseDate']
    print(f"    purchase_after_close_flag:          {sold['purchase_after_close_flag'].sum():,}")

# negative_timeline_flag: ListingContractDate > PurchaseContractDate
if all(c in sold.columns for c in ['ListingContractDate', 'PurchaseContractDate']):
    sold['negative_timeline_flag'] = sold['ListingContractDate'] > sold['PurchaseContractDate']
    print(f"    negative_timeline_flag:             {sold['negative_timeline_flag'].sum():,}")

# ─── 6. Geographic data quality checks ───────────────────────────────────────
print("\n[6] Geographic data quality checks (sold)...")

if 'Latitude' in sold.columns and 'Longitude' in sold.columns:
    sold['Latitude'] = pd.to_numeric(sold['Latitude'], errors='coerce')
    sold['Longitude'] = pd.to_numeric(sold['Longitude'], errors='coerce')

    sold['flag_missing_coords'] = sold['Latitude'].isnull() | sold['Longitude'].isnull()
    sold['flag_zero_coords'] = (sold['Latitude'] == 0) | (sold['Longitude'] == 0)
    # California longitudes should be negative (approx -114 to -124)
    sold['flag_positive_longitude'] = sold['Longitude'] > 0
    sold['flag_out_of_state'] = (
        sold['Latitude'].notna() & sold['Longitude'].notna() &
        (~sold['flag_missing_coords']) &
        (~sold['flag_zero_coords']) &
        ((sold['Latitude'] < 32) | (sold['Latitude'] > 42) |
         (sold['Longitude'] < -124) | (sold['Longitude'] > -114))
    )

    print(f"    flag_missing_coords:                {sold['flag_missing_coords'].sum():,}")
    print(f"    flag_zero_coords:                   {sold['flag_zero_coords'].sum():,}")
    print(f"    flag_positive_longitude:            {sold['flag_positive_longitude'].sum():,}")
    print(f"    flag_out_of_state:                  {sold['flag_out_of_state'].sum():,}")

if 'Latitude' in listings.columns and 'Longitude' in listings.columns:
    listings['Latitude'] = pd.to_numeric(listings['Latitude'], errors='coerce')
    listings['Longitude'] = pd.to_numeric(listings['Longitude'], errors='coerce')
    listings['flag_missing_coords'] = listings['Latitude'].isnull() | listings['Longitude'].isnull()
    listings['flag_positive_longitude'] = listings['Longitude'] > 0

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)
print(f"Sold    rows: {rows_before_sold:,} → {len(sold):,}  (no rows removed, only flagged)")
print(f"Listings rows: {rows_before_listings:,} → {len(listings):,}")
print(f"\nSold data types after cleaning:")
print(sold.dtypes.to_string())

# ─── Save cleaned datasets ─────────────────────────────────────────────────────
sold_out = os.path.join(INTERIM_PATH, 'sold_cleaned.csv')
listings_out = os.path.join(INTERIM_PATH, 'listings_cleaned.csv')

sold.to_csv(sold_out, index=False)
listings.to_csv(listings_out, index=False)

print(f"\nSaved: {sold_out}")
print(f"Saved: {listings_out}")
