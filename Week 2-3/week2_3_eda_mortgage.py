import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots to files
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

INTERIM_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'interim')
PLOTS_PATH = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_PATH, exist_ok=True)

# ─── Load combined datasets from Week 1 ───────────────────────────────────────
sold = pd.read_csv(os.path.join(INTERIM_PATH, 'sold_combined.csv'), low_memory=False)
listings = pd.read_csv(os.path.join(INTERIM_PATH, 'listings_combined.csv'), low_memory=False)

print("=" * 60)
print("SOLD DATASET STRUCTURE")
print("=" * 60)
print(f"Shape: {sold.shape[0]} rows x {sold.shape[1]} columns")
print(f"\nColumn data types:\n{sold.dtypes.to_string()}")

print("\n" + "=" * 60)
print("LISTINGS DATASET STRUCTURE")
print("=" * 60)
print(f"Shape: {listings.shape[0]} rows x {listings.shape[1]} columns")
print(f"\nColumn data types:\n{listings.dtypes.to_string()}")

# ─── Suggested Intern Questions ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUGGESTED INTERN QUESTIONS (Sold dataset)")
print("=" * 60)

# All data is already Residential (filtered in Week 1)
print(f"\nAll {len(sold)} sold records are PropertyType == 'Residential'")

if 'ClosePrice' in sold.columns:
    sold['ClosePrice'] = pd.to_numeric(sold['ClosePrice'], errors='coerce')
    print(f"\nMedian close price:  ${sold['ClosePrice'].median():,.0f}")
    print(f"Mean close price:    ${sold['ClosePrice'].mean():,.0f}")

if 'DaysOnMarket' in sold.columns:
    sold['DaysOnMarket'] = pd.to_numeric(sold['DaysOnMarket'], errors='coerce')
    print(f"\nDays on Market distribution:")
    print(sold['DaysOnMarket'].describe().to_string())

if 'ClosePrice' in sold.columns and 'ListPrice' in sold.columns:
    sold['ListPrice'] = pd.to_numeric(sold['ListPrice'], errors='coerce')
    above = (sold['ClosePrice'] > sold['ListPrice']).sum()
    below = (sold['ClosePrice'] < sold['ListPrice']).sum()
    total_valid = sold[['ClosePrice', 'ListPrice']].dropna().shape[0]
    print(f"\nSold above list price:  {above:,} ({above/total_valid*100:.1f}%)")
    print(f"Sold below list price:  {below:,} ({below/total_valid*100:.1f}%)")

if 'CountyOrParish' in sold.columns:
    print(f"\nTop 10 counties by median close price:")
    county_median = (sold.groupby('CountyOrParish')['ClosePrice']
                     .median().sort_values(ascending=False).head(10))
    print(county_median.to_string())

# ─── Missing Value Analysis ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS — SOLD")
print("=" * 60)

null_counts = sold.isnull().sum()
null_pct = (null_counts / len(sold) * 100).round(2)
missing_report = pd.DataFrame({'null_count': null_counts, 'null_pct': null_pct})
missing_report = missing_report[missing_report['null_count'] > 0].sort_values('null_pct', ascending=False)
print(missing_report.to_string())

high_missing_sold = missing_report[missing_report['null_pct'] > 90].index.tolist()
print(f"\nColumns with >90% missing values (sold): {high_missing_sold}")

missing_report.to_csv(os.path.join(os.path.dirname(__file__), 'missing_value_report_sold.csv'))

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS — LISTINGS")
print("=" * 60)

null_counts_l = listings.isnull().sum()
null_pct_l = (null_counts_l / len(listings) * 100).round(2)
missing_report_l = pd.DataFrame({'null_count': null_counts_l, 'null_pct': null_pct_l})
missing_report_l = missing_report_l[missing_report_l['null_count'] > 0].sort_values('null_pct', ascending=False)
print(missing_report_l.to_string())

high_missing_listings = missing_report_l[missing_report_l['null_pct'] > 90].index.tolist()
print(f"\nColumns with >90% missing values (listings): {high_missing_listings}")

missing_report_l.to_csv(os.path.join(os.path.dirname(__file__), 'missing_value_report_listings.csv'))

# ─── Numeric Distribution Review ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("NUMERIC DISTRIBUTION REVIEW — SOLD")
print("=" * 60)

numeric_fields = ['ClosePrice', 'LivingArea', 'DaysOnMarket',
                  'ListPrice', 'OriginalListPrice', 'LotSizeAcres',
                  'BedroomsTotal', 'BathroomsTotalInteger', 'YearBuilt']

for col in numeric_fields:
    if col in sold.columns:
        sold[col] = pd.to_numeric(sold[col], errors='coerce')

dist_summary = sold[numeric_fields].describe(percentiles=[.05, .25, .5, .75, .95]).T
print(dist_summary.to_string())

# Histograms and boxplots for key fields
key_fields = ['ClosePrice', 'LivingArea', 'DaysOnMarket']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Numeric Distribution Review — Sold Dataset', fontsize=14)

for i, col in enumerate(key_fields):
    if col not in sold.columns:
        continue
    data = sold[col].dropna()
    # Histogram
    axes[0, i].hist(data, bins=50, edgecolor='none', color='steelblue')
    axes[0, i].set_title(f'{col} — Histogram')
    axes[0, i].set_xlabel(col)
    axes[0, i].set_ylabel('Count')
    # Boxplot
    axes[1, i].boxplot(data, vert=False, patch_artist=True,
                       boxprops=dict(facecolor='steelblue', alpha=0.6))
    axes[1, i].set_title(f'{col} — Boxplot')
    axes[1, i].set_xlabel(col)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'numeric_distributions.png'), dpi=100)
plt.close()
print(f"\nPlot saved: {PLOTS_PATH}/numeric_distributions.png")

# Percentile summaries for ClosePrice, LivingArea, DaysOnMarket
print("\nPercentile summary (ClosePrice, LivingArea, DaysOnMarket):")
pct_summary = sold[key_fields].describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]).T
print(pct_summary.to_string())

# ─── Mortgage Rate Enrichment ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MORTGAGE RATE ENRICHMENT (FRED MORTGAGE30US)")
print("=" * 60)

# Step 1 — Fetch weekly mortgage rate data from FRED
fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
print("Fetching MORTGAGE30US from FRED...")
mortgage = pd.read_csv(fred_url)
mortgage.columns = ['date', 'rate_30yr_fixed']
mortgage['date'] = pd.to_datetime(mortgage['date'], errors='coerce')
# FRED encodes missing values as '.' — drop them
mortgage = mortgage[mortgage['rate_30yr_fixed'] != '.'].copy()
mortgage['rate_30yr_fixed'] = pd.to_numeric(mortgage['rate_30yr_fixed'], errors='coerce')
mortgage.dropna(subset=['rate_30yr_fixed'], inplace=True)
print(f"Fetched {len(mortgage)} weekly observations ({mortgage['date'].min().date()} to {mortgage['date'].max().date()})")

# Step 2 — Resample weekly rates to monthly averages
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (
    mortgage.groupby('year_month')['rate_30yr_fixed']
    .mean()
    .reset_index()
)
print(f"Resampled to {len(mortgage_monthly)} monthly averages")

# Step 3 — Create year_month join keys on the MLS datasets
sold['year_month'] = pd.to_datetime(sold['CloseDate'], errors='coerce').dt.to_period('M')
listings['year_month'] = pd.to_datetime(listings['ListingContractDate'], errors='coerce').dt.to_period('M')

# Step 4 — Merge
sold_with_rates = sold.merge(mortgage_monthly, on='year_month', how='left')
listings_with_rates = listings.merge(mortgage_monthly, on='year_month', how='left')

# Convert Period to string for CSV compatibility
sold_with_rates['year_month'] = sold_with_rates['year_month'].astype(str)
listings_with_rates['year_month'] = listings_with_rates['year_month'].astype(str)

# Step 5 — Validate the merge
null_rate_sold = sold_with_rates['rate_30yr_fixed'].isnull().sum()
null_rate_listings = listings_with_rates['rate_30yr_fixed'].isnull().sum()
print(f"\nNull rate_30yr_fixed after merge — sold:     {null_rate_sold}")
print(f"Null rate_30yr_fixed after merge — listings: {null_rate_listings}")

print("\nSample (sold):")
print(sold_with_rates[['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']].head(10).to_string())

# ─── Save enriched datasets ────────────────────────────────────────────────────
sold_out = os.path.join(INTERIM_PATH, 'sold_with_rates.csv')
listings_out = os.path.join(INTERIM_PATH, 'listings_with_rates.csv')

sold_with_rates.to_csv(sold_out, index=False)
listings_with_rates.to_csv(listings_out, index=False)

print(f"\nSaved: {sold_out}")
print(f"Saved: {listings_out}")
