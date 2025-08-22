#!/usr/bin/env python3
"""
Comprehensive analysis of county land cover proportions from NLCD data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('county_landcover_proportions.csv')

# Add state FIPS code (first 2 digits of county FIPS)
df['state_fips'] = df['county_fips'].astype(str).str.zfill(5).str[:2]

# Filter out counties with no data
df_valid = df[(df[['forest_proportion', 'agriculture_proportion', 'developed_proportion', 
                   'wetland_proportion', 'other_proportion']].sum(axis=1) > 0)].copy()

print("=" * 80)
print("LAND COVER ANALYSIS REPORT")
print("=" * 80)
print(f"\nTotal counties in dataset: {len(df):,}")
print(f"Counties with valid data: {len(df_valid):,}")
print(f"Counties with no data: {len(df) - len(df_valid):,}")

# Basic statistics
print("\n" + "=" * 80)
print("NATIONAL SUMMARY STATISTICS")
print("=" * 80)

land_cover_cols = ['forest_proportion', 'agriculture_proportion', 'developed_proportion', 
                   'wetland_proportion', 'other_proportion']

stats_df = df_valid[land_cover_cols].describe()
stats_df.loc['sum'] = df_valid[land_cover_cols].sum()
stats_df.columns = ['Forest', 'Agriculture', 'Developed', 'Wetland', 'Other']

print("\nLand Cover Proportions Statistics (across all counties):")
print(stats_df.round(4))

# Calculate weighted averages by assuming equal county areas (more accurate would use actual areas)
print("\n" + "-" * 60)
print("DOMINANT LAND COVER TYPES")
print("-" * 60)

# Find dominant land cover for each county
df_valid['dominant_cover'] = df_valid[land_cover_cols].idxmax(axis=1).str.replace('_proportion', '')
dominant_counts = df_valid['dominant_cover'].value_counts()

print("\nCounties by Dominant Land Cover Type:")
for cover, count in dominant_counts.items():
    percentage = (count / len(df_valid)) * 100
    print(f"  {cover.capitalize():12} : {count:4} counties ({percentage:5.1f}%)")

# Extreme values analysis
print("\n" + "=" * 80)
print("COUNTIES WITH EXTREME VALUES")
print("=" * 80)

def get_top_counties(df, column, n=5, ascending=False):
    """Get top n counties for a given column."""
    sorted_df = df.nlargest(n, column) if not ascending else df.nsmallest(n, column)
    return sorted_df[['county_fips', column]]

# Most forested counties
print("\nMost Forested Counties:")
top_forest = get_top_counties(df_valid, 'forest_proportion')
for idx, row in top_forest.iterrows():
    print(f"  {row['county_fips']:5} : {row['forest_proportion']*100:5.1f}%")

# Most agricultural counties
print("\nMost Agricultural Counties:")
top_ag = get_top_counties(df_valid, 'agriculture_proportion')
for idx, row in top_ag.iterrows():
    print(f"  {row['county_fips']:5} : {row['agriculture_proportion']*100:5.1f}%")

# Most developed counties
print("\nMost Developed Counties:")
top_dev = get_top_counties(df_valid, 'developed_proportion')
for idx, row in top_dev.iterrows():
    print(f"  {row['county_fips']:5} : {row['developed_proportion']*100:5.1f}%")

# Most wetland counties
print("\nCounties with Most Wetlands:")
top_wet = get_top_counties(df_valid, 'wetland_proportion')
for idx, row in top_wet.iterrows():
    print(f"  {row['county_fips']:5} : {row['wetland_proportion']*100:5.1f}%")

# State-level analysis
print("\n" + "=" * 80)
print("STATE-LEVEL ANALYSIS")
print("=" * 80)

# State FIPS to name mapping (partial list of major states)
state_names = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas', '06': 'California',
    '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '11': 'DC', '12': 'Florida',
    '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois', '18': 'Indiana',
    '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
    '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota', '28': 'Mississippi',
    '29': 'Missouri', '30': 'Montana', '31': 'Nebraska', '32': 'Nevada', '33': 'New Hampshire',
    '34': 'New Jersey', '35': 'New Mexico', '36': 'New York', '37': 'North Carolina', '38': 'North Dakota',
    '39': 'Ohio', '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island',
    '45': 'South Carolina', '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas', '49': 'Utah',
    '50': 'Vermont', '51': 'Virginia', '53': 'Washington', '54': 'West Virginia', '55': 'Wisconsin',
    '56': 'Wyoming'
}

# Calculate state averages
state_summary = df_valid.groupby('state_fips')[land_cover_cols].mean()
state_summary['county_count'] = df_valid.groupby('state_fips').size()

# Add state names
state_summary['state_name'] = state_summary.index.map(state_names).fillna('Unknown')

# Most forested states
print("\nMost Forested States (average across counties):")
top_forest_states = state_summary.nlargest(10, 'forest_proportion')[['state_name', 'forest_proportion', 'county_count']]
for idx, row in top_forest_states.iterrows():
    print(f"  {row['state_name']:20} : {row['forest_proportion']*100:5.1f}% ({int(row['county_count'])} counties)")

# Most agricultural states
print("\nMost Agricultural States (average across counties):")
top_ag_states = state_summary.nlargest(10, 'agriculture_proportion')[['state_name', 'agriculture_proportion', 'county_count']]
for idx, row in top_ag_states.iterrows():
    print(f"  {row['state_name']:20} : {row['agriculture_proportion']*100:5.1f}% ({int(row['county_count'])} counties)")

# Most developed states
print("\nMost Developed States (average across counties):")
top_dev_states = state_summary.nlargest(10, 'developed_proportion')[['state_name', 'developed_proportion', 'county_count']]
for idx, row in top_dev_states.iterrows():
    print(f"  {row['state_name']:20} : {row['developed_proportion']*100:5.1f}% ({int(row['county_count'])} counties)")

# Distribution analysis
print("\n" + "=" * 80)
print("DISTRIBUTION CHARACTERISTICS")
print("=" * 80)

# Calculate skewness and kurtosis
from scipy import stats

print("\nDistribution Skewness (0 = normal, >0 = right-skewed, <0 = left-skewed):")
for col in land_cover_cols:
    skew = stats.skew(df_valid[col])
    print(f"  {col.replace('_proportion', '').capitalize():12} : {skew:6.2f}")

print("\nDistribution Kurtosis (3 = normal, >3 = heavy-tailed, <3 = light-tailed):")
for col in land_cover_cols:
    kurt = stats.kurtosis(df_valid[col], fisher=False)
    print(f"  {col.replace('_proportion', '').capitalize():12} : {kurt:6.2f}")

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

corr_matrix = df_valid[land_cover_cols].corr()
print("\nCorrelation Matrix (strong negative = land uses compete for space):")
print(corr_matrix.round(3))

# Find strongest correlations
print("\nStrongest Correlations:")
corr_pairs = []
for i in range(len(land_cover_cols)):
    for j in range(i+1, len(land_cover_cols)):
        corr_pairs.append((land_cover_cols[i].replace('_proportion', ''),
                          land_cover_cols[j].replace('_proportion', ''),
                          corr_matrix.iloc[i, j]))

corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for cover1, cover2, corr in corr_pairs[:5]:
    print(f"  {cover1:12} vs {cover2:12} : {corr:6.3f}")

# Regional patterns
print("\n" + "=" * 80)
print("REGIONAL PATTERNS")
print("=" * 80)

# Define regions
regions = {
    'Northeast': ['09', '23', '25', '33', '44', '50', '34', '36', '42'],
    'Southeast': ['10', '11', '12', '13', '24', '37', '45', '51', '54', '01', '21', '28', '47'],
    'Midwest': ['17', '18', '26', '39', '55', '19', '20', '27', '29', '31', '38', '46'],
    'Southwest': ['04', '35', '40', '48'],
    'West': ['02', '06', '08', '15', '16', '30', '32', '41', '49', '53', '56']
}

# Create region mapping
region_map = {}
for region, states in regions.items():
    for state in states:
        region_map[state] = region

df_valid['region'] = df_valid['state_fips'].map(region_map).fillna('Other')

# Regional summaries
regional_summary = df_valid.groupby('region')[land_cover_cols].mean()
print("\nRegional Land Cover Averages:")
print(regional_summary.round(3))

# Save summary statistics to file
print("\n" + "=" * 80)
print("SAVING ANALYSIS RESULTS")
print("=" * 80)

# Save detailed statistics
with open('landcover_analysis_report.txt', 'w') as f:
    f.write("LAND COVER ANALYSIS DETAILED REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("Summary Statistics:\n")
    f.write(stats_df.to_string() + "\n\n")
    
    f.write("State-Level Summary:\n")
    f.write(state_summary.to_string() + "\n\n")
    
    f.write("Regional Summary:\n")
    f.write(regional_summary.to_string() + "\n\n")
    
    f.write("Correlation Matrix:\n")
    f.write(corr_matrix.to_string() + "\n")

print("Detailed analysis saved to: landcover_analysis_report.txt")

# Export key summaries to CSV
state_summary.to_csv('state_landcover_summary.csv')
regional_summary.to_csv('regional_landcover_summary.csv')
print("State summary saved to: state_landcover_summary.csv")
print("Regional summary saved to: regional_landcover_summary.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)