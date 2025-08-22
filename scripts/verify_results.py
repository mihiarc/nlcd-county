#!/usr/bin/env python3
"""
Quick verification script to spot-check the results
"""

import pandas as pd
import geopandas as gpd
import numpy as np

# Load results
results = pd.read_csv('/home/mihiarc/repos/nlcd-county/county_landcover_proportions.csv')

print("=== NLCD County Land Cover Analysis Results ===\n")

print("1. Dataset Overview:")
print(f"   - Total counties processed: {len(results)}")
print(f"   - Counties with land cover data: {len(results[results.iloc[:, 1:].sum(axis=1) > 0])}")
print(f"   - Counties with no data: {len(results[results.iloc[:, 1:].sum(axis=1) == 0])}")

print("\n2. Data Quality Check:")
proportion_cols = ['forest_proportion', 'agriculture_proportion', 'developed_proportion', 'wetland_proportion', 'other_proportion']
results['total_proportion'] = results[proportion_cols].sum(axis=1)

valid_counties = results[(results['total_proportion'] >= 0.99) & (results['total_proportion'] <= 1.01)]
print(f"   - Counties with valid proportions (sum ≈ 1.0): {len(valid_counties)}/{len(results)} ({len(valid_counties)/len(results)*100:.1f}%)")

print("\n3. National Land Cover Summary (Mean Proportions):")
for col in proportion_cols:
    mean_val = results[col].mean()
    class_name = col.replace('_proportion', '').title()
    print(f"   - {class_name:12}: {mean_val:.4f} ({mean_val*100:.1f}%)")

print("\n4. Sample Counties (Top 5 by different land cover types):")

# Forest counties
top_forest = results.nlargest(5, 'forest_proportion')[['county_fips', 'forest_proportion']]
print("\n   Most Forested Counties:")
for _, row in top_forest.iterrows():
    print(f"     County {row['county_fips']}: {row['forest_proportion']:.3f} forest")

# Agricultural counties
top_ag = results.nlargest(5, 'agriculture_proportion')[['county_fips', 'agriculture_proportion']]
print("\n   Most Agricultural Counties:")
for _, row in top_ag.iterrows():
    print(f"     County {row['county_fips']}: {row['agriculture_proportion']:.3f} agriculture")

# Developed counties
top_dev = results.nlargest(5, 'developed_proportion')[['county_fips', 'developed_proportion']]
print("\n   Most Developed Counties:")
for _, row in top_dev.iterrows():
    print(f"     County {row['county_fips']}: {row['developed_proportion']:.3f} developed")

print("\n5. File Location:")
print(f"   Results saved to: /home/mihiarc/repos/nlcd-county/county_landcover_proportions.csv")

print("\n=== Analysis Complete ===")