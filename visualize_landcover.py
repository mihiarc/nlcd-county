#!/usr/bin/env python3
"""
Create visualizations of county land cover proportions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('county_landcover_proportions.csv')

# Add state FIPS code
df['state_fips'] = df['county_fips'].astype(str).str.zfill(5).str[:2]

# Filter out counties with no data
df_valid = df[(df[['forest_proportion', 'agriculture_proportion', 'developed_proportion', 
                   'wetland_proportion', 'other_proportion']].sum(axis=1) > 0)].copy()

land_cover_cols = ['forest_proportion', 'agriculture_proportion', 'developed_proportion', 
                   'wetland_proportion', 'other_proportion']

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Distribution histograms for each land cover type
ax1 = fig.add_subplot(gs[0, :])
data_for_hist = []
labels_for_hist = []
for col in land_cover_cols:
    data_for_hist.append(df_valid[col].values * 100)
    labels_for_hist.append(col.replace('_proportion', '').capitalize())

bp = ax1.boxplot(data_for_hist, labels=labels_for_hist, patch_artist=True, 
                  showmeans=True, meanline=True)
colors = sns.color_palette("husl", 5)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_ylabel('Percentage of County Area (%)')
ax1.set_title('Distribution of Land Cover Types Across US Counties', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Correlation heatmap
ax2 = fig.add_subplot(gs[1, 0])
corr_matrix = df_valid[land_cover_cols].corr()
corr_labels = [col.replace('_proportion', '').capitalize() for col in land_cover_cols]
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=corr_labels, yticklabels=corr_labels, ax=ax2,
            vmin=-1, vmax=1, square=True)
ax2.set_title('Land Cover Type Correlations', fontsize=12, fontweight='bold')

# 3. Dominant land cover pie chart
ax3 = fig.add_subplot(gs[1, 1])
df_valid['dominant_cover'] = df_valid[land_cover_cols].idxmax(axis=1).str.replace('_proportion', '')
dominant_counts = df_valid['dominant_cover'].value_counts()
colors_pie = sns.color_palette("husl", len(dominant_counts))
wedges, texts, autotexts = ax3.pie(dominant_counts.values, 
                                    labels=[label.capitalize() for label in dominant_counts.index],
                                    autopct='%1.1f%%', startangle=90,
                                    colors=colors_pie)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax3.set_title('Counties by Dominant Land Cover Type', fontsize=12, fontweight='bold')

# 4. Regional comparison
ax4 = fig.add_subplot(gs[1, 2])
regions = {
    'Northeast': ['09', '23', '25', '33', '44', '50', '34', '36', '42'],
    'Southeast': ['10', '11', '12', '13', '24', '37', '45', '51', '54', '01', '21', '28', '47'],
    'Midwest': ['17', '18', '26', '39', '55', '19', '20', '27', '29', '31', '38', '46'],
    'Southwest': ['04', '35', '40', '48'],
    'West': ['02', '06', '08', '15', '16', '30', '32', '41', '49', '53', '56']
}
region_map = {}
for region, states in regions.items():
    for state in states:
        region_map[state] = region
df_valid['region'] = df_valid['state_fips'].map(region_map).fillna('Other')

regional_means = df_valid.groupby('region')[land_cover_cols].mean() * 100
regional_means.columns = [col.replace('_proportion', '').capitalize() for col in regional_means.columns]
regional_means.plot(kind='bar', stacked=True, ax=ax4, colormap='Set3')
ax4.set_xlabel('Region')
ax4.set_ylabel('Average Percentage (%)')
ax4.set_title('Regional Land Cover Composition', fontsize=12, fontweight='bold')
ax4.legend(title='Land Cover', bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

# 5. Top 10 states by forest coverage
ax5 = fig.add_subplot(gs[2, 0])
state_names = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
    '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
    '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
    '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
    '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
    '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
    '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI', '56': 'WY'
}
state_forest = df_valid.groupby('state_fips')['forest_proportion'].mean() * 100
state_forest.index = state_forest.index.map(state_names).fillna('??')
top_forest = state_forest.nlargest(10)
top_forest.plot(kind='barh', ax=ax5, color='forestgreen', alpha=0.7)
ax5.set_xlabel('Average Forest Coverage (%)')
ax5.set_ylabel('State')
ax5.set_title('Top 10 Most Forested States', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Top 10 states by agricultural coverage
ax6 = fig.add_subplot(gs[2, 1])
state_ag = df_valid.groupby('state_fips')['agriculture_proportion'].mean() * 100
state_ag.index = state_ag.index.map(state_names).fillna('??')
top_ag = state_ag.nlargest(10)
top_ag.plot(kind='barh', ax=ax6, color='goldenrod', alpha=0.7)
ax6.set_xlabel('Average Agricultural Coverage (%)')
ax6.set_ylabel('State')
ax6.set_title('Top 10 Most Agricultural States', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Top 10 states by developed coverage
ax7 = fig.add_subplot(gs[2, 2])
state_dev = df_valid.groupby('state_fips')['developed_proportion'].mean() * 100
state_dev.index = state_dev.index.map(state_names).fillna('??')
top_dev = state_dev.nlargest(10)
top_dev.plot(kind='barh', ax=ax7, color='gray', alpha=0.7)
ax7.set_xlabel('Average Developed Coverage (%)')
ax7.set_ylabel('State')
ax7.set_title('Top 10 Most Developed States', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. Scatter plot: Development vs Agriculture
ax8 = fig.add_subplot(gs[3, 0])
ax8.scatter(df_valid['agriculture_proportion']*100, df_valid['developed_proportion']*100, 
           alpha=0.3, s=10, c='blue')
ax8.set_xlabel('Agricultural Coverage (%)')
ax8.set_ylabel('Developed Coverage (%)')
ax8.set_title('Development vs Agriculture Trade-off', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df_valid['agriculture_proportion'], df_valid['developed_proportion'], 1)
p = np.poly1d(z)
x_trend = np.linspace(0, 1, 100)
ax8.plot(x_trend*100, p(x_trend)*100, "r--", alpha=0.5, label=f'Trend (corr={df_valid[["agriculture_proportion", "developed_proportion"]].corr().iloc[0,1]:.2f})')
ax8.legend()

# 9. Scatter plot: Forest vs Agriculture
ax9 = fig.add_subplot(gs[3, 1])
ax9.scatter(df_valid['forest_proportion']*100, df_valid['agriculture_proportion']*100, 
           alpha=0.3, s=10, c='green')
ax9.set_xlabel('Forest Coverage (%)')
ax9.set_ylabel('Agricultural Coverage (%)')
ax9.set_title('Forest vs Agriculture Trade-off', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df_valid['forest_proportion'], df_valid['agriculture_proportion'], 1)
p = np.poly1d(z)
x_trend = np.linspace(0, 1, 100)
ax9.plot(x_trend*100, p(x_trend)*100, "r--", alpha=0.5, label=f'Trend (corr={df_valid[["forest_proportion", "agriculture_proportion"]].corr().iloc[0,1]:.2f})')
ax9.legend()

# 10. National composition summary
ax10 = fig.add_subplot(gs[3, 2])
national_means = df_valid[land_cover_cols].mean() * 100
national_means.index = [idx.replace('_proportion', '').capitalize() for idx in national_means.index]
colors_bar = sns.color_palette("husl", len(national_means))
bars = ax10.bar(range(len(national_means)), national_means.values, color=colors_bar, alpha=0.7)
ax10.set_xticks(range(len(national_means)))
ax10.set_xticklabels(national_means.index, rotation=45, ha='right')
ax10.set_ylabel('Average Coverage (%)')
ax10.set_title('National Average Land Cover', fontsize=12, fontweight='bold')
ax10.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, national_means.values):
    height = bar.get_height()
    ax10.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

# Main title
fig.suptitle('US County Land Cover Analysis - NLCD 2024', fontsize=16, fontweight='bold', y=0.995)

# Save figure
plt.tight_layout()
plt.savefig('landcover_analysis_visualization.png', dpi=150, bbox_inches='tight')
print("Visualization saved as: landcover_analysis_visualization.png")

plt.show()