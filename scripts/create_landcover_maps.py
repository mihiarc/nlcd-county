#!/usr/bin/env python3
"""
Create map visualizations of county land cover proportions using the shapefile and analysis results.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
# Load the county shapefile
counties = gpd.read_file('tl_2024_us_county/tl_2024_us_county.shp')

# Load the land cover proportions
landcover_df = pd.read_csv('county_landcover_proportions.csv')

# Convert FIPS to string and ensure proper formatting
counties['GEOID'] = counties['GEOID'].astype(str).str.zfill(5)
landcover_df['county_fips'] = landcover_df['county_fips'].astype(str).str.zfill(5)

# Merge the data
print("Merging shapefile with land cover data...")
counties_with_data = counties.merge(landcover_df, left_on='GEOID', right_on='county_fips', how='left')

# Filter to continental US for better visualization (exclude Alaska, Hawaii, territories)
# Alaska (02), Hawaii (15), Puerto Rico (72), Virgin Islands (78), other territories
continental_states = counties_with_data[~counties_with_data['STATEFP'].isin(['02', '15', '72', '78', '60', '66', '69'])]

# Create figure with subplots for each land cover type
print("Creating land cover proportion maps...")
fig = plt.figure(figsize=(24, 20))
gs = GridSpec(3, 2, figure=fig, hspace=0.15, wspace=0.05)

# Define color schemes for each land cover type
color_schemes = {
    'forest_proportion': 'Greens',
    'agriculture_proportion': 'YlOrBr', 
    'developed_proportion': 'Greys',
    'wetland_proportion': 'Blues',
    'other_proportion': 'Oranges'
}

titles = {
    'forest_proportion': 'Forest Coverage',
    'agriculture_proportion': 'Agricultural Coverage',
    'developed_proportion': 'Developed Coverage',
    'wetland_proportion': 'Wetland Coverage',
    'other_proportion': 'Other Land Coverage (Water, Barren, Grassland, Shrub)'
}

# Create individual maps for each land cover type
for idx, (col, title) in enumerate(titles.items()):
    row = idx // 2
    column = idx % 2
    ax = fig.add_subplot(gs[row, column])
    
    # Plot the map
    continental_states.plot(column=col,
                           ax=ax,
                           legend=True,
                           cmap=color_schemes[col],
                           edgecolor='none',
                           linewidth=0,
                           missing_kwds={'color': 'lightgray'},
                           legend_kwds={'label': 'Proportion',
                                      'orientation': 'horizontal',
                                      'shrink': 0.8,
                                      'pad': 0.02,
                                      'fraction': 0.05})
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')
    
    # Add state boundaries for reference
    state_boundaries = continental_states.dissolve(by='STATEFP')
    state_boundaries.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5, alpha=0.3)

# Create dominant land cover map
print("Creating dominant land cover type map...")
ax_dominant = fig.add_subplot(gs[2, :])

# Determine dominant land cover for each county
land_cover_cols = ['forest_proportion', 'agriculture_proportion', 'developed_proportion', 
                   'wetland_proportion', 'other_proportion']

# Find dominant type
continental_states['dominant_type'] = continental_states[land_cover_cols].idxmax(axis=1)
continental_states['dominant_type'] = continental_states['dominant_type'].str.replace('_proportion', '')

# Handle missing data
continental_states.loc[continental_states[land_cover_cols].sum(axis=1) == 0, 'dominant_type'] = 'no_data'

# Create categorical color map
categories = ['forest', 'agriculture', 'developed', 'wetland', 'other', 'no_data']
colors_dominant = ['#2E7D32', '#F57C00', '#616161', '#1976D2', '#E65100', '#E0E0E0']
cmap_dominant = ListedColormap(colors_dominant)

# Map categories to numbers
continental_states['dominant_code'] = continental_states['dominant_type'].map({
    'forest': 0, 'agriculture': 1, 'developed': 2, 'wetland': 3, 'other': 4, 'no_data': 5
})

# Plot dominant land cover
continental_states.plot(column='dominant_code',
                        ax=ax_dominant,
                        cmap=cmap_dominant,
                        edgecolor='none',
                        linewidth=0,
                        vmin=0,
                        vmax=5)

# Add state boundaries
state_boundaries = continental_states.dissolve(by='STATEFP')
state_boundaries.boundary.plot(ax=ax_dominant, edgecolor='black', linewidth=0.5, alpha=0.3)

ax_dominant.set_title('Dominant Land Cover Type by County', fontsize=16, fontweight='bold', pad=10)
ax_dominant.axis('off')

# Create custom legend
legend_elements = [
    mpatches.Patch(color='#2E7D32', label='Forest'),
    mpatches.Patch(color='#F57C00', label='Agriculture'),
    mpatches.Patch(color='#616161', label='Developed'),
    mpatches.Patch(color='#1976D2', label='Wetland'),
    mpatches.Patch(color='#E65100', label='Other'),
    mpatches.Patch(color='#E0E0E0', label='No Data')
]
ax_dominant.legend(handles=legend_elements, loc='lower left', frameon=True, 
                  fancybox=True, shadow=True, ncol=6, fontsize=10)

# Main title
fig.suptitle('US County Land Cover Proportions - NLCD 2024 Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

# Save the figure
plt.tight_layout()
plt.savefig('county_landcover_maps.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Map saved as: county_landcover_maps.png")

# Create additional focused regional maps
print("\nCreating regional detail maps...")
fig2, axes = plt.subplots(2, 2, figsize=(20, 16))

# Define regions of interest
regions = {
    'Northeast Corridor': {'states': ['09', '25', '44', '36', '34', '10', '24', '11', '51', '42'],
                          'title': 'Northeast Corridor - Development Pressure'},
    'Corn Belt': {'states': ['19', '17', '18', '39', '27', '55'],
                  'title': 'Midwest Corn Belt - Agricultural Dominance'},
    'Southeast Forest': {'states': ['37', '45', '13', '01', '47', '21', '54', '51'],
                        'title': 'Appalachian/Southeast - Forest Coverage'},
    'Western Arid': {'states': ['04', '35', '32', '49', '08', '56'],
                    'title': 'Western States - Arid Lands and Development'}
}

for idx, (region_name, region_info) in enumerate(regions.items()):
    ax = axes[idx // 2, idx % 2]
    
    # Filter to region
    regional_data = continental_states[continental_states['STATEFP'].isin(region_info['states'])]
    
    # Plot dominant land cover
    regional_data.plot(column='dominant_code',
                      ax=ax,
                      cmap=cmap_dominant,
                      edgecolor='gray',
                      linewidth=0.1,
                      vmin=0,
                      vmax=5)
    
    # Add state boundaries
    regional_states = regional_data.dissolve(by='STATEFP')
    regional_states.boundary.plot(ax=ax, edgecolor='black', linewidth=1, alpha=0.5)
    
    ax.set_title(region_info['title'], fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Add mini legend
    if idx == 0:
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
                 fancybox=True, shadow=True, fontsize=8)

fig2.suptitle('Regional Land Cover Patterns - Detailed Views', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('regional_landcover_maps.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Regional maps saved as: regional_landcover_maps.png")

print("\nMap creation complete!")