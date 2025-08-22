#!/usr/bin/env python3
"""
Create interactive HTML map of county land cover proportions using Folium.
"""

import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import json
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

# Filter to continental US for better visualization
continental_states = counties_with_data[~counties_with_data['STATEFP'].isin(['02', '15', '72', '78', '60', '66', '69'])]

# Simplify geometry for web performance
print("Simplifying geometries for web performance...")
continental_states['geometry'] = continental_states['geometry'].simplify(0.01, preserve_topology=True)

# Convert to WGS84 for web mapping
continental_states = continental_states.to_crs('EPSG:4326')

# Calculate dominant land cover type
land_cover_cols = ['forest_proportion', 'agriculture_proportion', 'developed_proportion', 
                   'wetland_proportion', 'other_proportion']

continental_states['dominant_type'] = continental_states[land_cover_cols].idxmax(axis=1)
continental_states['dominant_type'] = continental_states['dominant_type'].str.replace('_proportion', '').str.capitalize()

# Handle missing data
continental_states.loc[continental_states[land_cover_cols].sum(axis=1) == 0, 'dominant_type'] = 'No Data'

# Create the base map
print("Creating interactive map...")
m = folium.Map(location=[39.5, -98.35], zoom_start=5, tiles='OpenStreetMap')

# Define color schemes for dominant land cover
color_map = {
    'Forest': '#2E7D32',
    'Agriculture': '#F57C00',
    'Developed': '#616161',
    'Wetland': '#1976D2',
    'Other': '#E65100',
    'No Data': '#E0E0E0'
}

# Create style function for the choropleth
def style_function(feature):
    dominant = feature['properties'].get('dominant_type', 'No Data')
    return {
        'fillColor': color_map.get(dominant, '#E0E0E0'),
        'color': 'black',
        'weight': 0.1,
        'fillOpacity': 0.7
    }

# Create highlight function
def highlight_function(feature):
    return {
        'weight': 2,
        'color': 'black',
        'fillOpacity': 0.9
    }

# Prepare tooltip text
continental_states['tooltip_text'] = continental_states.apply(
    lambda x: f"""
    <b>{x['NAME']}, {x['STATEFP']}</b><br>
    <b>Dominant: {x['dominant_type']}</b><br>
    <hr>
    Forest: {x['forest_proportion']*100:.1f}%<br>
    Agriculture: {x['agriculture_proportion']*100:.1f}%<br>
    Developed: {x['developed_proportion']*100:.1f}%<br>
    Wetland: {x['wetland_proportion']*100:.1f}%<br>
    Other: {x['other_proportion']*100:.1f}%
    """ if pd.notna(x['forest_proportion']) else f"<b>{x['NAME']}, {x['STATEFP']}</b><br>No Data Available",
    axis=1
)

# Convert to GeoJSON with properties
geojson_data = json.loads(continental_states[['geometry', 'dominant_type', 'NAME', 'STATEFP', 
                                              'forest_proportion', 'agriculture_proportion',
                                              'developed_proportion', 'wetland_proportion',
                                              'other_proportion', 'tooltip_text']].to_json())

# Add the choropleth layer
folium.GeoJson(
    geojson_data,
    style_function=style_function,
    highlight_function=highlight_function,
    tooltip=folium.GeoJsonTooltip(
        fields=['tooltip_text'],
        labels=False,
        sticky=True,
        style="background-color: white; border: 2px solid black; border-radius: 3px; box-shadow: 3px;",
        max_width=300
    ),
    name='County Land Cover'
).add_to(m)

# Add a custom legend
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; right: 50px; width: 200px; height: auto; 
            background-color: white; z-index:9999; font-size:14px;
            border:2px solid grey; border-radius: 5px; padding: 10px">
<h4 style="margin-top: 0;">Dominant Land Cover</h4>
<p style="margin: 0;"><span style="background-color: #2E7D32; width: 20px; height: 10px; display: inline-block; margin-right: 5px;"></span>Forest</p>
<p style="margin: 0;"><span style="background-color: #F57C00; width: 20px; height: 10px; display: inline-block; margin-right: 5px;"></span>Agriculture</p>
<p style="margin: 0;"><span style="background-color: #616161; width: 20px; height: 10px; display: inline-block; margin-right: 5px;"></span>Developed</p>
<p style="margin: 0;"><span style="background-color: #1976D2; width: 20px; height: 10px; display: inline-block; margin-right: 5px;"></span>Wetland</p>
<p style="margin: 0;"><span style="background-color: #E65100; width: 20px; height: 10px; display: inline-block; margin-right: 5px;"></span>Other</p>
<p style="margin: 0;"><span style="background-color: #E0E0E0; width: 20px; height: 10px; display: inline-block; margin-right: 5px;"></span>No Data</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Add layer control
folium.LayerControl().add_to(m)

# Add fullscreen button
plugins.Fullscreen().add_to(m)

# Save the map
print("Saving interactive map...")
m.save('county_landcover_interactive.html')
print("Interactive map saved as: county_landcover_interactive.html")

# Create a second map showing forest proportion as a continuous choropleth
print("\nCreating forest proportion choropleth map...")
m2 = folium.Map(location=[39.5, -98.35], zoom_start=5, tiles='CartoDB positron')

# Create choropleth for forest proportion
folium.Choropleth(
    geo_data=continental_states[['geometry', 'GEOID', 'forest_proportion']].to_json(),
    name='Forest Coverage',
    data=continental_states,
    columns=['GEOID', 'forest_proportion'],
    key_on='feature.properties.GEOID',
    fill_color='Greens',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Forest Coverage (%)',
    nan_fill_color='lightgray'
).add_to(m2)

# Add tooltips
for idx, row in continental_states.iterrows():
    if pd.notna(row['forest_proportion']):
        tooltip_text = f"{row['NAME']}, {row['STATEFP']}<br>Forest: {row['forest_proportion']*100:.1f}%"
    else:
        tooltip_text = f"{row['NAME']}, {row['STATEFP']}<br>No Data"
    
    folium.GeoJson(
        row['geometry'].__geo_interface__,
        style_function=lambda x: {'fillOpacity': 0, 'color': 'transparent', 'weight': 0},
        tooltip=folium.Tooltip(tooltip_text, sticky=True)
    ).add_to(m2)

m2.save('forest_coverage_interactive.html')
print("Forest coverage map saved as: forest_coverage_interactive.html")

# Create summary statistics for the interactive map
print("\nGenerating map statistics...")
stats = {
    'Total Counties': len(continental_states),
    'Counties with Data': len(continental_states[continental_states[land_cover_cols].sum(axis=1) > 0]),
    'Dominant Forest': len(continental_states[continental_states['dominant_type'] == 'Forest']),
    'Dominant Agriculture': len(continental_states[continental_states['dominant_type'] == 'Agriculture']),
    'Dominant Developed': len(continental_states[continental_states['dominant_type'] == 'Developed']),
    'Dominant Wetland': len(continental_states[continental_states['dominant_type'] == 'Wetland']),
    'Dominant Other': len(continental_states[continental_states['dominant_type'] == 'Other'])
}

print("\nMap Statistics (Continental US):")
for key, value in stats.items():
    print(f"  {key}: {value:,}")

print("\nInteractive maps created successfully!")
print("Open 'county_landcover_interactive.html' in a web browser to explore the data.")
print("Open 'forest_coverage_interactive.html' for forest-specific visualization.")