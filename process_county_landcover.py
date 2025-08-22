#!/usr/bin/env python3
"""
Process NLCD 2024 raster data to calculate land cover proportions by county.

This script performs the following operations:
1. Load NLCD raster and county shapefile
2. Reproject county boundaries to match raster CRS
3. For each county, extract raster values and calculate land cover proportions
4. Export results to CSV file

Dependencies: geopandas, rasterio, rasterstats, pandas, numpy, tqdm
"""

import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
from rasterstats import zonal_stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# File paths
NLCD_RASTER_PATH = '/home/mihiarc/repos/nlcd-county/Annual_NLCD_LndCov_2024_CU_C1V1/Annual_NLCD_LndCov_2024_CU_C1V1.tif'
COUNTY_SHAPEFILE_PATH = '/home/mihiarc/repos/nlcd-county/tl_2024_us_county/tl_2024_us_county.shp'
OUTPUT_CSV_PATH = '/home/mihiarc/repos/nlcd-county/county_landcover_proportions.csv'

# NLCD land cover reclassification mapping
NLCD_RECLASSIFICATION = {
    # Forest
    41: 'forest',    # Deciduous Forest
    42: 'forest',    # Evergreen Forest
    43: 'forest',    # Mixed Forest
    
    # Agriculture
    81: 'agriculture',    # Pasture/Hay
    82: 'agriculture',    # Cultivated Crops
    
    # Developed
    21: 'developed',    # Developed, Open Space
    22: 'developed',    # Developed, Low Intensity
    23: 'developed',    # Developed, Medium Intensity
    24: 'developed',    # Developed, High Intensity
    
    # Wetland
    90: 'wetland',    # Woody Wetlands
    95: 'wetland',    # Emergent Herbaceous Wetlands
    
    # Other
    11: 'other',     # Open Water
    12: 'other',     # Perennial Ice/Snow
    31: 'other',     # Barren Land
    52: 'other',     # Shrub/Scrub
    71: 'other',     # Grassland/Herbaceous
    
    # NoData (exclude from calculations)
    250: 'nodata'
}

def reclassify_array(array, reclassification_map):
    """
    Reclassify a numpy array based on a mapping dictionary.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Input array with original NLCD values
    reclassification_map : dict
        Mapping from original values to new class names
    
    Returns:
    --------
    dict : Dictionary with class names as keys and pixel counts as values
    """
    unique_values, counts = np.unique(array[array != 0], return_counts=True)  # Exclude masked/nodata pixels
    
    class_counts = {
        'forest': 0,
        'agriculture': 0,
        'developed': 0,
        'wetland': 0,
        'other': 0,
        'nodata': 0
    }
    
    for value, count in zip(unique_values, counts):
        if value in reclassification_map:
            class_name = reclassification_map[value]
            class_counts[class_name] += count
    
    return class_counts

def calculate_proportions(class_counts):
    """
    Calculate proportions for each land cover class, excluding NoData pixels.
    
    Parameters:
    -----------
    class_counts : dict
        Dictionary with class names as keys and pixel counts as values
    
    Returns:
    --------
    dict : Dictionary with class names as keys and proportions as values
    """
    # Calculate total pixels excluding NoData
    total_valid_pixels = sum(count for class_name, count in class_counts.items() 
                           if class_name != 'nodata')
    
    if total_valid_pixels == 0:
        # Handle case where all pixels are NoData
        return {
            'forest_proportion': 0.0,
            'agriculture_proportion': 0.0,
            'developed_proportion': 0.0,
            'wetland_proportion': 0.0,
            'other_proportion': 0.0
        }
    
    proportions = {
        'forest_proportion': class_counts['forest'] / total_valid_pixels,
        'agriculture_proportion': class_counts['agriculture'] / total_valid_pixels,
        'developed_proportion': class_counts['developed'] / total_valid_pixels,
        'wetland_proportion': class_counts['wetland'] / total_valid_pixels,
        'other_proportion': class_counts['other'] / total_valid_pixels
    }
    
    return proportions

def process_county_landcover():
    """
    Main function to process NLCD data and calculate county-level land cover proportions.
    """
    print("Loading datasets...")
    
    # Load county shapefile
    print("Loading county shapefile...")
    counties = gpd.read_file(COUNTY_SHAPEFILE_PATH)
    print(f"Loaded {len(counties)} counties")
    
    # Load NLCD raster to get CRS information
    print("Loading NLCD raster...")
    with rasterio.open(NLCD_RASTER_PATH) as src:
        raster_crs = src.crs
        print(f"Raster CRS: {raster_crs}")
        print(f"Raster shape: {src.shape}")
        print(f"Raster bounds: {src.bounds}")
    
    # Reproject counties to match raster CRS
    print(f"Reprojecting counties from {counties.crs} to {raster_crs}")
    counties_reprojected = counties.to_crs(raster_crs)
    
    # Initialize results list
    results = []
    
    print("Processing counties...")
    
    # Process each county
    for idx, county in tqdm(counties_reprojected.iterrows(), total=len(counties_reprojected), 
                           desc="Processing counties"):
        
        county_fips = county['GEOID']
        
        try:
            # Use rasterstats to extract raster values within county boundary
            # Using categorical=True to get counts of each unique value
            stats = zonal_stats(
                county.geometry,
                NLCD_RASTER_PATH,
                categorical=True,
                nodata=0  # Treat 0 as nodata for processing
            )
            
            if not stats or not stats[0]:
                # Handle case where no raster data intersects with county
                print(f"Warning: No raster data found for county {county_fips}")
                result = {
                    'county_fips': county_fips,
                    'forest_proportion': 0.0,
                    'agriculture_proportion': 0.0,
                    'developed_proportion': 0.0,
                    'wetland_proportion': 0.0,
                    'other_proportion': 0.0
                }
            else:
                # Get pixel counts for each NLCD class
                pixel_counts = stats[0]
                
                # Convert to class counts using reclassification mapping
                class_counts = {
                    'forest': 0,
                    'agriculture': 0,
                    'developed': 0,
                    'wetland': 0,
                    'other': 0,
                    'nodata': 0
                }
                
                for nlcd_value, count in pixel_counts.items():
                    if nlcd_value in NLCD_RECLASSIFICATION:
                        class_name = NLCD_RECLASSIFICATION[nlcd_value]
                        class_counts[class_name] += count
                
                # Calculate proportions
                proportions = calculate_proportions(class_counts)
                
                result = {
                    'county_fips': county_fips,
                    **proportions
                }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing county {county_fips}: {str(e)}")
            # Add default values for failed counties
            result = {
                'county_fips': county_fips,
                'forest_proportion': 0.0,
                'agriculture_proportion': 0.0,
                'developed_proportion': 0.0,
                'wetland_proportion': 0.0,
                'other_proportion': 0.0
            }
            results.append(result)
    
    # Convert results to DataFrame
    print("Creating results DataFrame...")
    results_df = pd.DataFrame(results)
    
    # Verify proportions sum to approximately 1.0 (allowing for floating point precision)
    print("Validating results...")
    proportion_cols = ['forest_proportion', 'agriculture_proportion', 'developed_proportion', 
                      'wetland_proportion', 'other_proportion']
    results_df['total_proportion'] = results_df[proportion_cols].sum(axis=1)
    
    # Check for counties with invalid proportions
    invalid_counties = results_df[
        (results_df['total_proportion'] < 0.99) | (results_df['total_proportion'] > 1.01)
    ]
    
    if len(invalid_counties) > 0:
        print(f"Warning: {len(invalid_counties)} counties have proportions that don't sum to ~1.0")
        print("Sample invalid counties:")
        print(invalid_counties[['county_fips', 'total_proportion']].head())
    
    # Drop the validation column before saving
    results_df = results_df.drop('total_proportion', axis=1)
    
    # Save results to CSV
    print(f"Saving results to {OUTPUT_CSV_PATH}...")
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total counties processed: {len(results_df)}")
    print(f"Counties with data: {len(results_df[results_df[proportion_cols].sum(axis=1) > 0])}")
    
    print("\nLand cover statistics (mean proportions):")
    for col in proportion_cols:
        mean_prop = results_df[col].mean()
        print(f"  {col.replace('_proportion', '').title()}: {mean_prop:.4f}")
    
    print(f"\nResults saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    process_county_landcover()