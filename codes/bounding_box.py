""""Code to get the bounding box of the Lake Erie Basin shapefile
"""

import geopandas as gpd
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()
directory = os.getenv("path")

# Load shapefile
shapefile_path = os.path.join(directory, 'leb_subbasins', 'LakeErieBasinBoundaries.shp')
shapefile_gdf = gpd.read_file(shapefile_path)

bounding_box = shapefile_gdf.total_bounds
print(f'box coordinates: {bounding_box}')