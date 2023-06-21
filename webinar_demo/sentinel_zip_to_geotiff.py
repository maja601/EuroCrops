import numpy as np
import os
import subprocess

import rasterio

from pathlib import Path
from rasterio.plot import reshape_as_image
from tqdm import tqdm

os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'

def generate_geotiffs_from_safe_format(safe_path, vrt_path, geotiff_path, bands_to_use):
    """
    Generates a geotiff from a specified number of bands
    :param safe_path:       (string) Path where the SAFE files are stored
    :param vrt_path:        (string) Path where to store the intermediate VRT files
    :param geotiff_path:    (string) Path where the resulting GeoTIFFs should end up
    :param bands_to_use:    (list)   List of the bands that should go into the GeoTIFF
    """
    for f in tqdm(os.scandir(safe_path)):
        # Prepare input path
        input_path = f.path
        tile_name = f.name[38:44]
        datatake_sensing_time = f.name[11:19]
        

        # Build path to the images
        image_path = os.path.join(input_path, 'GRANULE')
        image_path = [f.path for f in os.scandir(image_path)][0]
        basename_granule = os.path.basename(image_path)


        image_path = os.path.join(image_path, 'IMG_DATA')

        # Outputs
        output_vrt_name = tile_name + '_' + datatake_sensing_time + '.vrt'
        output_tiff_name = tile_name + '_' + datatake_sensing_time + '.tif'
        output_full_path_vrt = os.path.join(vrt_path, output_vrt_name)
        output_full_path_tiff = os.path.join(geotiff_path, output_tiff_name)

        # Construct the bands command
        bands_path = [f.path for f in os.scandir(image_path)]
        bands = {}
        for band in bands_to_use:
            bands_key = 'band_' + band
            band_val_ending = 'B' + band + '.jp2'
            band_val = [b for b in bands_path if band_val_ending in b][0]
            bands[bands_key] = band_val

        # Command to build the VRT File
        # from https://github.com/dairejpwalsh/Sentinel-Scripts/blob/master/Sentinel%202/tiff-generator.py
        cmd = ['gdalbuildvrt', '-resolution', 'user', '-tr', '10', '10', '-separate', output_full_path_vrt]
        for band in sorted(bands.values()):     # 8A is last band now
            cmd.append(band)
        my_file = Path(output_full_path_vrt)
        if not my_file.is_file():
            # file exists
            subprocess.call(cmd)

        # Command to build the GEOTIFF
        cmd = ['gdal_translate', '-of', 'GTiff', output_full_path_vrt, output_full_path_tiff]
        my_file = Path(output_full_path_tiff)
        if not my_file.is_file():
            # file exists
            subprocess.call(cmd)


def check_final_result(safe_path, geotiff_path):
    """
    For debugging: Check the resulting GeoTIFF and the JPEG2000 files that created it
    :param safe_path:       (string) Path where the unzipped SAFE files are stored
    :param geotiff_path:    (string) Path the the GeoTIFF is stored
    """
    for f in tqdm(os.scandir(safe_path)):
        # Prepare input path
        input_path = f.path
        image_path = os.path.join(input_path, 'GRANULE')
        image_path = [f.path for f in os.scandir(image_path)][0]
        image_path = os.path.join(image_path, 'IMG_DATA')
        bands_path = [f.path for f in os.scandir(image_path)]

        # Read the JP2000 images
        imgs = {}
        for band in bands_path:
            with rasterio.open(band, driver='JP2OpenJPEG') as src:
                imgs[band[-7:-4]] = np.squeeze(src.read())

        # Read GeoTIFF
        gtif_path = os.path.join(geotiff_path, 'T32VNH_20200909.tif')
        with rasterio.open(gtif_path) as src:
            gtif = src.read()
            gtif = reshape_as_image(gtif)


def main():
    """
    Extract GeoTIFFs from either SENTINEL-2 zip files or already extraced SAFE folder with GDAL.
    Currently just for L1C, but can also be applicable for L2A data.

    Specifications:
    - sentinel_tiles_path (string): Directory with the folders for each SENTINEL tile
    - tile_names (list):            A list of tile names / directory names
    - sentinel_bands (list):        A list of the bands that should be used for the GeoTIFF
    - processing_level (string):    Used processing level (rn only working for L1C)
    """
    # For debugging the check
    testing = False

    # Specifications
    sentinel_tiles_path = '/home/eouser/'
    tile_names = ['T31UFS']
    sentinel_bands = ['01', '02', '03', '04', '05', '06', '07', '08', '8A', '09', '10', '11', '12']
    processing_level = 'L1C'

    # For all available tile names
    for tile_name in tile_names:
        # Prepare paths
        # zipfile_path = os.path.join(sentinel_tiles_path, tile_name, processing_level)
        safe_path = os.path.join(sentinel_tiles_path, tile_name, 'L1C_SAFE/')
        vrt_path = os.path.join(sentinel_tiles_path, tile_name, 'L1C_VRT/')
        geotiff_path = os.path.join(sentinel_tiles_path, tile_name, 'L1C_GeoTIFFs/')
        if not testing:
            # Go
            generate_geotiffs_from_safe_format(safe_path, vrt_path, geotiff_path, sentinel_bands)
        else:
            check_final_result(safe_path, geotiff_path)


if __name__ == "__main__":
    main()

