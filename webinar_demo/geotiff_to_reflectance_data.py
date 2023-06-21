import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import warnings
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from shapely.geometry import mapping
from tqdm import tqdm


def extract_reflectance_data_into_hd5(geotiff_path, shp_path, output_name, mode='Median'):
    """
    Collect the average reflectance data of a patch for each sentinel band and store it into a HDF5 file.
    :param geotiff_path:    (string) The folder where all the GeoTIFFs are stored
    :param shp_path:    (string) The folder where the shp files, each containing the reference data for a
                                        region, is stored
    :param mode:            (string) How the average of the parcel reflectance is calulated (for now: Just MEDIAN)
    """
    # Create a df with the timesteps
    timesteps = [f.name[7:-4] for f in os.scandir(geotiff_path) if f.name.endswith(".tif")]
    

    # Iterate over all regions of a country
    for shp_idx, shp_f in enumerate(os.scandir(shp_path)):
        if shp_f.name.endswith(".shp"):
            shp_file = shp_f.path
            region_name = shp_f.name[0:2]
            print('Region: {}'.format(region_name))

            reflectances_df = pd.DataFrame(columns=timesteps)

            # Iterate over all available Sentinel images of this corresponding region
            for f in tqdm(os.scandir(geotiff_path)):
                if f.name.endswith(".tif"):
                    input_path = f.path
                    current_timestep = f.name[7:-4]
                    with rasterio.open(input_path) as src:
                        # Just to have a sneak at the image
                        # raster_img = src.read()
                        # raster_img = reshape_as_image(raster_img)
                        raster_meta = src.meta
                    
                        

                        # Open shp file and reproject it into the geotiff CRS
                        demo_ec_df = gpd.read_file(shp_file)
                        
                        demo_ec_df = demo_ec_df[demo_ec_df.geometry.notnull()]  # remove entries without geometry
                        # demo_ec_df.crs = {'init': 'epsg:4326'}  # Current CRS
                        demo_ec_df = demo_ec_df.to_crs(raster_meta['crs']['init'])  # CRS of the images

                        # Iterate over all shp df entries
                        failed = []
                        for num, row in demo_ec_df.iterrows():


                            if num % 10000 == 0:
                                print(num)
                            try:
                                masked_img, out_transform = rasterio.mask.mask(src, [mapping(row['geometry'])], crop=True,
                                                                            nodata=0)
                                masked_img = reshape_as_image(masked_img)
                                # Calculate the Median of each patch for each channel
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=RuntimeWarning)
                                    patch_median = np.asarray(
                                        [np.nanmedian(masked_img[:, :, a][masked_img[:, :, a] != 0]).astype(np.int16) for a in
                                        range(masked_img.shape[-1])])
                                    patch_median[patch_median < 0] = 0
                                

                                
                                
                                




                                # Add a list of all 13 bands into a cell which can be accessed by [Field_ID, Timestep]
                                if not (reflectances_df.index == row['fid']).any():
                                    temp_df = pd.DataFrame(columns=timesteps, index=[row['fid']])
                                    #reflectances_df = reflectances_df.append(temp_df)
                                    reflectances_df = pd.concat([reflectances_df, temp_df],axis=0, join='outer')
                                  
                                reflectances_df.at[row['fid'], current_timestep] = patch_median.tolist()
                            except Exception as e:
                                failed.append(num)  # Save all failed parcels
                    print("Rasterio failed to mask approximately {} files".format(len(failed)))

            # Store the reflectances into a hdf5 file, where they can be accessed by the region name

            reflectances_df.to_hdf(output_name,region_name  )
        


def main():
    """
    Run a method that stores the average reflectance data

    """
    geotiff_dir = '/home/eouser/T31UFS/L1C_GeoTIFFs/'
    shp_dir = '/home/eouser/VLG/'
    output_name = '/home/eouser/output/T31UFS.hdf5'
    extract_reflectance_data_into_hd5(geotiff_dir, shp_dir, output_name)


if __name__ == "__main__":
    main()
