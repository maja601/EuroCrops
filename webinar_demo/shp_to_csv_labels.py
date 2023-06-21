import csv
import os


os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

from tqdm import tqdm


def extract_labels_single(shp_dir, csv_dir):
    for shp_idx, shp_f in enumerate(os.scandir(shp_dir)):
        if shp_f.name.endswith(".shp"):
            region_name = shp_f.name[15:-4]
            shp_file = shp_f.path
            demo_ec_df = gpd.read_file(shp_file)
            demo_ec_df = demo_ec_df[demo_ec_df.geometry.notnull()]  # remove entries without geometry

            csv_file_name = csv_dir + shp_f.name[:-4] + '.csv'
            with open(csv_file_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([['recno', 'hcat_c', 'hcat_n']])
                for num, row in tqdm(demo_ec_df.iterrows()):
                    line = [row['fid'],  row['EC_hcat_c'],row['EC_hcat_n']]
                    writer.writerows([line])


def main():
    shp_dir = '/home/eouser/VLG/'
    csv_dir = '/home/eouser/output/'

    extract_labels_single(shp_dir, csv_dir)


if __name__ == "__main__":
    main()
