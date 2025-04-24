import torch
import torch.utils.data
import pandas as pd  
import os
import numpy as np
from numpy import genfromtxt
import datetime
import h5py


BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8','B9','B10', 'B11', 'B12',
       'B8A']
classes = [33101011, 33101012, 33101021, 33101022, 33101032, 33101041, 33101042,
 33101051, 33101052, 33101060, 33101071, 33101072, 33101080, 33101100,
 33102020, 33103000, 33104000, 33105000, 33106020, 33106042, 33106050,
 33106060, 33106080, 33106100, 33106120, 33106130, 33107000, 33109000,
 33110000, 33111010, 33111022, 33111023, 33112000, 33114000, 33200000,
 33301010, 33301030, 33301040, 33304000, 33305000, 33402000, 33500000,
 33600000, 33700000,]
NORMALIZING_FACTOR = 1e-4
PADDING_VALUE = -1

class EuroCropsDataset(torch.utils.data.Dataset):

    def __init__(self, root, partition, country ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        else:
            self.device = "cpu"

        self.partition = partition
        
        

        self.root = root
        if self.partition == "train":
            self.h5_file_path = os.path.join(self.root, "HDF5s", "train", country+"_train"+".h5")
        elif  self.partition == "test":    
            self.h5_file_path = os.path.join(self.root, "HDF5s", "test", country+"_test"+".h5")
        

        
        h5_file = h5py.File(self.h5_file_path)
        
        region_all = []
        for name, h5obj in h5_file.items():
            if isinstance(h5obj,h5py.Group):
                region_all.append(name)
        all_labelsfile = []
        all_data= []
        for i in range(len(region_all)):
            region = region_all[i]
            csv_file_name = 'demo_eurocrops_' + region + '.csv'
            if self.partition == "train":
                csv_file_path = os.path.join(self.root, "csv_labels", "train", csv_file_name)
        
            elif  self.partition == "test":    
                csv_file_path = os.path.join(self.root, "csv_labels", "test", csv_file_name)

            labelsfile = pd.read_csv(csv_file_path, index_col=0)
            all_labelsfile.append(labelsfile)
            data = pd.read_hdf(self.h5_file_path, region)
            all_data.append(data)

        
        self.labelsfile = pd.concat(all_labelsfile)
        
        self.mapping = self.labelsfile.set_index("crpgrpc")
        self.classes = self.labelsfile["crpgrpc"].unique()
        self.crpgrpn = self.labelsfile.groupby("crpgrpc").first().crpgrpn.values
        self.nclasses = len(self.classes)

        
        self.data = pd.concat(all_data)
        
        
        ids = list(self.data.index)
        self.ids = ids    
        print('{} parcels in file with {} classes '.format(len(ids),self.nclasses))
       
        

    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        

        spectral_data = self.data.loc[self.ids[idx]].sort_index()
        
        self.id = self.ids[idx]

        
        crop_no = self.labelsfile.loc[self.id]['crpgrpc']
        label = self.labelsfile.loc[self.id]['crpgrpn']
        
        y_label = classes.index(int(crop_no))

        #length = max(map(len,spectral_data))
        length = 13
        
        spectral_data_array = np.empty((0, length))

        for ii in range(spectral_data.shape[0]):
            if np.isnan(spectral_data[ii]).all()==True:
                test = np.zeros(length)
                test.shape = (-1, len(test))
                spectral_data_array = np.concatenate((spectral_data_array,test))
            else:
                test = np.array(spectral_data[ii])* NORMALIZING_FACTOR
                test.shape = (-1, len(test))
                spectral_data_array = np.concatenate((spectral_data_array,test))
               


        

        X = torch.tensor(spectral_data_array).type(torch.FloatTensor).to(self.device)
        # y= torch.from_numpy(np.array(crop_no)).type(torch.LongTensor).to(self.device)
        
        dates_json = spectral_data.index
        max_len = len(spectral_data)
        # Instead of taking the position, the numbers of days since the first observation is used
        days = torch.zeros(max_len)
        date_0 = dates_json[0]
        date_0 = datetime.datetime.strptime(str(date_0), "%Y%m%d")
        days[0] = 0
        for i in range(max_len - 1):
            date = dates_json[i + 1]
            date = datetime.datetime.strptime(str(date), "%Y%m%d")
            days[i + 1] = (date - date_0).days
        days = days.unsqueeze(1)
        
        return {'data':X, 'label':y_label, 'ids':self.id, 'crop name':label, 'dates':days}

            

