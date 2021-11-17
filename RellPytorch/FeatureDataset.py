import os
import pandas as pd
import torch
#from torchvision.io import read_image
from torch.utils.data import Dataset

class RobotArmDataset(Dataset):
    def __init__(self, filename):
        #read in CSV file
        file_out = pd.read_csv(filename)
        x = file_out.iloc[1:211742, 0:4].values
        y = file_out.iloc[1:211742, 4].values

        #convert to tensors
        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)


    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]