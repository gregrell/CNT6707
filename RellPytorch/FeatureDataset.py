import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, filename):
        print(filename)
        i=1

    def __len__(self):
        return 0

    def __getitem__(self, idx):

        return 0,0