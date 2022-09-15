import pandas as pd
from torch.utils .data import Dataset
import numpy as np


def load_file(filepath):
    col = ["unit_num", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"senor_{i}" for i in range(1,22)]
    df = pd.read_csv(filepath, sep="\s+",header=None,names=col)
    df = df.drop(columns=["setting_1", "setting_2", "setting_3"]+[f"senor_{i}" for i in (1,5,6,10,16,18,19)])
    return df

def load_RUL(filepath):
    return pd.read_csv(filepath,sep='\s+', header=None, names=['RUL'])

class TrainLoader(Dataset):
    def __init__(self,filepath) -> None:
        self.data = load_file(filepath)
        self.filter_()
        
    def norm(self,data, title):
        data_norm = (data - data.min()) / (data.max() - data.min())
        train_norm = pd.concat([title, data_norm], axis=1)
        
        grouped_by_unit = train_norm.groupby(by="unit_num")
        max_cycle = grouped_by_unit["time_cycles"].max()

        result_frame = train_norm.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_num', right_index=True)

        remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
        result_frame["RUL"] = remaining_useful_life

        result_frame = result_frame.drop(columns="max_cycle", axis=1)
        
        train_norm = result_frame
        train_norm['RUL'].clip(upper=125, inplace=True)
        
        group = train_norm.groupby(by="unit_num")
        return group
        
    def filter_(self):
        title = self.data.iloc[:, 0:2]
        data = self.data.iloc[:, 2:]
        
        self.group = self.norm(data, title)
        
    def __len__(self):
        return len(self.group)
    
    def __getitem__(self, index):
        return self.group.get_group(index+1).to_numpy(dtype=np.float32)
    
    
class TestLoader(Dataset):
    def __init__(self, filepath) -> None:
        self.test = load_file(filepath)
        self.test_norm = self.norm()

    def norm(self):
        title = self.test.iloc[:,0:2]
        data = self.test.iloc[:,2:]
        data_norm =  (data - data.min()) / (data.max() - data.min())
        test_norm = pd.concat([title, data_norm], axis=1)
        return test_norm.groupby(by="unit_num")

    def get_data(self,idx):
        x_test = self.test_norm.get_group(idx+1).to_numpy(dtype=np.float32)
        return x_test

    
    def __len__(self) -> int:
        return len(self.test_norm)

    def __getitem__(self, index):
        return self.get_data(index)

if __name__ == "__main__":
    tl = TestLoader()
    for t in tl:
        print(t)