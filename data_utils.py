import numpy as np
import torch
from torch.utils.data import Dataset



class LPDataset(Dataset):  # Dataset是数据集位置，就是下面的path

    def __init__(self, data, input_n, output_n):
        super(LPDataset, self).__init__()
        self.data = data
        # self.data = torch.from_numpy(self.data)
        # print(type(self.data))
        self.data = self.data.reshape((-1, self.data.shape[1], 8))
        self.input_n = input_n
        self.output_n = output_n
        self.num = self.data.shape[0] - (input_n + output_n + input_n)

    def __len__(self):
        # print(self.num)
        return self.num

    def __getitem__(self, item):
        a = item
        b = item + self.input_n
        c = item + self.input_n + self.output_n
        d = item + self.input_n + self.output_n + self.input_n
        # print('a',a,'b',b,'c',c,d)
        # print(type(self.data))
        return self.data[a: b], self.data[b: c], self.data[d - 1:c - 1:-1].copy()









