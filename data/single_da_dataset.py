from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import csv
import time
import logging
import os
# only load one target domain data, this is designed for Table one experiment in our paper. you can ignore this part.
domain_index={
    'a': 1,
    'b': 3,
    'c': 2,
    's1': 4,
    's2': 6,
    's3': 5,
    's4': 7,
    's5': 8,
    's6': 9
}
labels = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 
    'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park', 'unknown']
    
# classes_num = len(labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
class ASC2020_single(Dataset):
    def __init__(self,h5_path,tg_name):
        self.X = None
        self.y = None
        self.index = None
        with h5py.File(h5_path, 'r') as hf:
            X = hf['feature'][:].astype(np.float32)
            
            if 'target' in hf.keys():
                y = np.array(
                    [scene_label \
                        for scene_label in hf['target'][:]])
                
            if 'source_label' in hf.keys():
                index = np.array(
                    [source_label.decode() \
                        for source_label in hf['source_label'][:]])
        print(X.shape,y.shape,index.shape)
        X1=[]
        y1=[]
        index1=[]
        for i,label in enumerate(index):
            if label=='a' or label==tg_name:
                X1.append(X[i])
                y1.append(y[i])
                index1.append(index[i])
        self.X = np.array(X1)
        self.y = np.array(y1)
        self.index = np.array(index1)
        #print(self.X.shape,self.y.shape,self.index.shape)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], int(self.targets[index])
        X,y_sparse,tmp = self.X[index],self.y[index],self.index[index]
        t = []
        t.append(domain_index[tmp])
        t = np.array(t)
        # print('t   ..',t.shape)
        domain = t
        #y = sparse_to_categorical(y_sparse,10)
        return X,y_sparse,t,domain

    def __len__(self):
        return len(self.X)

    def get(self):
        return self.X,self.y,self.index
