from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import csv
import time
import logging
import os
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

class ASC2020(Dataset):
    def __init__(self,h5_path):
        self.X = None
        self.y = None
        self.index = None
        # self.name = None
        with h5py.File(h5_path, 'r') as hf:
            self.X = hf['feature'][:].astype(np.float32)
            
            if 'target' in hf.keys():
                self.y = np.array(
                    [scene_label \
                        for scene_label in hf['target'][:]])
                
            if 'source_label' in hf.keys():
                self.index = np.array(
                    [source_label.decode() \
                        for source_label in hf['source_label'][:]])
            if 'domain_rank' in hf.keys():
                self.index = hf['domain_rank'][:].astype(np.float32)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        X,y_sparse,t,domain = self.X[index],self.y[index],self.index[index]+1,self.index[index]+1
        return X,y_sparse,t,domain

    def __len__(self):
        return len(self.X)

class ASC2020_pre(Dataset):
    def __init__(self,h5_path):
        self.X = None
        self.y = None
        self.index = None
        with h5py.File(h5_path, 'r') as hf:
            self.X = hf['feature'][:].astype(np.float32)
            
            if 'target' in hf.keys():
                self.y = np.array(
                    [scene_label \
                        for scene_label in hf['target'][:]])
                
            if 'source_label' in hf.keys():
                self.index = np.array(
                    [source_label.decode() \
                        for source_label in hf['source_label'][:]])

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

class ASC2020_DANN(Dataset): # Design for train DANN model
    def __init__(self,h5_path):
        self.X = None
        self.y = None
        self.index = None
        with h5py.File(h5_path, 'r') as hf:

            self.X = hf['feature'][:].astype(np.float32)
            
            if 'target' in hf.keys():
                self.y = np.array(
                    [scene_label \
                        for scene_label in hf['target'][:]])
                
            if 'source_label' in hf.keys():
                self.index = np.array(
                    [source_label.decode() \
                        for source_label in hf['source_label'][:]])

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
        if domain_index[tmp]>1:
            domain_label = np.array([1,0])
        else:
            domain_label = np.array([0,1])
        return X,y_sparse,t,domain,domain_label

    def __len__(self):
        return len(self.X)

    def get(self):
        return self.X,self.y,self.index

class ASC2020_bc(Dataset): # just load device A, B,C data
    def __init__(self,h5_path):
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
            if label=='a' or label=='b' or label=='c':
                X1.append(X[i])
                y1.append(y[i])
                index1.append(index[i])

        self.X = np.array(X1)
        self.y = np.array(y1)
        self.index = np.array(index1)
        print(self.X.shape,self.y.shape,self.index.shape)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        X,y_sparse,tmp = self.X[index],self.y[index],self.index[index]
        t = []
        t.append(domain_index[tmp])
        t = np.array(t)
        # print('t   ..',t.shape)
        domain = t
        #y = sparse_to_categorical(y_sparse,10)
        if domain_index[tmp]>1:
            domain_label = np.array([1,0])
        else:
            domain_label = np.array([0,1])
        return X,y_sparse,t,domain

    def __len__(self):
        return len(self.X)

    def get(self):
        return self.X,self.y,self.index
