from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import csv
import time
import logging
import os
# domain_index={
#     'a': 5,
#     'b': 4,
#     'c': 6,
#     's1': 3,
#     's2': 7,
#     's3': 2,
#     's4': 8,
#     's5': 1,
#     's6': 9
# }
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
def sparse_to_categorical(x, n_out):
    x = x.astype(int)
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros((N,n_out))
    x_categ[np.arange(N), x] = 1
    return x_categ.reshape((shape)+(n_out,))

class MyDataset(Dataset):
    def __init__(self):
        x = list(range(0,50,1))
        self.data = x
        print('len ',len(self.data))
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class ASC2020(Dataset):
    def __init__(self,h5_path):
        # root = './data'
        # processed_folder = os.path.join(root, 'MNIST', 'processed')
        # data_file = 'training.pt'
        # self.data, self.targets = torch.load(os.path.join(processed_folder, data_file))
        self.X = None
        self.y = None
        self.index = None
        # self.name = None
        with h5py.File(h5_path, 'r') as hf:
            # self.data_dict['audio_name'] = np.array(
            #     [audio_name.decode() for audio_name in hf['audio_name'][:]])
            # self.name = np.array(
            #     [audio_name.decode() for audio_name in hf['audio_name'][:]])
            self.X = hf['feature'][:].astype(np.float32)
            
            if 'target' in hf.keys():
                self.y = np.array(
                    [scene_label \
                        for scene_label in hf['target'][:]])
                
            # if 'identifier' in hf.keys():
            #     self.data_dict['identifier'] = np.array(
            #         [identifier.decode() for identifier in hf['identifier'][:]])
                
            if 'source_label' in hf.keys():
                self.index = np.array(
                    [source_label.decode() \
                        for source_label in hf['source_label'][:]])
            if 'domain_rank' in hf.keys():
                self.index = hf['domain_rank'][:].astype(np.float32)
        # self.rotate_angle = rotate_angle
        # print('.. ',len(self.data_dict['feature']))
        # print('...  ',self.data_dict['source_label'])
        # print('len- ',self.data_dict['audio_name'])
        # for s in self.index:
        #     print(s)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # print(self.y[index],self.name[index])
        # a = lb_to_idx[self.y[index]]
        # img, target = self.data[index], int(self.targets[index])
        X,y_sparse,t,domain = self.X[index],self.y[index],self.index[index]+1,self.index[index]+1
        # print(self.index[index],domain)
        # a = lb_to_idx[self.y[index]]
        # print('X ',X.shape)
        # print('y ',y.shape)
        # print('tmp ',tmp.shape)
        # print(tmp)
        
        #y = sparse_to_categorical(y_sparse,10)
        return X,y_sparse,t,domain

    def __len__(self):
        return len(self.X)



class ASC2020_pre(Dataset):
    def __init__(self,h5_path):
        # root = './data'
        # processed_folder = os.path.join(root, 'MNIST', 'processed')
        # data_file = 'training.pt'
        # self.data, self.targets = torch.load(os.path.join(processed_folder, data_file))
        self.X = None
        self.y = None
        self.index = None
        with h5py.File(h5_path, 'r') as hf:
            # self.data_dict['audio_name'] = np.array(
            #     [audio_name.decode() for audio_name in hf['audio_name'][:]])

            self.X = hf['feature'][:].astype(np.float32)
            
            if 'target' in hf.keys():
                self.y = np.array(
                    [scene_label \
                        for scene_label in hf['target'][:]])
                
            # if 'identifier' in hf.keys():
            #     self.data_dict['identifier'] = np.array(
            #         [identifier.decode() for identifier in hf['identifier'][:]])
                
            if 'source_label' in hf.keys():
                self.index = np.array(
                    [source_label.decode() \
                        for source_label in hf['source_label'][:]])
        # self.rotate_angle = rotate_angle
        # print('.. ',len(self.data_dict['feature']))
        # print('...  ',self.data_dict['source_label'])
        # print('len- ',self.data_dict['audio_name'])
        # for s in self.index:
        #     print(s)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], int(self.targets[index])
        X,y_sparse,tmp = self.X[index],self.y[index],self.index[index]
        # print('X ',X.shape)
        # print('y ',y.shape)
        # print('tmp ',tmp.shape)
        # print(tmp)
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

class ASC2020_DANN(Dataset):
    def __init__(self,h5_path):
        # root = './data'
        # processed_folder = os.path.join(root, 'MNIST', 'processed')
        # data_file = 'training.pt'
        # self.data, self.targets = torch.load(os.path.join(processed_folder, data_file))
        self.X = None
        self.y = None
        self.index = None
        with h5py.File(h5_path, 'r') as hf:
            # self.data_dict['audio_name'] = np.array(
            #     [audio_name.decode() for audio_name in hf['audio_name'][:]])

            self.X = hf['feature'][:].astype(np.float32)
            
            if 'target' in hf.keys():
                self.y = np.array(
                    [scene_label \
                        for scene_label in hf['target'][:]])
                
            # if 'identifier' in hf.keys():
            #     self.data_dict['identifier'] = np.array(
            #         [identifier.decode() for identifier in hf['identifier'][:]])
                
            if 'source_label' in hf.keys():
                self.index = np.array(
                    [source_label.decode() \
                        for source_label in hf['source_label'][:]])
        # self.rotate_angle = rotate_angle
        # print('.. ',len(self.data_dict['feature']))
        # print('...  ',self.data_dict['source_label'])
        # print('len- ',self.data_dict['audio_name'])
        # for s in self.index:
        #     print(s)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], int(self.targets[index])
        X,y_sparse,tmp = self.X[index],self.y[index],self.index[index]
        # print('X ',X.shape)
        # print('y ',y.shape)
        # print('tmp ',tmp.shape)
        # print(tmp)
        t = []
        t.append(domain_index[tmp])
        t = np.array(t)
        # print('t   ..',t.shape)
        domain = t
        if domain_index[tmp]>1:
            domain_label = np.array([1,0])
        else:
            domain_label = np.array([0,1])
        #y = sparse_to_categorical(y_sparse,10)
        return X,y_sparse,t,domain,domain_label

    def __len__(self):
        return len(self.X)

    def get(self):
        return self.X,self.y,self.index

class ASC2020_bc(Dataset):
    def __init__(self,h5_path):
        # root = './data'
        # processed_folder = os.path.join(root, 'MNIST', 'processed')
        # data_file = 'training.pt'
        # self.data, self.targets = torch.load(os.path.join(processed_folder, data_file))
        self.X = None
        self.y = None
        self.index = None
        with h5py.File(h5_path, 'r') as hf:
            # self.data_dict['audio_name'] = np.array(
            #     [audio_name.decode() for audio_name in hf['audio_name'][:]])

            X = hf['feature'][:].astype(np.float32)
            
            if 'target' in hf.keys():
                y = np.array(
                    [scene_label \
                        for scene_label in hf['target'][:]])
                
            # if 'identifier' in hf.keys():
            #     self.data_dict['identifier'] = np.array(
            #         [identifier.decode() for identifier in hf['identifier'][:]])
                
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
        # a=0
        # b = 0
        # for label in index:
        #     if label=='a':
        #         a=a+1
        #     if label =='b':
        #         b = b+1
        # print(a,b)
        # print('s_t_index ',len(s_t_index))
        # for i,label in enumerate(self.index):
        #     if label=='a' or label==tg_name:
        #         #print(label)
        #         s_t_index[i]=1
        # print(s_t_index[-1000:-1])
        self.X = np.array(X1)
        self.y = np.array(y1)
        self.index = np.array(index1)
        print(self.X.shape,self.y.shape,self.index.shape)
        # self.rotate_angle = rotate_angle
        # print('.. ',len(self.data_dict['feature']))
        # print('...  ',self.data_dict['source_label'])
        # print('len- ',self.data_dict['audio_name'])
        # for s in self.index:
        #     print(s)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], int(self.targets[index])
        X,y_sparse,tmp = self.X[index],self.y[index],self.index[index]
        # print('X ',X.shape)
        # print('y ',y.shape)
        # print('tmp ',tmp.shape)
        # print(tmp)
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
        #y = sparse_to_categorical(y_sparse,10)
        # X,y_sparse,t,domain,domain_label
        return X,y_sparse,t,domain

    def __len__(self):
        return len(self.X)

    def get(self):
        return self.X,self.y,self.index
