import os
import sys
sys.path.append('/home/ydc/DACSE2021/task1a/CIDA-ASC-plot')  # 需要引入models文件夹下的utils文件，所以需要添加路径
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool
import h5py
from data.utils import *
sr = 32000 #44100
duration = 10
num_freq_bin = 64
num_fft = 1024 #2048
hop_length = 500 #int(num_fft / 2)
frames_num = duration*sr/hop_length 
print('frames_num ',frames_num)
num_time_bin = int(np.ceil(duration * sr / hop_length)) 
num_channel = 1
def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return mean, std
def get_norm(data_dict,indexes):
    features = data_dict['feature'][indexes]
    print('feature shape',features.shape)
    features = np.concatenate(features, axis=0)
    (mean, std) = calculate_scalar_of_tensor(features)
    return mean,std
def find_soft_by_name(name,soft_dict):
    # print('soft ',len(soft_dict['audio_name'][:]))
    ans = -1
    for n,name_i in enumerate(soft_dict['audio_name'][:]):
        # print(name_i,name)
        if name_i==name:
            ans = n
            break
    assert ans!=-1
    # print('ans ',ans)
    print(name,ans)
    tmp = soft_dict['soft_label'][ans]
    return np.exp(tmp)

def get_train():
    data_dict = load_hdf5('/home/ydc/DACSE2021/task1a/CIDA-ASC-plot/data/features/logmel.h5') # feature_extract.py 提取得到的feature 文件路径
    # soft_h5 = load_soft('/home/ydc/DACSE2021/task1a/CIDA-ASC-plot/data/features/soft_label.h5')
    train_meta = read_metadata('/home/cdd/code2/dcase2020/task1a/TAU-urban-acoustic-scenes-2020-mobile-development/evaluation_setup/fold1_train.csv') # train.csv路径
    validate_meta = read_metadata('/home/cdd/code2/dcase2020/task1a/TAU-urban-acoustic-scenes-2020-mobile-development/evaluation_setup/fold1_evaluate.csv') # test.csv 路径
    train_audio_indexes = get_audio_indexes(train_meta,data_dict)
    validate_audio_indexes = get_audio_indexes(validate_meta,data_dict)
    all_indexes = np.concatenate((train_audio_indexes,validate_audio_indexes),axis=0)
    new_data_dict = {}
    new_data_dict['audio_name'] = np.array(
            [audio_name for audio_name in data_dict['audio_name'][train_audio_indexes]])
    new_data_dict['feature'] = data_dict['feature'][train_audio_indexes].astype(np.float32)
    new_data_dict['target'] = data_dict['target'][train_audio_indexes]
    new_data_dict['source_label'] = np.array(
                [source_label \
                    for source_label in data_dict['source_label'][train_audio_indexes]])
    new_data_dict['identifier'] = np.array(
                [identifier for identifier in data_dict['identifier'][train_audio_indexes]])
    # new_data_dict['soft_label'] = np.array([find_soft_by_name(name,soft_h5) for name in data_dict['audio_name'][train_audio_indexes]])
    print('self.data_dict ', data_dict['audio_name'][train_audio_indexes].shape)
    print('audio name ',new_data_dict['audio_name'].shape)
    print('soft_label ',new_data_dict['soft_label'].shape)
    print('identifier ',new_data_dict['identifier'].shape)
    # print('soft_label ',new_data_dict['soft_label'][:10])
    print('target ',new_data_dict['target'][:10])
    hf_train = h5py.File('/home/ydc/DACSE2021/task1a/CIDA-ASC-plot/data/features/logmel_train.h5', 'w') # 存储训练数据路径
    hf_train.create_dataset(
        name='audio_name', 
        data=[audio_name.encode() for audio_name in new_data_dict['audio_name'][:]], 
        dtype='S80')
    hf_train.create_dataset(
    name='target', 
    shape=(len(new_data_dict['audio_name'][:]), ), 
    dtype=np.float32)
    for n,scene_label in enumerate(new_data_dict['target'][:]):
        hf_train['target'][n] = scene_label
    if 'identifier' in data_dict.keys():
        hf_train.create_dataset(
            name='identifier', 
            data=[identifier.encode() for identifier in new_data_dict['identifier'][:]], 
            dtype='S24')
            
    if 'source_label' in data_dict.keys():
        hf_train.create_dataset(
            name='source_label', 
            data=[source_label.encode() for source_label in new_data_dict['source_label'][:]], 
            dtype='S8')
    hf_train.create_dataset(
    name='feature', 
    shape=(0, frames_num, num_freq_bin), 
    maxshape=(None, frames_num, num_freq_bin), 
    dtype=np.float32)
    for n,feature in enumerate(new_data_dict['feature'][:]):
        hf_train['feature'].resize((n + 1, frames_num, num_freq_bin))
        hf_train['feature'][n] = feature

    hf_train.close()    
    print('source_label ',new_data_dict['source_label'][0:100])
    print('write new_data_dict success')

def get_test():
    data_dict = load_hdf5('/home/ydc/DACSE2021/task1a/CIDA-ASC-plot/data/features/logmel.h5')
    train_meta = read_metadata('/home/cdd/code2/dcase2020/task1a/TAU-urban-acoustic-scenes-2020-mobile-development/evaluation_setup/fold1_train.csv')
    validate_meta = read_metadata('/home/cdd/code2/dcase2020/task1a/TAU-urban-acoustic-scenes-2020-mobile-development/evaluation_setup/fold1_evaluate.csv')
    train_audio_indexes = get_audio_indexes(train_meta,data_dict)
    validate_audio_indexes = get_audio_indexes(validate_meta,data_dict)
    all_indexes = np.concatenate((train_audio_indexes,validate_audio_indexes),axis=0)
    new_data_dict = {}
    new_data_dict['audio_name'] = np.array(
            [audio_name for audio_name in data_dict['audio_name'][validate_audio_indexes]])
    new_data_dict['feature'] = data_dict['feature'][validate_audio_indexes].astype(np.float32)
    new_data_dict['target'] = data_dict['target'][validate_audio_indexes]
    new_data_dict['source_label'] = np.array(
                [source_label \
                    for source_label in data_dict['source_label'][validate_audio_indexes]])
    new_data_dict['identifier'] = np.array(
                [identifier for identifier in data_dict['identifier'][validate_audio_indexes]])
    print('self.data_dict ', data_dict['audio_name'][validate_audio_indexes].shape)
    print('audio name ',new_data_dict['audio_name'].shape)
    print('source_label ',new_data_dict['source_label'].shape)
    print('identifier ',new_data_dict['identifier'].shape)
    hf_train = h5py.File('/home/ydc/DACSE2021/task1a/CIDA-ASC-plot/data/features/logmel_test.h5', 'w')
    hf_train.create_dataset(
        name='audio_name', 
        data=[audio_name.encode() for audio_name in new_data_dict['audio_name'][:]], 
        dtype='S80')
    hf_train.create_dataset(
    name='target', 
    shape=(len(new_data_dict['audio_name'][:]), ), 
    dtype=np.float32)
    for n,scene_label in enumerate(new_data_dict['target'][:]):
        hf_train['target'][n] = scene_label
    if 'identifier' in data_dict.keys():
        hf_train.create_dataset(
            name='identifier', 
            data=[identifier.encode() for identifier in new_data_dict['identifier'][:]], 
            dtype='S24')
            
    if 'source_label' in data_dict.keys():
        hf_train.create_dataset(
            name='source_label', 
            data=[source_label.encode() for source_label in new_data_dict['source_label'][:]], 
            dtype='S8')

    hf_train.create_dataset(
    name='feature', 
    shape=(0, frames_num, num_freq_bin), 
    maxshape=(None, frames_num, num_freq_bin), 
    dtype=np.float32)
    for n,feature in enumerate(new_data_dict['feature'][:]):
        hf_train['feature'].resize((n + 1, frames_num, num_freq_bin))
        hf_train['feature'][n] = feature

    hf_train.close()    
    print('source_label ',new_data_dict['source_label'][0:10])
    print('write new_data_dict success')
get_train()
get_test()
# get_mean_std()