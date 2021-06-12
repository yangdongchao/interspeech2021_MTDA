import os
import sys
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool
import h5py
# from models.utils import *

def read_metadata(metadata_path):
    '''Read metadata from a csv file. 
    
    Returns:
      meta_dict: dict of meta data, e.g.:
        {'audio_name': np.array(['a.wav', 'b.wav', ...]), 
         'scene_label': np.array(['airport', 'bus', ...]), 
         ...}
    '''
    df = pd.read_csv(metadata_path, sep='\t')
    
    meta_dict = {}
    
    meta_dict['audio_name'] = np.array(
        [name.split('/')[1] for name in df['filename'].tolist()])
    
    if 'scene_label' in df.keys():
        meta_dict['scene_label'] = np.array(df['scene_label'])
        
    if 'identifier' in df.keys():
        meta_dict['identifier'] = np.array(df['identifier'])
        
    if 'source_label' in df.keys():
        meta_dict['source_label'] = np.array(df['source_label'])
    
    return meta_dict

dataset_path = '/home/cdd/code2/dcase2020/task1a/TAU-urban-acoustic-scenes-2020-mobile-development/'
file_path = '/home/cdd/code2/dcase2020/task1a/TAU-urban-acoustic-scenes-2020-mobile-development/'
audio_data_path = '/home/cdd/code2/dcase2020/task1a/TAU-urban-acoustic-scenes-2020-mobile-development/audio/'
csv_file = '/home/cdd/code2/dcase2020/task1a/TAU-urban-acoustic-scenes-2020-mobile-development/meta.csv'
output_path = '/home/ydc/DACSE2021/task1a/CIDA-ASC-plot/data/features/all_data_no_norm.h5'
feature_type = 'logmel'

sr = 32000 #44100
duration = 10
num_freq_bin = 64
num_fft = 1024 #2048
hop_length = 500 #int(num_fft / 2)
frames_num = duration*sr/hop_length + 1 
print('frames_num ',frames_num)
num_time_bin = int(np.ceil(duration * sr / hop_length)) +1
num_channel = 1

print('Extracting features of all audio files ...')
meta_dict = read_metadata('/home/cdd/code2/dcase2020/task1a/TAU-urban-acoustic-scenes-2020-mobile-development/meta.csv')
# Hdf5 file for storing features and targets
hf = h5py.File(output_path, 'w')

hf.create_dataset(
    name='audio_name', 
    data=[audio_name.encode() for audio_name in meta_dict['audio_name']], 
    dtype='S80')

if 'scene_label' in meta_dict.keys():
    hf.create_dataset(
        name='scene_label', 
        data=[scene_label.encode() for scene_label in meta_dict['scene_label']], 
        dtype='S24')
        
if 'identifier' in meta_dict.keys():
    hf.create_dataset(
        name='identifier', 
        data=[identifier.encode() for identifier in meta_dict['identifier']], 
        dtype='S24')
        
if 'source_label' in meta_dict.keys():
    hf.create_dataset(
        name='source_label', 
        data=[source_label.encode() for source_label in meta_dict['source_label']], 
        dtype='S8')

hf.create_dataset(
    name='feature', 
    shape=(0, frames_num, num_freq_bin), 
    maxshape=(None, frames_num, num_freq_bin), 
    dtype=np.float32)
n = 0
for audio_name in meta_dict['audio_name']:
    audio_path = os.path.join(audio_data_path, audio_name)
    print(n,audio_path)
    stereo, fs = sound.read(audio_path, stop=duration*sr)
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    #print('logmel_data ',logmel_data.shape)
    logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)

    logmel_data = np.log(logmel_data+1e-8)
    

    feat_data = logmel_data
    #feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
    feat_data = feat_data.reshape((feat_data.shape[1],feat_data.shape[0]))
    hf['feature'].resize((n + 1, frames_num, num_freq_bin))
    hf['feature'][n] = feat_data
    n += 1
print('n ',n)
hf.close()
        
        

