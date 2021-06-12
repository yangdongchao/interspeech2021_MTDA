import numpy as np
import pickle
import random
import pandas as pd
import h5py
import torch
import soundfile
import librosa
labels = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 
    'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park']
    
# classes_num = len(labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
def frequency_masking(mel_spectrogram, frequency_masking_para=13, frequency_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(frequency_mask_num):
        f = random.randrange(0, frequency_masking_para)
        f0 = random.randrange(0, fbank_size[0] - f)

        if (f0 == f0 + f):
            continue

        mel_spectrogram[f0:(f0+f),:] = 0
    return mel_spectrogram
   
   
def time_masking(mel_spectrogram, time_masking_para=40, time_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(time_mask_num):
        t = random.randrange(0, time_masking_para)
        t0 = random.randrange(0, fbank_size[1] - t)

        if (t0 == t0 + t):
            continue

        mel_spectrogram[:, t0:(t0+t)] = 0
    return mel_spectrogram


def cmvn(data):
    shape = data.shape
    eps = 2**-30
    for i in range(shape[0]):
        utt = data[i].squeeze().T
        mean = np.mean(utt, axis=0)
        utt = utt - mean
        std = np.std(utt, axis=0)
        utt = utt / (std + eps)
        utt = utt.T
        data[i] = utt.reshape((utt.shape[0], utt.shape[1], 1))
    return data


def frequency_label(num_sample, num_frequencybins, num_timebins):

    data = np.arange(num_frequencybins, dtype='float32').reshape(num_frequencybins, 1) / num_frequencybins
    #print(data.shape)
    data = np.broadcast_to(data, (num_frequencybins, num_timebins))
    #print(data.shape)
    data = np.broadcast_to(data, (num_sample, num_frequencybins, num_timebins))
    #print(data.shape)
    data = np.expand_dims(data, -1)
    
    return data
def nll_loss(output, target):
    '''Negative likelihood loss. The output should be obtained using F.log_softmax(x). 
    
    Args:
      output: (N, classes_num)
      target: (N, classes_num)
    '''
    loss = - torch.mean(target * output)
    return loss

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
    
'''Load hdf5 file.     
Returns:
    data_dict: dict of data, e.g.:
    {'audio_name': np.array(['a.wav', 'b.wav', ...]), 
        'feature': (audios_num, frames_num, mel_bins)
        'target': (audios_num,), 
        ...}
'''
def load_hdf5(hdf5_path):
    data_dict = {}
    with h5py.File(hdf5_path, 'r') as hf:
        data_dict['audio_name'] = np.array(
            [audio_name.decode() for audio_name in hf['audio_name'][:]])

        data_dict['feature'] = hf['feature'][:].astype(np.float32)
        
        if 'scene_label' in hf.keys():
            data_dict['target'] = np.array(
                [lb_to_idx[scene_label.decode()] \
                    for scene_label in hf['scene_label'][:]])
            
        if 'identifier' in hf.keys():
            data_dict['identifier'] = np.array(
                [identifier.decode() for identifier in hf['identifier'][:]])
            
        if 'source_label' in hf.keys():
            data_dict['source_label'] = np.array(
                [source_label.decode() \
                    for source_label in hf['source_label'][:]]) 
    return data_dict
def load_soft(hdf5_path):
    data_dict = {}
    with h5py.File(hdf5_path, 'r') as hf:
        data_dict['audio_name'] = np.array(
            [audio_name.decode() for audio_name in hf['audio_name'][:]])

        data_dict['soft_label'] = np.array(
                    [scene_label \
                        for scene_label in hf['soft_label'][:]])
    return data_dict
def get_audio_indexes(meta_data, data_dict):
        '''Get train or validate indexes. 
        '''
        audio_indexes = []
        for name in meta_data['audio_name']:
            loct = np.argwhere(data_dict['audio_name'] == name)
            if len(loct) > 0:
                index = loct[0, 0]
                # label = self.idx_to_lb[self.data_dict['target'][index]]
                audio_indexes.append(index)
        return np.array(audio_indexes)

def get_indexes_by_device(indexes, data_dict, source): # 从训练集或者测试集中，根据设备名称选出数据
        '''Get indexes of specific source. 
        '''
        source_indexes = np.array([index for index in indexes \
            if data_dict['source_label'][index] == source])
        return source_indexes

def read_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]
    