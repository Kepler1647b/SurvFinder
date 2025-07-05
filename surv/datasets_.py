# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import pandas as pd
import torch
import os
import pickle

from torch.utils.data import Dataset

foldn = 1

feat_path_inn = '/inner/fold{}/feature_kmeans'.format(foldn)
feat_path_tj = '/CRC/fold{}/feature_kmeans'.format(foldn)
feat_path_liu = '/liuyuan/fold{}/feature_kmeans_vahadafne'.format(foldn)

class SurvivalDataset(Dataset):
    def __init__(self, h5_file, selected_feat, is_train):
        self.selected_feat = selected_feat
        self.X_cli, self.e, self.y, self.namelist= self._read_csv(h5_file, is_train)
        if 'CRC' in h5_file:
            self.X_pat, self.e, self.y = self._read_pkl(h5_file, feat_path_tj, is_train)
        elif 'liuyuan' in h5_file:
            self.X_pat, self.e, self.y = self._read_pkl(h5_file, feat_path_liu, is_train)
        else:
            self.X_pat, self.e, self.y = self._read_pkl(h5_file, feat_path_inn, is_train)

        print('=> load {} samples'.format(self.e.shape[0]))
    
    def get_labels(self):
        return self.e

    def _read_h5_file(self, h5_file, is_train):
        ''' The function to parsing data from .h5 file.

        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test'
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
        return X, e, y

    def _read_csv(self, csvpath, is_train):
        ''' The function to parsing data from .h5 file.

        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        csv = pd.read_csv(csvpath)
        namelist = csv['病理号'].tolist()
        if self.selected_feat == []:
            csv = csv.iloc[:,1:]
        else:
            csv = csv[['time','censor']+self.selected_feat]
        csv = csv.dropna()
        
        X = np.array(csv.iloc[:,2:])
        e = np.array(csv['censor'].tolist())
        y = np.array(csv['time'].tolist())
        print(np.isnan(X).any())
        #print(X,e,y)

        #print(type(e))
        return X, e, y, namelist
    
    def _read_pkl(self, csvpath, feat_path, is_train):
        csv = pd.read_csv(csvpath)
        namelist = csv['病理号'].tolist()
        if self.selected_feat == []:
            csv = csv.iloc[:,1:]
        else:
            csv = csv[['time','censor']+self.selected_feat]
        csv = csv.dropna()
        
        e = np.array(csv['censor'].tolist())
        y = np.array(csv['time'].tolist())
        X = []
        for name in namelist:
            with open(os.path.join(feat_path, str(name) + ".pkl"), "rb") as f:
                feature = pickle.load(f)
                feature = np.array(feature)    
                X.append(feature)
        X = np.array(X)
                
        print(np.isnan(X).any())
        return X, e, y


    def _normalize(self):
        self.X = (self.X-self.X.min(axis=0)) / \
            (self.X.max(axis=0)-self.X.min(axis=0))

    def __getitem__(self, item):
        X_pat_item = self.X_pat[item] # (m)
        X_cli_item = self.X_cli[item]
        y_item = self.y[item] # (1)
        e_item = self.e[item]
        X_pat_tensor = torch.tensor(X_pat_item)
        X_cli_tensor = torch.tensor(X_cli_item)
        X_pat_tensor = torch.Tensor.float(X_pat_tensor)
        X_cli_tensor = torch.Tensor.float(X_cli_tensor)
        e_tensor = torch.Tensor([e_item])
        y_tensor = torch.Tensor([y_item])
        name = self.namelist[item]
        return X_pat_tensor, X_cli_tensor, y_tensor, e_tensor, name

    def __len__(self):
        return self.y.shape[0]
    
    def label_count(self):
        a = 0
        b = 0
        for i in list(self.y):
            if i < 24:
                a += 1
            else:
                b += 1
        return a, b