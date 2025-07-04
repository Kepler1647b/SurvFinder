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

foldn = 2

#feat_path_inn = '/data15/data15_5/dexia/CLAM-ntl-V2/ntl_lympho_bag_40x_resnet_thum_fold1_inner_kmeans'
feat_path_inn = '/home/21/zihan/Storage/lympho/lympho_analysis/features/inner/fold{}/feature_kmeans'.format(foldn)
npy_path1 = '/home/21/zihan/Storage/lympho/lympho_analysis/features/inner/feature_avg_type'
npy_path2 = '/home/21/zihan/Storage/lympho/lympho_analysis/features/inner/feature_avg_distance'
npy_path3 = '/home/21/zihan/Storage/lympho/lympho_analysis/features/inner/feature_avg_area'
feat_path_tj = '/home/21/zihan/Storage/lympho/lympho_analysis/features/CRC/fold{}/feature_kmeans'.format(foldn)
feat_path_liu = '/home/21/zihan/Storage/lympho/lympho_analysis/features/liuyuan/fold{}/feature_kmeans_vahadane'.format(foldn)

rm_list = []
# rm_list = [654213, 651999, 663409, 650191, 664229, 645390, 652585, 666421, 652764, 643459, 665102, 652766, 657163, 657001, 657887, 627011, 663251, 665953, 660335, 627382, 637279, 664216,581490, 580143, 575194, 584087, 566565, 583477, 578638, 566812, 579255, 575293, 567520, 578336, 567104, 581484, 568760, 572364, 586144, 570208, 633724, 578936, 571342, 571604,609527, 608720, 597437, 608414, 589656, 598341, 587383, 587449, 600934, 612671, 597782, 600231, 593432, 597919, 606619, 615330, 613957, 603907, 604365, 578642, 665719, 575055,628521, 638909, 618336, 623417, 624143, 637464, 639077, 618372, 634427, 641539, 623467, 635959, 618702, 636169, 631691, 626331, 636990, 623916, 633864, 636293, 601935, 625079,693893, 667094, 681231, 694270, 679472, 684771, 695211, 667240, 670744, 675045, 695252, 676472, 671553, 670550, 689798, 690864, 673004, 684790, 693471, 693243, 671521, 671699]
#rm_list = [566101, 576396, 572364, 578638, 585661, 570330, 567104, 580143, 577574, 566812, 575194, 570208, 567520, 568760, 581490, 578336, 583839, 581952, 633724, 578936, 571342, 571604,
#615330, 606619, 600231, 613957, 587832, 597782, 587449, 613923, 605012, 612671, 609527, 604564, 612085, 596821, 602960, 603907, 586301, 598509, 575055, 604365, 578642, 665719,
#628521, 631865, 639241, 632538, 635959, 634427, 598863, 636169, 633858, 618702, 639077, 639772, 624143, 625182, 618372, 632835, 631313, 623916, 636293, 633864, 625079, 608138,
#659441, 643484, 657163, 665821, 650719, 654719, 657887, 647285, 645390, 652764, 650191, 665102, 652263, 647512, 652585, 627011, 665953, 662199, 627382, 660305, 648925, 660335,
#690893, 679472, 667240, 697233, 691117, 681231, 694865, 677675, 695252, 692892, 675045, 694270, 671553, 673004, 684790, 676472, 695111, 670550, 693243, 671521, 693471, 662231]
class SurvivalDataset(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, selected_feat, is_train):
        ''' Loading data from .h5 file based on (h5_file, is_train).

        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        #self.X, self.e, self.y = \
            #self._read_h5_file(h5_file, is_train)
        self.selected_feat = selected_feat
        self.X_cli, self.e, self.y, self.namelist= self._read_csv(h5_file, is_train)
        if 'CRC' in h5_file:
            self.X_pat, self.e, self.y = self._read_pkl(h5_file, feat_path_tj, is_train)
        elif 'liuyuan' in h5_file:
            self.X_pat, self.e, self.y = self._read_pkl(h5_file, feat_path_liu, is_train)
        else:
            self.X_pat, self.e, self.y = self._read_pkl(h5_file, feat_path_inn, is_train)

        # normalizes data
        #self._normalize()

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
        if 'inner' in csvpath:
            csv = csv.drop(csv[csv['病理号'].isin(rm_list)].index)
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
        if 'inner' in csvpath:
            csv = csv.drop(csv[csv['病理号'].isin(rm_list)].index)
        namelist = csv['病理号'].tolist()
        if self.selected_feat == []:
            csv = csv.iloc[:,1:]
        else:
            csv = csv[['time','censor']+self.selected_feat]
        csv = csv.dropna()
        
        #X = np.array(csv.iloc[:,2:])
        e = np.array(csv['censor'].tolist())
        y = np.array(csv['time'].tolist())
        X = []
        #namelist = csv['病理号'].tolist()
        for name in namelist:
            with open(os.path.join(feat_path, str(name) + ".pkl"), "rb") as f:
                feature = pickle.load(f)
                feature = np.array(feature)    
                #feature = feature.flatten()
                X.append(feature)
        X = np.array(X)
                
        print(np.isnan(X).any())
        #print(X,e,y)

        #print(type(e))
        return X, e, y

    def _read_npy(self, csvpath, is_train):
        csv = pd.read_csv(csvpath)
        namelist = csv['病理号'].tolist()
        if self.selected_feat == []:
            csv = csv.iloc[:,1:]
        else:
            csv = csv[['time','censor']+self.selected_feat]
        csv = csv.dropna()
        
        #X = np.array(csv.iloc[:,2:])
        e = np.array(csv['censor'].tolist())
        y = np.array(csv['time'].tolist())
        X = []
        #namelist = csv['病理号'].tolist()
        '''for name in namelist:
            with open(os.path.join(feat_path, str(name) + ".pkl"), "rb") as f:
                feature = pickle.load(f)
                feature = np.array(feature)    
                feature = feature.flatten()
                X.append(feature)'''
        for name in namelist:
            #print(name)
            npy1 = np.load(os.path.join(npy_path1, str(name)+'.npy'), allow_pickle=True)
            npy2 = np.load(os.path.join(npy_path2, str(name)+'.npy'), allow_pickle=True)
            npy3 = np.load(os.path.join(npy_path3, str(name)+'.npy'), allow_pickle=True)
            npy_list = list(npy1.item().values()) + list(npy2.item().values()) + list(npy3.item().values())
            npy_arr = []
            for i in range(len(npy_list)):
                npy_arr.append(npy_list[i])
            npy_arr = np.array(npy_arr)
            #print(npy_arr)
            #npy_arr = npy_arr.flatten()
            X.append(npy_arr)
        X = np.array(X)
        #print(X)
                
        #print(np.isnan(X).any())
        #print(X,e,y)

        #print(type(e))
        return X, e, y


    def _normalize(self):
        ''' Performs normalizing X data. '''
        self.X = (self.X-self.X.min(axis=0)) / \
            (self.X.max(axis=0)-self.X.min(axis=0))

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_pat_item = self.X_pat[item] # (m)
        X_cli_item = self.X_cli[item]
        #e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)
        e_item = int(y_item<36)
        # constructs torch.Tensor object
        X_pat_tensor = torch.tensor(X_pat_item)
        X_cli_tensor = torch.tensor(X_cli_item)
        #print(torch.isnan(X_tensor).any())
        #print(X_item)
        #print(X_tensor)
        X_pat_tensor = torch.Tensor.float(X_pat_tensor)
        X_cli_tensor = torch.Tensor.float(X_cli_tensor)
        #print(torch.isnan(X_tensor).any())
        #print(X_tensor.dtype)
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
            if i < 36:
                a += 1
            else:
                b += 1
        return a, b