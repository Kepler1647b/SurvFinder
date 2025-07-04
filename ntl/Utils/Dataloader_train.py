import torch.utils.data as data_utils
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import random
import glob
import os
from Utils import utils
import math
import torch
import copy
import pandas as pd

utils.seed_torch(seed=0)
Map = {
    "CD10": 0,
    "CD20": 0,
    "CD21": 0,
    "normal": 1,
    "tumor": 2

}


limit_patchs = 0

class ZSData(data_utils.Dataset):

    def __init__(self, path, task, transforms=None, transforms_512 = None, padding=0, bi=True):

        self.bi = bi
        self.__img_name = []
        self.__label_list = []
        self.__img_list = []
        self.train_labels = []
        slide_path_list = glob.glob(os.path.join(path, "*"))
        for slide_path in slide_path_list:
            slidename = os.path.basename(slide_path)
            Type_list = os.listdir(slide_path)
            for Type in Type_list:
                if Type not in Map.keys():
                    continue
                path_arr = glob.glob(os.path.join(slide_path, Type, "*"))
                if ("train" in path) and (Map[Type] != -1) and limit_patchs != 0:
                    random.shuffle(path_arr)
                    path_arr = path_arr[:limit_patchs]
                label_arr = [Map[Type]] * len(path_arr)
                self.__img_name.extend(path_arr)
                self.__label_list.extend(label_arr)
                self.__img_list.extend(['original_256']*len(path_arr))


        self.train_labels = torch.tensor(self.__label_list)
        self.__img_name = np.array(self.__img_name)

        if padding != 0 and len(self.__img_name) % padding != 0:
            need = (len(self.__img_name) // padding + 1) * padding - len(self.__img_name)
            indice = np.random.choice(len(self.__img_name), need, replace=True)
            self.__img_name = np.concatenate([self.__img_name, self.__img_name[indice]])
        
        self.data_transforms = transforms
        self.data_transforms_512 = transforms_512
    def __len__(self):
        return len(self.__img_name)
        
    def get_label_list(self):
        return self.__label_list

    def get_index_dic(self):
        index_dic = []
        for i in range(3):
            index_dic.append([t for t, x in enumerate(self.__label_list) if x == i])
        return index_dic


    def __getitem__(self, item):
        img = Image.open(self.__img_name[item]).copy()
        img_label = self.__label_list[item].copy()
        img = self.data_transforms(img)
        return img, img_label

