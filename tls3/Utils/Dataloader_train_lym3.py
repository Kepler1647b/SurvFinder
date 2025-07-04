import torch.utils.data as data_utils
from PIL import Image
import numpy as np
import random
import glob
import os
from Utils import utils
import math
import torch
Image.MAX_IMAGE_PIXELS = None

utils.seed_torch(seed=0)
Map1 = {
    "CD10": 0,
    "CD20": 0,
    "CD21": 0,
    "normal": 1,
    "tumor": 2

}

Map2 = {
    'CD20': 0,
    'CD21': 1,
    'CD10': 2
}
# Map2 = {
#     'CD20': 0,
#     'CD21': 1,
# }

limit_patchs = 0

# dpath = '/home/21/zihan/Storage/lympho/img_lym_120'
dpath = '/data15/data15_5/dexia/code_ctrans_ntl/lympho_bag/img_lym_120_40x'

class ZSData(data_utils.Dataset):

    def __init__(self, path, task, transforms=None, transforms_512 = None, padding=0, bi=True):

        self.bi = bi
        self.__img_name = []
        self.__label_list = []
        self.__img_list = []
        self.train_labels = []

        slide_path_list = glob.glob(os.path.join(path, "*"))
        if task == 'ntl':
            Map = Map1
        elif task == 'lym':
            Map = Map2
        for slide_path in slide_path_list:
            slidename = os.path.basename(slide_path)
            Type_list = os.listdir(slide_path)
            for Type in Type_list:
                if Type not in Map.keys():
                    continue
                path_arr = glob.glob(os.path.join(dpath, slidename, Type, "*"))
                if ("train" in path) and (Map[Type] != -1) and limit_patchs != 0:
                    random.shuffle(path_arr)
                    path_arr = path_arr[:limit_patchs]
                label_arr = [Map[Type]] * len(path_arr)
                self.__img_name.extend(path_arr)
                self.__label_list.extend(label_arr)
                self.__img_list.extend(['original_256']*len(path_arr))

            '''path_add_256 = os.path.join(utils.path_add_256, slidename)
            Type_list = os.listdir(path_add_256)
            for Type in Type_list:
                if Type not in Map.keys():
                    continue
                path_arr = glob.glob(os.path.join(path_add_256, Type, "*"))
                if ("train" in path) and (Map[Type] != -1) and limit_patchs != 0:
                    random.shuffle(path_arr)
                    path_arr = path_arr[:limit_patchs]
                label_arr = [Map[Type]] * len(path_arr)
                self.__img_name.extend(path_arr)
                self.__label_list.extend(label_arr)
                self.__img_list.extend(['add_256']*len(path_arr))
            
            path_add_512 = os.path.join(utils.path_add_512, slidename)
            Type_list = os.listdir(path_add_512)
            for Type in Type_list:
                if Type not in Map.keys():
                    continue
                path_arr = glob.glob(os.path.join(path_add_512, Type, "*"))
                if ("train" in path) and (Map[Type] != -1) and limit_patchs != 0:
                    random.shuffle(path_arr)
                    path_arr = path_arr[:limit_patchs]
                label_arr = [Map[Type]] * len(path_arr)
                self.__img_name.extend(path_arr)
                self.__label_list.extend(label_arr)
                self.__img_list.extend(['add_512']*len(path_arr))'''

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
        # for i in range(2):
        for i in range(3):
            index_dic.append([t for t, x in enumerate(self.__label_list) if x == i])
        return index_dic


    def __getitem__(self, item):
        img = Image.open(self.__img_name[item])
        img_label = self.__label_list[item]

        if self.data_transforms is not None:
            if self.__img_list[item] == 'add_512':
                img = self.data_transforms_512(img)
            else:
                img = self.data_transforms(img)
        return img, img_label

    def get_labels(self):
        return self.__label_list
