# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.optim as optim
import prettytable as pt
from sklearn.metrics import roc_auc_score
from collections import OrderedDict

from networks_v2 import LateMMFAdv
from datasets_ import SurvivalDataset
from utils import read_config
from utils import create_logger
import torch.nn.functional as F
#import wandb
import pandas as pd
import numpy as np
import random
import timm
import timm.scheduler

import tensorboardX

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
foldn = 1
datatype = 'fusion'

def seed_all(seed=1):
    ''' Performs setting random seed for all possible random modules'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(1)

def train(ini_file, logs_dir):

    writer = tensorboardX.SummaryWriter(logs_dir)
    models_dir = os.path.join(logs_dir, 'models')

    # read csv file
    df = pd.read_csv('./dfs_and_data_for_analysis_inner.csv', index_col=0, header=0)
    start_col = df.columns.get_loc('avg_lymcal_area-0_dis-0_lym-0')
    selected_feat = df.columns[start_col:].tolist()

    config = read_config(ini_file)
    model = LateMMFAdv(config)

    ptpath = '/fold%s_mvnet.pt' % foldn

    state_dict = torch.load(ptpath) 
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k]=v
    model_dict = model.state_dict()
    ignore = {k: v for k, v in new_state_dict.items() if k not in model_dict}
    print(ignore)
    weights = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(weights)
    model.load_state_dict(model_dict, True)
    model = model.to(device)
    model.eval()

    ptpath1 = '/cluster_%s.pth' % foldn
    ptpath2 = '/space_%s.pth' % foldn
    new_state_dict = OrderedDict()
    for ptpath in [ptpath1, ptpath2]:
        print(ptpath)
        state_dict = torch.load(ptpath)['model']
        print(state_dict.keys())

        for k, v in state_dict.items():
            if 'space' in ptpath:
                k = 'clinical_branch.'+ ptpath
            elif 'cluster' in ptpath:
                k = 'patho_branch.' + ptpath
            new_state_dict[k]=v
    print()

    model_dict = model.state_dict()
    for k, v in model_dict.items():
        print(k)
    ignore = {k: v for k, v in new_state_dict.items() if k not in model_dict}
    print('ignore')
    print(ignore.keys())
    weights = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(weights)
    model.load_state_dict(model_dict, True)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.FloatTensor([0.5,1.2]),reduction="mean")
    criterion = criterion.to(device)
    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])
    
    # cosine annealing
    scheduler = timm.scheduler.CosineLRScheduler(optimizer,
                                                 t_initial=config['train']['epochs'],
                                                 lr_min=1e-6,
                                                 warmup_t=10,
                                                 warmup_lr_init=1e-6)

    # constructs data loaders based on configuration
    test_dataset = SurvivalDataset('./test%s_0122.csv' % foldn, selected_feat, is_train=False)
    test_dataset_liu = SurvivalDataset('/liu.csv', selected_feat, is_train=False)
    test_dataset_crc = SurvivalDataset('/tj.csv', selected_feat, is_train=False)
    print('label_count:', test_dataset.label_count())
    test_loader_inn = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())
    test_loader_liu = torch.utils.data.DataLoader(
        test_dataset_liu, batch_size=test_dataset_liu.__len__())
    test_loader_crc = torch.utils.data.DataLoader(
        test_dataset_crc, batch_size=test_dataset_crc.__len__())
    # training
    
    testloader_dic = {'inner': test_loader_inn, 'liuyuan': test_loader_liu, 'CRC': test_loader_crc}

        
        
    # valid step
    model.eval()
    
    for test_ind in testloader_dic.keys():
        print(test_ind, 'testing')
        test_loader = testloader_dic[test_ind]

        for Xp, Xc, y, e, name in test_loader:
            # makes predictions
            Xp = Xp.to(device)
            Xc = Xc.to(device)
            y = y.to(device)
            e = e.to(device)

            # makes predictions
            with torch.no_grad():
                risk_pred = model(Xp, Xc)
                e = e.to(torch.int64) 
                e = e.flatten()
                valid_loss = criterion(risk_pred, e)
                
                # calculates c-index
                e = e.cpu().detach().numpy()
                risk_pred = risk_pred.cpu().detach().numpy()
                valid_c = F.softmax(torch.tensor(risk_pred), dim=1).cpu().detach().numpy()
                valid_c = roc_auc_score(e, valid_c[:, 1])


        print('Test: Loss: {:.6f}, C-index: {:.6f}'.format(
            valid_loss.item(), valid_c.item()))
        writer.add_scalar('Valid/{}/Loss'.format(test_ind), valid_loss.item())
        writer.add_scalar('Valid/{}/C-index'.format(test_ind), valid_c.item())

    return 

if __name__ == '__main__':
    # global settings
    logs_dir = './ckpt'
    models_dir = os.path.join(logs_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    logger = create_logger(logs_dir)
    
    print("device:", device)
    # device = 'cpu'
    configs_dir = 'configs'
    params = [
        ('self_mlp_512', 'self_v2.ini')]
    patience = 500
    # training
    headers = []
    values = []

    for name, ini_file in params:
        logger.info('Running {}({})...'.format(name, ini_file))
        train(os.path.join(configs_dir, ini_file), logs_dir)
        headers.append(name)
        print('')
        logger.info('')
    # prints results
    tb = pt.PrettyTable()
    tb.field_names = headers
    tb.add_row(values)
    logger.info(tb)

