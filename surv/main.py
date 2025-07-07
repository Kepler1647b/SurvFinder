
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
    train_dataset = SurvivalDataset('./train%s_0122.csv' % foldn, selected_feat, is_train=True)
    val_dataset = SurvivalDataset('./val%s_0122.csv' % foldn, selected_feat, is_train=False)
    print('label_count:', train_dataset.label_count(), val_dataset.label_count())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True)
    val_loader_inn = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_dataset.__len__())
    # training
    
    valloader_dic = {'inner': val_loader_inn}
    for epoch in range(1, config['train']['epochs']+1):
        scheduler.step(epoch-1)
        lr = optimizer.param_groups[0]['lr']

        # train step
        model.train()
        e_list = []
        risk_pred_list = []
        for Xp, Xc, y, e, name in train_loader:
            # makes predictions
            Xp = Xp.to(device)
            Xc = Xc.to(device)
            y = y.to(device)
            e = e.to(device)

            risk_pred = model(Xp, Xc)
            e = e.to(torch.int64)
            e = e.flatten()
            train_loss = criterion(risk_pred, e)
            e = e.cpu().detach().numpy()

            if len(e_list) == 0:
                e_list = e
                risk_pred_list = risk_pred.cpu().detach().numpy()
            else:
                e_list = np.concatenate((e_list, e))
                risk_pred_list = np.concatenate((risk_pred_list, risk_pred.cpu().detach().numpy()))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        e_list = e_list.flatten()
        e = e_list
        train_c = F.softmax(torch.tensor(risk_pred_list), dim=1).cpu().detach().numpy()
        train_c = roc_auc_score(e_list, train_c[:, 1])
        
        
        # valid step
        model.eval()
        
        for val_ind in valloader_dic.keys():
            print(val_ind, 'testing')
            val_loader = valloader_dic[val_ind]

            for Xp, Xc, y, e, name in val_loader:
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

            if val_ind == 'inner':
                print('Train: Epoch: {}, Loss: {:.6f}, C-index: {:.6f}, LR: {:.6f}'.format(
                    epoch, train_loss.item(), train_c.item(), lr))
                writer.add_scalar('Train/Loss', train_loss.item(), epoch)
                writer.add_scalar('Train/C-index', train_c.item(), epoch)
                writer.add_scalar('Train/LR', lr, epoch)
            print('Valid: Epoch: {}, Loss: {:.6f}, C-index: {:.6f}'.format(
                epoch, valid_loss.item(), valid_c.item()))
            writer.add_scalar('Valid/{}/Loss'.format(val_ind), valid_loss.item(), epoch)
            writer.add_scalar('Valid/{}/C-index'.format(val_ind), valid_c.item(), epoch)

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

