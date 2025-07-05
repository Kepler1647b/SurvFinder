
import sys
sys.path.append('../')
import os
import time
import numpy as np
import Utils.config_lympho as CONFIG
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
#from Utils.Dataloader_innertest_lym import ZSData
from Utils.Dataloader_train_lym3 import ZSData
from Utils import utils
from Model.create_model import create_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from pytorch_balanced_sampler.sampler import SamplerFactory
from prefetch_generator import BackgroundGenerator
import torch.multiprocessing as mp
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import pandas as pd
from collections import OrderedDict

plt.switch_backend('agg')

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_val_csv(modellist, dataloaders, phase, criterion, optimizer):
    print('start val csv')
    running_loss = 0.0

    all_labels = np.array([])
    all_predicts = np.array([])
    all_path = []
    all_vl_1, all_vl_2, all_vl_3 = [],[],[]
    all_path = []

    for i, (inputs, labels, path) in tqdm(enumerate(dataloaders)):

        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad()

        with torch.torch.set_grad_enabled(phase == 'train'):
            tot_outputs = []
            for model in modellist:
                output = model(inputs)
                output = F.softmax(output, dim=1)
                tot_outputs.append(output)
                print(output)
                print(tot_outputs)
            print(torch.stack(tot_outputs))
            print(torch.stack(tot_outputs).shape)
            outputs = torch.mean(torch.stack(tot_outputs), dim=0)
            print(outputs)
            _, preds = torch.max(outputs, 1)
            value = outputs
            _loss = criterion(outputs, labels)
            
            # update model and optimizer
            if phase == 'train':
                _loss.backward()
                optimizer.step()

        # update train/valid diagnostics
        running_loss += _loss.item() * inputs.size(0)

        all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
        all_predicts = np.concatenate((all_predicts, preds.cpu().numpy()))
        all_path += path
            
        if i == 0:
            all_values = value.detach().cpu().numpy()
            onehot_labels = utils.one_hot(labels.cpu().numpy())
            onehot_preds = utils.one_hot(preds.cpu().numpy())
        else:
            all_values = np.concatenate((all_values, value.detach().cpu().numpy()))
            onehot_labels = np.concatenate((onehot_labels, utils.one_hot(labels.cpu().numpy())))
            onehot_preds = np.concatenate((onehot_preds, utils.one_hot(preds.cpu().numpy())))
        v = value.detach().cpu().numpy()
        print(v)
        #all_value_list.append(value.detach().cpu().numpy())
        all_vl_1.extend(list(v[:,0]))
        all_vl_2.extend(list(v[:,1]))
        all_vl_3.extend(list(v[:,2]))

    aucdic, fprdic, tprdic = utils.multi_auc(onehot_labels, all_values)
    eer_dic = utils.eer_threshold(onehot_labels, all_values)
    mAP = utils.compute_mAP(onehot_labels, onehot_preds)
    _, _, _ = utils.print_result_multi(onehot_labels, all_values, onehot_preds)

    # calculate the sensitivity and specificity of each class
    # first calculate the TP, TN, FP, FN of each class
    TP = np.zeros(3)
    TN = np.zeros(3)
    FP = np.zeros(3)
    FN = np.zeros(3)
    for i in range(3):
        TP[i] = ((all_predicts == i) & (all_labels == i)).sum()
        TN[i] = ((all_predicts != i) & (all_labels != i)).sum()
        FP[i] = ((all_predicts == i) & (all_labels != i)).sum()
        FN[i] = ((all_predicts != i) & (all_labels == i)).sum()
    # then calculate the sensitivity and specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    print('sensitivity:', sensitivity)
    print('specificity:', specificity)
    

    # print train/valid diagnostics
    Loss = running_loss / (len(all_labels))
    Acc = (all_labels == all_predicts).sum() / len(all_predicts)
    lympho_acc = ((all_predicts == 0) & (all_labels == 0)).sum() / (all_labels == 0).sum()
    normal_acc = ((all_predicts == 1) & (all_labels == 1)).sum() / (all_labels == 1).sum()
    tumor_acc = ((all_predicts == 2) & (all_labels == 2)).sum() / (all_labels == 2).sum()
    print('label_sum:', (all_labels == 0).sum(), (all_labels == 1).sum(), (all_labels == 2).sum())
    print('valid eer:', eer_dic)
    print('valid AUROC:', aucdic)


    # process all_path into slide_name and bag_name
    slide_name = []
    type_name = []
    bag_name = []
    for idx, cur_path in enumerate(all_path):
        bag_name.append(cur_path.split('/')[-1].split('.')[0])
        type_name.append(cur_path.split('/')[-2])
        slide_name.append(cur_path.split('/')[-3])
    slide_name = np.array(slide_name)
    type_name = np.array(type_name)
    bag_name = np.array(bag_name)
    csv_dict = {"slide_name": type_name,  "bag_name": bag_name, "predictions": all_predicts.astype(int), 'value0': all_vl_1, 'value1': all_vl_2, 'value2': all_vl_3}
    df = pd.DataFrame(csv_dict)
    return df

def run(model, dataloaders, phase, criterion, optimizer):
    
    if phase == 'train':
        model.train()
    elif phase == 'valid':
        model.eval()
    else:
        raise Exception("Error phase")
    running_loss = 0.0

    all_labels = np.array([])
    all_predicts = np.array([])

    for i, (inputs, labels, pos) in tqdm(enumerate(dataloaders)):

        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad()

        with torch.torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            value = F.softmax(outputs, dim=1)
            _loss = criterion(outputs, labels)
            
            # update model and optimizer
            if phase == 'train':
                _loss.backward()
                optimizer.step()

        # update train/valid diagnostics
        running_loss += _loss.item() * inputs.size(0)

        all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
        all_predicts = np.concatenate((all_predicts, preds.cpu().numpy()))
        if i == 0:
            all_values = value.detach().cpu().numpy()
            onehot_labels = utils.one_hot(labels.cpu().numpy())
            onehot_preds = utils.one_hot(preds.cpu().numpy())
        else:
            all_values = np.concatenate((all_values, value.detach().cpu().numpy()))
            onehot_labels = np.concatenate((onehot_labels, utils.one_hot(labels.cpu().numpy())))
            onehot_preds = np.concatenate((onehot_preds, utils.one_hot(preds.cpu().numpy())))
    
    aucdic, fprdic, tprdic = utils.multi_auc(onehot_labels, all_values)
    eer_dic = utils.eer_threshold(onehot_labels, all_values)
    mAP = utils.compute_mAP(onehot_labels, onehot_preds)
    Loss = running_loss / (len(all_labels))
    Acc = (all_labels == all_predicts).sum() / len(all_predicts)
    lympho_acc = ((all_predicts == 0) & (all_labels == 0)).sum() / (all_labels == 0).sum()
    normal_acc = ((all_predicts == 1) & (all_labels == 1)).sum() / (all_labels == 1).sum()
    tumor_acc = ((all_predicts == 2) & (all_labels == 2)).sum() / (all_labels == 2).sum()
    print('label_sum:', (all_labels == 0).sum(), (all_labels == 1).sum(), (all_labels == 2).sum())

    Writer.add_scalar('%s/Loss' % phase.capitalize(), Loss, global_epoch)  #
    Writer.add_scalar('%s/Acc' % phase.capitalize(), Acc, global_epoch)  #
    Writer.add_scalar('%s/lympho_acc' % phase.capitalize(), lympho_acc, global_epoch)
    Writer.add_scalar('%s/normal_acc' % phase.capitalize(), normal_acc, global_epoch)
    Writer.add_scalar('%s/tumor_acc' % phase.capitalize(), tumor_acc, global_epoch)

    return Loss, Acc, lympho_acc, normal_acc, tumor_acc, aucdic, eer_dic, mAP


def start_train(train_loader, valid_loader, test_loader, modellist, device, criterion, optimizer, scheduler, num_epochs, resume, stat):
    
    best_acc = 0.0
    best_loss = 10000000
    best_auc = 0
    auc_epoch = 0
    loss_epoch = 0

    if stat == 'train':
        model = modellist[0]
        for epoch in range(1 + resume, num_epochs + 1):
            global global_epoch
            global_epoch = epoch
            print('\n##### Epoch [{}/{}]'.format(epoch, num_epochs))
            
            print('\n####### Train #######')

            t_Loss, t_Acc, t_lympho_acc, t_normal_acc, t_tumor_acc, t_aucdic, t_eer, t_mAP = run(model, train_loader, 'train',
                                                                                criterion, optimizer)

            print('\n####### Valid #######')
            v_Loss, v_Acc, v_lympho_acc, v_normal_acc, v_tumor_acc, v_aucdic, v_eer, v_mAP = run(model, valid_loader, 'valid', criterion, optimizer)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            Writer.add_scalar('Learning Rate', current_lr, epoch)   #
            print("Epoch {} with lr {:.15f}: t_loss: {:.4f} t_Acc: {:.4f} t_lympho_acc:{:.4f} t_normal_acc: {:.4f} t_tumor_acc: {:.4f}".format(epoch,
                                                        current_lr, t_Loss, t_Acc, t_lympho_acc, t_normal_acc, t_tumor_acc))
            print('train AUROC:', t_aucdic)
            print('train eer:', t_eer)
            print('train mAP:', t_mAP)
            print("Epoch {} with lr {:.15f}: v_loss: {:.4f} v_Acc: {:.4f} v_lympho_acc:{:.4f} v_normal_acc: {:.4f} v_tumor_acc: {:.4f}".format(epoch,
                    current_lr, v_Loss, v_Acc, v_lympho_acc, v_normal_acc, v_tumor_acc))
            print('valid AUROC:', v_aucdic)
            print('valid eer:', v_eer)
            print('valid mAP:', v_mAP)
            print('best auc epoch:' , auc_epoch)
            print('best loss epoch:', loss_epoch)

        torch.save(model.state_dict(), os.path.join(
            checkpoints, 'FINAL.pt')
        )
    elif stat == 'test':
        df = get_val_csv(modellist, test_loader, 'test', criterion, optimizer)
        print('getting val csv')
        savepath = '/csvpath'   
        for slidename in list(set(df['slide_name'].tolist())):
            if not os.path.exists(os.path.join(savepath, slidename[:12])):
                os.makedirs(os.path.join(savepath, slidename[:12]))
            df_slide = df[(df['slide_name'] == slidename)]
            df_slide.to_csv(os.path.join(savepath, slidename[:12], slidename + '_predict_lym3.csv'))

def prepare_data(dataset_path, task, batch_size, stat):
    train_loader, valid_loader, test_loader = '','',''
    if stat == 'train':
        train_dataset = ZSData(os.path.join(dataset_path, "train"), task, transforms=utils.transform_train_lym3, transforms_512 = utils.transform_train_512, bi=True, padding=0)
        valid_dataset = ZSData(os.path.join(dataset_path, "test"), task, transforms=utils.transform_valid_lym3, transforms_512 = utils.transform_valid_512, bi=True, padding=0)
        label_list_train = train_dataset.get_label_list()
        label_list_valid = valid_dataset.get_label_list()
        class_idxs = train_dataset.get_index_dic()
        batch_sampler = SamplerFactory().get(
        class_idxs=class_idxs,
        batch_size=batch_size,
        n_batches=int(len(label_list_train)/batch_size),
        alpha=0.5,
        kind='fixed')

        print('train_data', label_list_train.count(0), label_list_train.count(1), label_list_train.count(2))
        print('valid_data', label_list_valid.count(0), label_list_valid.count(1), label_list_valid.count(2))
        weights, weight_per_class = utils.make_weights_for_balanced_classes(train_dataset.get_label_list(), nclasses=3)
        train_loader = DataLoaderX(train_dataset, batch_sampler = batch_sampler, num_workers=8, pin_memory=False)
        valid_loader = DataLoaderX(valid_dataset, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=False)
    elif stat == 'test':
        print(dataset_path)
        test_dataset = ZSData(os.path.join(dataset_path, str(args.foldn), 'test'), task, transforms=utils.transform_valid_lym3, transforms_512 = utils.transform_valid_512, bi=True, padding=0)
        label_list_test = test_dataset.get_label_list()
        print('test_data', label_list_test.count(0), label_list_test.count(1), label_list_test.count(2))
        test_loader = DataLoaderX(test_dataset, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=False)
        weights, weight_per_class = utils.make_weights_for_balanced_classes(test_dataset.get_label_list(), nclasses=3)
    return train_loader, valid_loader, test_loader, weight_per_class
   
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--TrainFolder", dest='train_folder', type=str, help='the datasets path')
    parser.add_argument("--FoldN", dest='foldn', default=1, type=int, help='choose which one to training')
    parser.add_argument("--Loss", dest='loss_func', default='cross', type=str, help='choose the loss function')
    parser.add_argument("--Epochs", dest='epochs', default=50, type=int, help='the training epoch')
    parser.add_argument("--Seed", dest='seed', default=0, type=int, help='select the seed')
    parser.add_argument("--Model", dest='model', default='resnet18', type=str, help='choose a model to use')
    parser.add_argument("--BatchSize", dest='batch_size', default=64, type=int, help='batch size')
    parser.add_argument("--LearningRate", dest='learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument("--Optimizer", dest='optimizer', default='SGD', type=str,
                        help='choose the optimizer, SGD or Adam')
    parser.add_argument("--WeightDecay", dest='weight_decay', default=0.0005, type=float, help='learning rate decay')   # learning rate decay
    parser.add_argument("--DeviceId", dest='device_id', default='0', type=str, help='choose the GPU id to use')
    parser.add_argument("--Comment", dest='comment', type=str, help='the result file name consisted by the hyperparameters')
    parser.add_argument("--Pretrain", dest='pretrain', action="store_true", help='whether use the pretrained model')
    parser.add_argument("--Save_path", dest='save_path', default='../../lympho_checkpoints', help='the path to save the training result')
    parser.add_argument("--Resume", dest='resume', type = int, default = 0, help='resume training process')
    parser.add_argument("--Task", dest='task', type=str, default = 'lym', help='the task of training')
    parser.add_argument("--Stat", dest='stat', type=str, default = 'train', help='the task of training')
    
    args = parser.parse_args()
    utils.seed_torch(args.seed)
    loss_func = args.loss_func
    print('os.environ:', os.environ['CUDA_VISIBLE_DEVICES'])
    print('args.device_id:', args.device_id)
    now = time.strftime("%Y_%m_%d_", time.localtime())

    mp.set_start_method('spawn')

    global checkpoints
    checkpoints = os.path.join(CONFIG.CHECK_POINT, 
        now+"BS"+str(args.batch_size)+","+args.model+','+args.comment +
                               ',epochs'+str(args.epochs)+",seed"+str(args.seed)+",fold"+str(args.foldn)+'_%s' % args.task)
    print('Summary write in %s' % checkpoints)
    
    Writer = SummaryWriter(log_dir=checkpoints)
    
    train_loader, valid_loader, test_loader, weight_per_class = prepare_data(os.path.join(args.train_folder), args.task, args.batch_size, args.stat)


    print('num train images %d x %d' % (len(train_loader), args.batch_size))
    print('num val images %d x %d' % (len(valid_loader), args.batch_size))
    print("CUDA is_available:", torch.cuda.is_available())
    
    if args.device_id is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print('\n### Build model %s' % args.model)
    
    modellist = []
    if args.stat == 'test':
        ptpathlist = ['/fold1_tls.pt',
                      '/fold2_tls.pt',
                      '/fold3_tls.pt',
                      '/fold4_tls.pt',
                      '/fold5_tls.pt'
                      ]

        for ptpath in ptpathlist:
            model = create_model(args.model, args.pretrain)
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
            modellist.append(model)
            
    else:
        model = create_model(args.model, args.pretrain)
        model = model.to(device)
        modellist.append(model)
    print('torch.cuda.device_count:', torch.cuda.device_count())
    if torch.cuda.device_count() == 2:
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    elif torch.cuda.device_count() == 4:
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    if loss_func == 'cross':
        weight_per_class = torch.Tensor(weight_per_class).to(device)
        criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'SGD':
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], momentum=0.9, lr=args.learning_rate, weight_decay=args.weight_decay,
                              nesterov=True)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=args.weight_decay, amsgrad=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)
    
    resume_checkpoints = os.path.join(CONFIG.CHECK_POINT, 
        "BS"+str(args.batch_size)+","+args.model+','+args.comment +
                               ',epochs'+str(args.epochs)+",seed"+str(args.seed)+",fold"+str(args.foldn))

    if args.resume != 0:
        resume_point = os.path.join(resume_checkpoints, 'checkpoint_resume_%s' % str(args.resume))
        if os.path.isfile(resume_point):
            checkpoint = torch.load(resume_point)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    start_train(train_loader, valid_loader, test_loader, modellist, device, criterion,
                optimizer, scheduler, args.epochs, args.resume, args.stat)

