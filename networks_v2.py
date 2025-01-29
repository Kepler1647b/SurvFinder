# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config):
        super(DeepSurv, self).__init__()
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        # builds network
        self.model = self._build_network()
        # reloads model to device
        self.reload_model()


    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None: # adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            if self.norm: # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            # adds activation layer
            layers.append(eval('nn.{}()'.format(self.activation)))
        # builds sequential network
        return nn.Sequential(*layers)
    
    def reload_model(self):
        ''' Performs reloading model to device'''
        self.model = self.model.to(self.device_)


    def forward(self, X):
        return self.model(X)
    

class LateMMFAdv(nn.Module):
    def __init__(self, config, n_classes=2, dropout=0.25, ckpt=None, train_backbone=True):
        super(LateMMFAdv, self).__init__()

        # Pathomic
        self.patho_branch = PorpoiseAMIL(n_classes=n_classes, dropout=dropout, fusion=True)
        self.clinical_branch = DeepSurv(config['network2'])
        if ckpt is not None:
            print('Loading checkpoints...')
            mdl_params = torch.load(ckpt, map_location=torch.device('cuda'))
            self.patho_branch.load_state_dict(mdl_params, strict=True)
        
        # Fusion module
        self.fusion_patho_cat = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid(),
        )
        self.fusion_patho_single = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fusion_rad_cat = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid(),
        )
        self.fusion_rad_single = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fusion_attn = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )


        self.train_backbone = train_backbone
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self.patho_branch, "relocate"):
            self.patho_branch.relocate()
        else:
            self.patho_branch = self.patho_branch.to(torch.device('cuda'))
            
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.fusion_patho_cat = nn.DataParallel(self.fusion_patho_cat, device_ids=device_ids).to('cuda:0')
            self.fusion_patho_single = nn.DataParallel(self.fusion_patho_single, device_ids=device_ids).to('cuda:0')
            self.fusion_rad_cat = nn.DataParallel(self.fusion_rad_cat, device_ids=device_ids).to('cuda:0')
            self.fusion_rad_single = nn.DataParallel(self.fusion_rad_single, device_ids=device_ids).to('cuda:0')
            self.fusion_attn = nn.DataParallel(self.fusion_attn, device_ids=device_ids).to('cuda:0')
        else:
            self.fusion_patho_cat = self.fusion_patho_cat.to(torch.device('cuda'))
            self.fusion_patho_single = self.fusion_patho_single.to(torch.device('cuda'))
            self.fusion_rad_cat = self.fusion_rad_cat.to(torch.device('cuda'))
            self.fusion_rad_single = self.fusion_rad_single.to(torch.device('cuda'))
            self.fusion_attn = self.fusion_attn.to(torch.device('cuda'))
    
        self.classifier.to(device)
    
    def forward(self, data1, data2):
        x = data1
        rad_feats = data2
        # patho feats
        if self.train_backbone:
            h = self.patho_branch(x)
            rad_feats = self.clinical_branch(rad_feats)
        else:
            with torch.no_grad():
                h = self.patho_branch(x)

        # simple fusion
        concat_feats = torch.concat((h, rad_feats), dim=1)
        
        # patho sub-block
        patho_cat = self.fusion_patho_cat(concat_feats)
        patho_single = self.fusion_patho_single(h)
        patho_attn = self.fusion_attn(patho_cat * patho_single)
        # rad sub-block
        rad_cat = self.fusion_rad_cat(concat_feats)
        rad_single = self.fusion_rad_single(rad_feats)
        rad_attn = self.fusion_attn(rad_cat * rad_single)

        output = self.classifier(torch.cat((patho_attn, rad_attn), dim=1))

        return output
    
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class PorpoiseAMIL(nn.Module):
    def __init__(self, size_arg = "small", n_classes=2, dropout = 0.25, fusion=False):
        super(PorpoiseAMIL, self).__init__()
        size = [512, 384, 256, 64, 16]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        fc1 = [nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(0.25)]
        fc2 = [nn.Linear(size[2], size[3]), nn.ReLU(), nn.Dropout(0.25)]
        fc = fc+fc1+fc2
        attention_net = Attn_Net_Gated(L=size[3], D=size[4], dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.fusion = fusion
        
        self.classifier = nn.Linear(size[3], n_classes)
                
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, x):
        h = x
        A, h = self.attention_net(h)  
        A = torch.permute(A, (0, 2, 1))

        A = F.softmax(A, dim=2) 
        M = torch.matmul(A, h).squeeze(dim=1)
        if self.fusion == True:
            return M
        h  = self.classifier(M)
        return h


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
