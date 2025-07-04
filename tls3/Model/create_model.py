import torch.nn as nn
from Model.resnet import resnet18, resnet34, resnet50,resnet101
from Model.vgg import vgg16_bn, vgg19_bn
from efficientnet_pytorch import EfficientNet
from Model.DenseNet import densenet121
from Model.vit_model import vit_base_patch16_224_in21k as vit_base_patch16_224
import pretrainedmodels
import torch


def create_model(model, pretrain, freeze=1):
    if model == 'vgg16_bn':
        Model = vgg16_bn(pretrained=pretrain)
    if model == 'vgg19_bn':
        Model = vgg19_bn(pretrained=pretrain)
    if model == 'resnet18':
        Model = resnet18(pretrained=pretrain)
    if model == 'resnet34':
        Model = resnet34(pretrained=pretrain)
    if model == 'resnet50':
        Model = resnet50(pretrained=pretrain)
    if model == 'resnet101':
        Model = resnet101(pretrained=pretrain)
    if model == 'densenet121':
        Model = densenet121(pretrained=pretrain)
    if model == 'efficientnet-b0':
        Model = EfficientNet.from_pretrained(model_name='efficientnet-b0', num_classes=3)
    if model == 'efficientnet-b1':
        Model = EfficientNet.from_pretrained(model_name='efficientnet-b1', num_classes=3)
    if model == 'efficientnet-b5':                    #  输入图像的尺寸是456*456
        # weights_path="../pretrained/efficientnet_b5_lukemelas-b6417697.pth"
        Model = EfficientNet.from_pretrained(model_name='efficientnet-b5', num_classes=3)
    if model == 'se_resnet50':
        Model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')
        Model.last_linear = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )
    if model == 'vit_base_patch16_224':
        weights = '../pretrained/vit_base_patch16_224_in21k.pth'
        freeze_layers = freeze
        Model = vit_base_patch16_224(num_classes=3, has_logits=False)
        if pretrain:
            weights_dict = torch.load(weights)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if Model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(Model.load_state_dict(weights_dict, strict=False))
        if freeze_layers:
            for name, para in Model.named_parameters():
                # 除head, pre_logits外，其他权重全部冻结
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    print(model)
    return Model



# https://github.com/Lornatang/pytorch-inception-v3-cifar100
