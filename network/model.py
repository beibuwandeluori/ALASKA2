import torch
import pretrainedmodels
import torch.nn as nn
from torch.nn import init
import torchvision
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np
import math
from resnest.torch import resnest50

# fc layer weight init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

# 当in_channel != 3 时，初始化模型的第一个Conv的weight， 把之前的通道copy input_chaneel/3 次
def init_imagenet_weight(_conv_stem_weight, input_channel=3):
    for i in range(input_channel//3):
        if i == 0:
            _conv_stem_weight_new = _conv_stem_weight
        else:
            _conv_stem_weight_new = torch.cat([_conv_stem_weight_new, _conv_stem_weight], axis=1)

    return torch.nn.Parameter(_conv_stem_weight_new)

class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0, is_DCT=False, is_DCT_new=False, is_SRM=False):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        self.is_DCT = is_DCT
        self.is_DCT_new = is_DCT_new
        self.is_SRM = is_SRM
        self.input_channel = 3
        if self.is_DCT_new:
            self.DCT = DCT_Layer_new()
            self.input_channel = 48
        if self.is_DCT:
            self.DCT = DCT_Layer()
        if self.is_SRM:
            self.SRM = SRM_Layer()
            self.input_channel = 90
        if modelchoice == 'resnet50' or modelchoice == 'resnet18' or modelchoice == 'resnet101' or modelchoice == 'resnet152'\
                or modelchoice == 'resnext101_32x8d' or modelchoice == 'resnext50_32x4d':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=False)
                self.model.load_state_dict(
                    torch.load(
                        '/data1/cby/py_project/ALASKA2/network/pretrained_model/resnet50-19c8e357.pth')
                )
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=False)
                self.model.load_state_dict(
                    torch.load(
                        '/data1/cby/py_project/FaceForensics/classification/faceforensics++_models_subset/pretrain_model/resnet18-5c106cde.pth')
                )
            if modelchoice == 'resnet101':
                self.model = torchvision.models.resnet101(pretrained=False)
                self.model.load_state_dict(
                    torch.load('/data1/cby/py_project/FaceForensics/classification/faceforensics++_models_subset/pretrain_model/resnet101-5d3b4d8f.pth')
                )
            if modelchoice == 'resnet152':
                self.model = torchvision.models.resnet152(pretrained=False)
                self.model.load_state_dict(
                    torch.load(
                        '/data1/cby/py_project/FaceForensics/classification/faceforensics++_models_subset/pretrain_model/resnet152-b121ed2d.pth')
                )
            if modelchoice == 'resnext101_32x8d':
                self.model = torchvision.models.resnext101_32x8d(pretrained=False)
                self.model.load_state_dict(
                    torch.load(
                        '/data1/cby/py_project/FaceForensics/classification/faceforensics++_models_subset/pretrain_model/resnext101_32x8d-8ba56ff5.pth')
                )
            if modelchoice == 'resnext50_32x4d':
                self.model = torchvision.models.resnext50_32x4d(pretrained=False)
                self.model.load_state_dict(
                    torch.load(
                        '/data1/cby/py_project/FaceForensics/classification/faceforensics++_models_subset/pretrain_model/resnext50_32x4d-7cdf4587.pth')
                )

            # replace first Conv2d
            if self.input_channel != 3:
                conv1_weight = self.model.conv1.weight
                self.model.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.model.conv1.weight = init_imagenet_weight(conv1_weight, self.input_channel)

            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
                init.normal_(self.model.fc.weight.data, std=0.001)
                init.constant_(self.model.fc.bias.data, 0.0)
            else:
                self.model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 256),
                    nn.Dropout(p=dropout),
                    nn.Linear(256, num_out_classes)
                )
                init.normal_(self.model.fc[2].weight.data, std=0.001)
                init.constant_(self.model.fc[2].bias.data, 0.0)
        elif modelchoice == 'resnest50':
            if modelchoice == 'resnest50':
                self.model = resnest50(pretrained=True)
                # model_path = '/data1/cby/py_project/ALASKA2/network/pretrained_model/resnest50-528c19ca.pth'
            # self.model.load_state_dict(torch.load(model_path))
            # replace first Conv2d
            self.model.conv1[0] = nn.Conv2d(self.input_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
                init.normal_(self.model.fc.weight.data, std=0.001)
                init.constant_(self.model.fc.bias.data, 0.0)
            else:
                self.model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 256),
                    nn.Dropout(p=dropout),
                    nn.Linear(256, num_out_classes)
                )
                init.normal_(self.model.fc[2].weight.data, std=0.001)
                init.constant_(self.model.fc[2].bias.data, 0.0)
        elif modelchoice == 'se_resnext101_32x4d' or modelchoice == 'se_resnext50_32x4d':
            if modelchoice == 'se_resnext101_32x4d':
                self.model = pretrainedmodels.se_resnext101_32x4d(pretrained=None)
                self.model.load_state_dict(
                    torch.load('/data1/cby/py_project/FaceForensics/classification/faceforensics++_models_subset/pretrain_model/se_resnext101_32x4d-3b2fe3d8.pth')
                )
            if modelchoice == 'se_resnext50_32x4d':
                self.model = pretrainedmodels.se_resnext50_32x4d(pretrained=None)
                # self.model.load_state_dict(
                #     torch.load(
                #         '/data1/cby/py_project/FaceForensics/classification/faceforensics++_models_subset/pretrain_model/se_resnext101_32x4d-3b2fe3d8.pth')
                # )

            # replace first Conv2d
            self.model.layer0.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
                # init.normal_(self.model.last_linear.weight.data, std=0.001)
                # init.constant_(self.model.last_linear.bias.data, 0.0)
            else:
                print('Using dropout', dropout, num_ftrs)
                # self.model.last_linear = nn.Sequential(
                #     nn.Dropout(p=dropout),
                #     nn.Linear(num_ftrs, num_out_classes)
                # )
                self.model.last_linear = nn.Sequential(
                    nn.Linear(num_ftrs, 256),
                    nn.BatchNorm1d(256),
                    nn.Dropout(p=dropout),
                    nn.Linear(256, num_out_classes)
                )
                # weights init
                init.kaiming_normal_(self.model.last_linear[0].weight.data, a=0, mode='fan_out')
                init.constant_(self.model.last_linear[0].bias.data, 0.0)
                init.normal_(self.model.last_linear[1].weight.data, 1.0, 0.02)
                init.constant_(self.model.last_linear[1].bias.data, 0.0)
                init.normal_(self.model.last_linear[3].weight.data, std=0.001)
                init.constant_(self.model.last_linear[3].bias.data, 0.0)
        elif modelchoice == 'efficientnet-b7' or modelchoice == 'efficientnet-b6'\
                or modelchoice == 'efficientnet-b5' or modelchoice == 'efficientnet-b4'\
                or modelchoice == 'efficientnet-b3' or modelchoice == 'efficientnet-b2'\
                or modelchoice == 'efficientnet-b1' or modelchoice == 'efficientnet-b0':
            # self.model = EfficientNet.from_name(modelchoice, override_params={'num_classes': num_out_classes})
            self.model = get_efficientnet(model_name=modelchoice, num_classes=num_out_classes)
            if self.input_channel != 3:
                # print(self.input_channel)
                self.model._conv_stem.in_channels = self.input_channel
                self.model._conv_stem.weight = init_imagenet_weight(self.model._conv_stem.weight, self.input_channel)
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def forward(self, x):
        if self.is_DCT or self.is_DCT_new:
            x = self.DCT(x)
        if self.is_SRM:
            x = self.SRM(x)
        x = self.model(x)
        return x

def model_selection(modelname, num_out_classes, dropout=None, is_DCT=False, is_DCT_new=False, is_SRM=False):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    # torchvision
    if modelname == 'resnet18' or modelname == 'resnet50' or modelname == 'resnet101' or modelname == 'resnet152'\
            or modelname == 'resnext101_32x8d' or modelname == 'resnext50_32x4d':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes, is_DCT=is_DCT, is_DCT_new=is_DCT_new, is_SRM=is_SRM), \
               224, True, ['image'], None
    # ResNeSt
    elif modelname == 'resnest50':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes, is_DCT=is_DCT, is_DCT_new=is_DCT_new, is_SRM=is_SRM), \
               224, True, ['image'], None
    # pretrainedmodels
    elif modelname == 'se_resnext101_32x4d' or modelname == 'se_resnext50_32x4d':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes, is_DCT=is_DCT, is_DCT_new=is_DCT_new,is_SRM=is_SRM), \
               224, True, ['image'], None
    elif modelname == 'efficientnet-b7' or modelname == 'efficientnet-b6'\
            or modelname == 'efficientnet-b5' or modelname == 'efficientnet-b4' \
            or modelname == 'efficientnet-b3' or modelname == 'efficientnet-b2' \
            or modelname == 'efficientnet-b1' or modelname == 'efficientnet-b0':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes, is_DCT=is_DCT, is_DCT_new=is_DCT_new, is_SRM=is_SRM), \
               224, True, ['image'], None

    else:
        raise NotImplementedError(modelname)

# 自定义DCT Layer, 可以即插即用
class DCT_Layer(nn.Module):
    def __init__(self,):
        super(DCT_Layer, self).__init__()
        self.conv1 = nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2, bias=False)

    # DCT Keras
    def init_DCT(self, shape=(4, 4, 1, 16)):
        PI = math.pi
        DCT_kernel = np.zeros(shape, dtype=np.float32)  # [height,width,input,output], shape=(4, 4, 1, 16)
        u = np.ones([4], dtype=np.float32) * math.sqrt(2.0 / 4.0)
        u[0] = math.sqrt(1.0 / 4.0)
        for i in range(0, 4):
            for j in range(0, 4):
                for k in range(0, 4):
                    for l in range(0, 4):
                        DCT_kernel[i, j, :, k * 4 + l] = u[k] * u[l] * math.cos(PI / 8.0 * k * (2 * i + 1)) * math.cos(
                            PI / 8.0 * l * (2 * j + 1))
        DCT_kernel = DCT_kernel.transpose(3, 2, 0, 1)
        return torch.from_numpy(DCT_kernel)  # [output,,input, height,width] shape=(16, 1, 4, 4)
    # Trancation operation for DCT
    @staticmethod
    def DCT_Trunc(x):
        trunc = -(F.relu(-x + 8) - 8)
        return trunc

    def forward(self, x):
        DCT = self.init_DCT(shape=(4, 4, x.size(1), 16))
        DCT = torch.autograd.Variable(DCT, requires_grad=False)
        device = torch.device(x.data.device)
        out = F.conv2d(x, DCT.to(device), padding=2)
        out = self.DCT_Trunc(torch.abs(out))
        out = self.conv1(out)
        return out

# 自定义DCT Layer, 可以即插即用
class DCT_Layer_new(nn.Module):
    def __init__(self,):
        super(DCT_Layer_new, self).__init__()

    # DCT Keras
    def init_DCT(self, shape=(4, 4, 1, 16)):
        PI = math.pi
        DCT_kernel = np.zeros(shape, dtype=np.float32)  # [height,width,input,output], shape=(4, 4, 1, 16)
        u = np.ones([4], dtype=np.float32) * math.sqrt(2.0 / 4.0)
        u[0] = math.sqrt(1.0 / 4.0)
        for i in range(0, 4):
            for j in range(0, 4):
                for k in range(0, 4):
                    for l in range(0, 4):
                        DCT_kernel[i, j, :, k * 4 + l] = u[k] * u[l] * math.cos(PI / 8.0 * k * (2 * i + 1)) * math.cos(
                            PI / 8.0 * l * (2 * j + 1))
        DCT_kernel = DCT_kernel.transpose(3, 2, 0, 1)
        return torch.from_numpy(DCT_kernel)  # [output,,input, height,width] shape=(16, 1, 4, 4)

    # Trancation operation for DCT
    @staticmethod
    def DCT_Trunc(x):
        trunc = -(F.relu(-x + 8) - 8)
        return trunc

    def forward(self, x):

        DCT = self.init_DCT(shape=(4, 4, 1, 16))
        DCT = torch.autograd.Variable(DCT, requires_grad=False)
        device = torch.device(x.data.device)

        for i in range(x.size(1)):
            out = F.conv2d(x[:, i:i+1, :, :], DCT.to(device), padding=2)
            out = self.DCT_Trunc(torch.abs(out))
            if i == 0:
                outs = out
            else:
                outs = torch.cat([outs, out], dim=1)

        return outs

# Truncation operation
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)
        return output

# 自定义SRM Layer, 可以即插即用
class SRM_Layer(nn.Module):
    def __init__(self, TLU_threshold=3.0):
        super(SRM_Layer, self).__init__()
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = self.init_SRM()
        # Truncation, threshold = 3
        self.tlu = TLU(threshold=TLU_threshold)

    @staticmethod
    def init_SRM():
        srm_kernel = np.load('/data1/cby/py_project/ALASKA2/network/SRM_Kernels.npy')  # shape=(5, 5, 1, 30)
        srm_kernel = srm_kernel.transpose(3, 2, 0, 1)  # shape=(30, 1, 5, 5)
        hpf_weight = nn.Parameter(torch.Tensor(srm_kernel).view(30, 1, 5, 5), requires_grad=False)

        return hpf_weight

    def forward(self, x):
        for i in range(x.size(1)):
            out = self.hpf(x[:, i:i+1, :, :])
            out = self.tlu(out)
            if i == 0:
                outs = out
            else:
                outs = torch.cat([outs, out], dim=1)
        return outs

# SRNet
class SRNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SRNet, self).__init__()
        # Layer 1
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        # Layer 3
        self.layer31 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.layer32 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(16)
        # Layer 4
        self.layer41 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(16)
        self.layer42 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(16)
        # Layer 5
        self.layer51 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(16)
        self.layer52 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(16)
        # Layer 6
        self.layer61 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn61 = nn.BatchNorm2d(16)
        self.layer62 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(16)
        # Layer 7
        self.layer71 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(16)
        self.layer72 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(16)
        # Layer 8
        self.layer81 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(16)
        self.layer82 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn82 = nn.BatchNorm2d(16)
        self.layer83 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 9
        self.layer91 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.layer92 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(64)
        self.layer93 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 10
        self.layer101 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.layer102 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn102 = nn.BatchNorm2d(128)
        self.layer103 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn103 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 11
        self.layer111 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.layer112 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.layer113 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 12
        self.layer121 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn121 = nn.BatchNorm2d(512)
        self.layer122 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122 = nn.BatchNorm2d(512)
        # avgp = torch.mean() in forward before fc
        # Fully Connected layer
        self.fc = nn.Linear(512*1*1, num_classes)

    def forward(self, inputs):
        # Layer 1
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))
        # Layer 2
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))
        # Layer 3
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)
        # Layer 4
        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)
        # Layer 5
        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)
        # Layer 6
        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)
        # Layer 7
        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)
        # Layer 8
        convs = self.layer81(res)
        convs = self.bn81(convs)
        conv1 = self.layer82(res)
        actv1 = F.relu(self.bn82(conv1))
        conv2 = self.layer83(actv1)
        bn = self.bn83(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # Layer 9
        convs = self.layer91(res)
        convs = self.bn91(convs)
        conv1 = self.layer92(res)
        actv1 = F.relu(self.bn92(conv1))
        conv2 = self.layer93(actv1)
        bn = self.bn93(conv2)
        pool = self.pool2(bn)
        res = torch.add(convs, pool)
        # Layer 10
        convs = self.layer101(res)
        convs = self.bn101(convs)
        conv1 = self.layer102(res)
        actv1 = F.relu(self.bn102(conv1))
        conv2 = self.layer103(actv1)
        bn = self.bn103(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # Layer 11
        convs = self.layer111(res)
        convs = self.bn111(convs)
        conv1 = self.layer112(res)
        actv1 = F.relu(self.bn112(conv1))
        conv2 = self.layer113(actv1)
        bn = self.bn113(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # Layer 12
        conv1 = self.layer121(res)
        actv1 = F.relu(self.bn121(conv1))
        conv2 = self.layer122(actv1)
        bn = self.bn122(conv2)
        # print("L12:",res.shape)
        avgp = torch.mean(bn, dim=(2,3), keepdim=True)
        # fully connected
        flatten = avgp.view(avgp.size(0),-1)
        # print("flatten:", flatten.shape)
        fc = self.fc(flatten)
        # print("FC:",fc.shape)
        out = F.log_softmax(fc, dim=1)
        return fc

# For XuNet
class HPFLayer(nn.Module):
    def __init__(self):
        super(HPFLayer, self).__init__()
        self.KV = torch.tensor([[-1, 2, -2, 2, -1],
                           [2, -6, 8, -6, 2],
                           [-2, 8, -12, 8, -2],
                           [2, -6, 8, -6, 2],
                           [-1, 2, -2, 2, -1]]) / 12.
        self.KV = self.KV.view(1, 1, 5, 5)
        self.KV = torch.autograd.Variable(self.KV, requires_grad=False)
        # self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2, bias=False)

    def forward(self, x):
        device = torch.device(x.data.device)
        out = None
        for t in range(x.size(1)):
            out_t = F.conv2d(x[:, t:t+1, :, :], self.KV.to(device), padding=2)
            if out is None:
                out = out_t
            else:
                out = torch.cat([out, out_t], dim=1)
        # out = self.conv1(out)
        return out

class XuNet(nn.Module):
    def __init__(self, num_classes=2):
        super(XuNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128 * 1 * 1, num_classes)

        # KV = torch.tensor([[-1, 2, -2, 2, -1],
        #         #                    [2, -6, 8, -6, 2],
        #         #                    [-2, 8, -12, 8, -2],
        #         #                    [2, -6, 8, -6, 2],
        #         #                    [-1, 2, -2, 2, -1]]) / 12.
        #         # KV = KV.view(1, 1, 5, 5)  # .to(device=device, dtype=torch.float)
        #         # self.KV = torch.autograd.Variable(KV, requires_grad=False)

        self.HPF = HPFLayer()

    def forward(self, x):
        # device = torch.device(x.data.device)
        #
        # prep = F.conv2d(x, self.KV.to(device), padding=2)
        prep = self.HPF(x[:, 0:1, :, :])

        out = F.tanh(self.bn1(torch.abs(self.conv1(prep))))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.tanh(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.relu(self.bn3(self.conv3(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.relu(self.bn4(self.conv4(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.relu(self.bn5(self.conv5(out)))
        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


#___________________________________________________________
def get_efficientnet(model_name='efficientnet-b6', num_classes=4, model_path=None, original_num_classes=4):
    net = EfficientNet.from_pretrained(model_name)
    # net = EfficientNet.from_name(model_name)
    in_features = net._fc.in_features
    if model_path is not None:
        net._fc = nn.Linear(in_features=in_features, out_features=original_num_classes, bias=True)  # b4-b5=2048
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Model found in {}'.format(model_path))
    net._fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)  # b4-b5=2048

    return net

#___________________________________________________________
class DCT_EfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b2', num_classes=4, model_path=None):
        super(DCT_EfficientNet, self).__init__()
        self.DCT = DCT_Layer_new()
        self.input_channel = 48
        self.efficient = get_efficientnet(model_name=model_name, num_classes=num_classes)
        if model_path is not None:
            self.efficient.load_state_dict(torch.load(model_path, map_location='cpu'))
            print('Model found in {}'.format(model_path))
        else:
            print('No model found, initializing random model.')

        self.efficient._conv_stem.in_channels = self.input_channel
        self.efficient._conv_stem.weight = init_imagenet_weight(self.efficient._conv_stem.weight, self.input_channel)

    def forward(self, x):
        x = self.DCT(x)
        x = self.efficient(x)
        return x

class Noise_Layer(nn.Module):
    def __init__(self, TLU_threshold=3.0):
        super(Noise_Layer, self).__init__()
        self.hpf = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = self.init_SRM()
        # Truncation, threshold = 3
        self.tlu = TLU(threshold=TLU_threshold)

    @staticmethod
    def init_SRM():
        # srm_kernel = np.load('/data1/cby/py_project/ALASKA2/network/SRM_Kernels.npy')  # shape=(5, 5, 1, 30)
        srm_kernel = np.load('/pubdata/chenby/py_project/ALASKA2/network/SRM_Kernels.npy')  # shape=(5, 5, 1, 30)
        srm_kernel = srm_kernel[:, :, :, [10, 20, 25]]
        srm_kernel[:, :, :, 0] /= 2.0
        srm_kernel[:, :, :, 1] /= 4.0
        srm_kernel[:, :, :, 2] /= 12.0
        # print(srm_kernel, srm_kernel.shape)
        srm_kernel = srm_kernel.transpose(3, 2, 0, 1)  # shape=(3, 1, 5, 5)
        noise_kernel = np.concatenate([srm_kernel, srm_kernel, srm_kernel], axis=1)  # shape=(3, 3, 5, 5)
        hpf_weight = nn.Parameter(torch.Tensor(noise_kernel).view(3, 3, 5, 5), requires_grad=False)

        return hpf_weight

    def forward(self, x):
        out = self.hpf(x)
        out = self.tlu(out)

        return out

class RGB_N_EfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b2', num_classes=4, model_path=None):
        super(RGB_N_EfficientNet, self).__init__()
        self.Noise = Noise_Layer()
        # self.efficient_rgb = get_efficientnet(model_name=model_name, num_classes=num_classes)
        self.efficient_noise = get_efficientnet(model_name=model_name, num_classes=num_classes)
        if model_path is not None:
            # model = torch.load(model_path)
            # self.efficient_rgb.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.efficient_noise.load_state_dict(torch.load(model_path, map_location='cpu'))
            print('Model found in {}'.format(model_path))
        else:
            print('No model found, initializing random model.')
        in_features = self.efficient_noise._fc.in_features # * 2
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)  # b4-b5=2048

    def forward(self, x):
        noise = self.Noise(x)
        noise = self.efficient_noise.extract_features(noise)
        noise = torch.add(nn.AdaptiveMaxPool2d(1)(noise), nn.AdaptiveAvgPool2d(1)(noise))
        noise = noise.view(noise.size(0), -1)

        # rgb = self.efficient_rgb.extract_features(x)
        # rgb = torch.add(nn.AdaptiveMaxPool2d(1)(rgb), nn.AdaptiveAvgPool2d(1)(rgb))
        # rgb = rgb.view(rgb.size(0), -1)
        #
        # # print(noise.size(), rgb.size())
        # out = torch.cat([noise, rgb], dim=1)
        out = self.fc(noise)
        return out

class Ensamble_EfficientNet(nn.Module):
    def __init__(self, model_name1='efficientnet-b2', model_name2='efficientnet-b4', num_classes=4,
                 model_path1='/data1/cby/py_project/ALASKA2/output/weights/efficientnet/efficientnet-b2_4c.pth64',
                 model_path2='/data1/cby/py_project/ALASKA2/output/weights/efficientnet/efficientnet-b4_4c.pth73'):
        super(Ensamble_EfficientNet, self).__init__()
        self.efficient_1 = get_efficientnet(model_name=model_name1, num_classes=num_classes)
        self.efficient_2 = get_efficientnet(model_name=model_name2, num_classes=num_classes)
        if model_path1 is not None:
            # model = torch.load(model_path)
            self.efficient_1.load_state_dict(torch.load(model_path1, map_location='cpu'))
            print('{} found in {}'.format(model_name1, model_path1))
        else:
            print('No model found, initializing random model.')

        if model_path2 is not None:
            # model = torch.load(model_path)
            self.efficient_2.load_state_dict(torch.load(model_path2, map_location='cpu'))
            print('{} found in {}'.format(model_name2, model_path2))
        else:
            print('No model found, initializing random model.')

        in_features = self.efficient_1._fc.in_features + self.efficient_2._fc.in_features
        self.h = nn.Linear(in_features=in_features, out_features=128, bias=True)  # b4-b5=2048
        self.fc = nn.Linear(in_features=128, out_features=num_classes, bias=True)  # b4-b5=2048

    def forward(self, x):
        with torch.no_grad():
            x1 = self.efficient_1.extract_features(x)
            x2 = self.efficient_2.extract_features(x)

        x1 = torch.add(nn.AdaptiveMaxPool2d(1)(x1), nn.AdaptiveAvgPool2d(1)(x1))
        x1 = x1.view(x1.size(0), -1)

        x2 = torch.add(nn.AdaptiveMaxPool2d(1)(x2), nn.AdaptiveAvgPool2d(1)(x2))
        x2 = x2.view(x2.size(0), -1)

        out = torch.cat([x1, x2], dim=1)
        out = self.h(out)
        out = self.fc(out)
        return out

class RGB_YCrCb_EfficientNet(nn.Module):
    def __init__(self, model_name1='efficientnet-b2', model_name2='efficientnet-b2', num_classes=4,
                 model_path1='/data1/cby/py_project/ALASKA2/output/weights/efficientnet/efficientnet-b2_4c.pth64',
                 model_path2='/data1/cby/py_project/ALASKA2/output/weights/efficientnet_YCrCb/efficientnet-b2_YCrCb_4c.pth51'):
        super(RGB_YCrCb_EfficientNet, self).__init__()
        self.efficient_1 = get_efficientnet(model_name=model_name1, num_classes=num_classes)
        self.efficient_2 = get_efficientnet(model_name=model_name2, num_classes=num_classes)
        if model_path1 is not None:
            # model = torch.load(model_path)
            self.efficient_1.load_state_dict(torch.load(model_path1, map_location='cpu'))
            print('{} found in {}'.format(model_name1, model_path1))
        else:
            print('No model found, initializing random model.')

        if model_path2 is not None:
            # model = torch.load(model_path)
            self.efficient_2.load_state_dict(torch.load(model_path2, map_location='cpu'))
            print('{} found in {}'.format(model_name2, model_path2))
        else:
            print('No model found, initializing random model.')

        in_features = self.efficient_1._fc.in_features + self.efficient_2._fc.in_features
        self.h = nn.Linear(in_features=in_features, out_features=128, bias=True)  # b4-b5=2048
        self.fc = nn.Linear(in_features=128, out_features=num_classes, bias=True)  # b4-b5=2048

    def forward(self, x1, x2):
        with torch.no_grad():
            x1 = self.efficient_1.extract_features(x1)
            x2 = self.efficient_2.extract_features(x2)

        x1 = torch.add(nn.AdaptiveMaxPool2d(1)(x1), nn.AdaptiveAvgPool2d(1)(x1))
        x1 = x1.view(x1.size(0), -1)

        x2 = torch.add(nn.AdaptiveMaxPool2d(1)(x2), nn.AdaptiveAvgPool2d(1)(x2))
        x2 = x2.view(x2.size(0), -1)

        out = torch.cat([x1, x2], dim=1)
        out = self.h(out)
        out = self.fc(out)
        return out
#___________________________________________________________

class REC_EfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b2', num_classes=4, model_path=None):
        super(REC_EfficientNet, self).__init__()
        self.efficient = get_efficientnet(model_name=model_name, num_classes=num_classes)
        if model_path is not None:
            self.efficient.load_state_dict(torch.load(model_path, map_location='cpu'))
            print('Model found in {}'.format(model_path))
        else:
            print('No model found, initializing random model.')

        in_features = self.efficient._fc.in_features
        self.layer_u5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_features, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer_u4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            # torch.nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer_u3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            # torch.nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer_u2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            # torch.nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.layer_u1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            # nn.ReLU()
            nn.Tanh()
        )

        self.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)  # b4-b5=2048

    def forward(self, x):
        x = self.efficient.extract_features(x)
        reconstruct = self.layer_u5(x)
        reconstruct = self.layer_u4(reconstruct)
        reconstruct = self.layer_u3(reconstruct)
        reconstruct = self.layer_u2(reconstruct)
        reconstruct = self.layer_u1(reconstruct)

        clf = nn.AdaptiveMaxPool2d(1)(x)
        clf = clf.view(clf.size(0), -1)
        clf = self.fc(clf)
        return clf, reconstruct

if __name__ == '__main__':
    # model, image_size, *_ = model_selection('resnet18', num_out_classes=2, is_DCT=False, is_DCT_new=True, is_SRM=False)
    # model, image_size = SRNet(), 512
    # model, image_size = HPFLayer(), 512
    # model, image_size = XuNet(), 512
    model, image_size = DCT_Layer_new(), 224
    # model, image_size = SRM_Layer(), 512
    # model, image_size = ResNeSt(), 512
    # model, image_size = get_efficientnet(model_name='efficientnet-b0', num_classes=10,
    #                                      model_path=None,
    #                                      original_num_classes=4), 224
    # model, image_size = Noise_Layer(), 512
    # model, image_size = RGB_N_EfficientNet(model_path='/pubdata/chenby/py_project/ALASKA2/output/weights/efficientnet/efficientnet-b2_4c.pth49'), 512
    # model, image_size = RGB_N_ResNet(num_classes=4), 512
    # model, image_size = Ensamble_EfficientNet(num_classes=4), 512
    # model, image_size = RGB_YCrCb_EfficientNet(num_classes=4), 512
    # model, image_size = DCT_EfficientNet(model_path='../output/weights/efficientnet/efficientnet-b2_4c.pth49'), 512
    # model, image_size = REC_TwoStage_EfficientNet(model_path='../output/weights/efficientnet/efficientnet-b2_4c.pth49'), 512

    # print(model)
    model = model.to(torch.device('cpu'))
    from torchsummary import summary
    input_s = (3, image_size, image_size)
    # input_s = [(3, image_size, image_size), (3, image_size, image_size)]
    print(summary(model, input_s, device='cpu'))


    # srm_kernel = np.load('/data1/cby/py_project/ALASKA2/network/SRM_Kernels.npy')  # shape=(5, 5, 1, 30)
    # print(srm_kernel.shape)
    # for i in range(30):
    #     print('index:', i)
    #     print(srm_kernel[:, :, 0, i])

    pass

