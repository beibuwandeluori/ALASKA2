import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import *
import time
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score,  roc_curve, auc
from network.model import model_selection, get_efficientnet, RGB_YCrCb_EfficientNet, Ensamble_EfficientNet
from dataset.dataset import AlaskaDataset, shuffle_two_tensor
from network.loss import LabelSmoothing
from catalyst.data.sampler import BalanceClassSampler

def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2,   1]

    fpr, tpr, thresholds = roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        # pdb.set_trace()

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization

def eval_model(model, epoch, eval_loader):
    model.eval()
    n_correct = 0
    labels = np.array([])
    preds = np.array([])
    outputs = np.array([])
    with torch.no_grad():
        for i, (img, img_YCrCb, label) in enumerate(eval_loader):
            img, img_YCrCb, label = Variable(img.cuda(device_id)), Variable(img_YCrCb.cuda(device_id)), Variable(label.cuda(device_id))
            class_output = model(img, img_YCrCb)
            class_output = nn.Softmax(dim=1)(class_output)

            pred = torch.max(class_output, 1)
            n_correct += (pred[1] == label).sum().item()

            labels = np.concatenate([labels, label.data.cpu().numpy()], axis=0)
            preds = np.concatenate([preds, pred[1].data.cpu().numpy()], axis=0)
            outputs = np.concatenate([outputs, 1-class_output.data.cpu().numpy()[:, 0]], axis=0)

            if i % 100 == 0:
                print(i+1, 'current_correct:', n_correct, 'current_len:', (i+1) * test_batch_size, len(eval_loader.dataset))

    # Change four classed to two classes
    for i in range(len(labels)):
        if labels[i] != 0:
            labels[i] = 1
        if preds[i] != 0:
            preds[i] = 1

    # print(labels[:4], preds[:4], outputs[:4])
    loss = log_loss(labels, outputs)  # .clip(0.25, 0.75)
    acc = accuracy_score(labels, preds)
    AUC = roc_auc_score(labels, outputs)
    weight_auc = alaska_weighted_auc(labels, outputs)
    print('loss:', loss, 'acc:', acc, 'AUC:', AUC, 'weigh_auc:', weight_auc)

    accu = float(n_correct) / len(eval_loader.dataset) * 100
    print('Epoch: [{}/{}], four classes accuracy : {:.4f}'.format(epoch, num_epochs, accu))

    with open(writeFile, 'a') as outF:
        outF.write('Epoch: [{}/{}], loss : {:.4f},  acc_m : {:.4f}, acc_b : {:.4f}, auc : {:.4f}, weight_auc : {:.4f}\n'
                   .format(epoch, num_epochs, loss, accu, acc, AUC, weight_auc))

    return weight_auc

def train_model(model, criterion, optimizer, epoch, is_shuffle=False):

    loss_aver = []
    model.train(True)
    start = time.time()
    for i, (XI, XI_YCrCb, label) in enumerate(train_loader):
        if is_shuffle:
            XI, label = shuffle_two_tensor(XI, label)  # 同时打乱tensor
        x = Variable(XI.cuda(device_id))
        x_YCrCb = Variable(XI_YCrCb.cuda(device_id))
        # label = Variable(torch.LongTensor(label).cuda(device_id))
        label = Variable(label.cuda(device_id))

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x, x_YCrCb)
        # Compute and print loss
        loss = criterion(y_pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_aver.append(loss.item())
        if i % 500 == 1:
            print('%s train %s images, use %s seconds, loss %s\n' % (i, i * batch_size, time.time() - start, sum(loss_aver) / len(loss_aver) if len(loss_aver) > 0 else 'NoLoss'))
        if i % 1000 == 1:
            with open(writeFile, 'a') as outF:
                outF.write('%s train %s images, use %s seconds, loss %s\n' % (i, i*batch_size, time.time() - start, sum(loss_aver) / len(loss_aver) if len(loss_aver) > 0 else 'NoLoss'))
    print('%s %s %s\n' % (epoch, sum(loss_aver) / len(loss_aver), time.time()-start))
    # torch.save(model.state_dict(), store_name + str(epoch))
    scheduler.step()


if __name__ == '__main__':

    batch_size = 64
    test_batch_size = 128
    device_id = 7
    lr = 0.00002
    is_RGB_YCrCb = True

    model_name = 'efficientnet-b4'
    writeFile = 'output/logs/efficientnet_RGB_YCrCb/' + model_name + '_YCrCb_4c.out'  # DCT_YCrCb
    store_name = 'output/weights/efficientnet_RGB_YCrCb/' + model_name + '_YCrCb_4c.pth'
    model_path = None

    # Load model
    model = RGB_YCrCb_EfficientNet(model_name1='efficientnet-b2', model_name2='efficientnet-b2', num_classes=4,
                                   model_path1='/data1/cby/py_project/ALASKA2/output/weights/efficientnet/efficientnet-b2_4c.pth64',
                                   model_path2='/data1/cby/py_project/ALASKA2/output/weights/efficientnet_YCrCb/efficientnet-b2_YCrCb_4c.pth51')
    if model_path is not None:
        # model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    model = model.cuda(device_id)

    is_train = True
    if is_train:
        # criterion = nn.CrossEntropyLoss()
        criterion = LabelSmoothing().cuda(device_id)
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        xdl = AlaskaDataset(data_type='train', classes_num=4, is_RGB_YCrCb=is_RGB_YCrCb, is_one_hot=True)
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=False, num_workers=4,
                                  sampler=BalanceClassSampler(labels=xdl.get_labels(), mode="downsampling"))
        train_dataset_len = len(xdl)

        xdl_eval = AlaskaDataset(data_type='val', classes_num=4, is_RGB_YCrCb=is_RGB_YCrCb, is_one_hot=False)
        eval_loader = DataLoader(xdl_eval, batch_size=test_batch_size, shuffle=False, num_workers=4)
        eval_dataset_len = len(xdl_eval)
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)

        epoch_start = 0
        num_epochs = 50
        best_weight_auc = 0.5 if epoch_start == 0 else eval_model(model, epoch_start, eval_loader)
        for epoch in range(epoch_start, num_epochs):
            train_model(model, criterion, optimizer, epoch, is_shuffle=False)
            weight_auc = eval_model(model, epoch, eval_loader)
            if weight_auc > best_weight_auc:
                torch.save(model.state_dict(), store_name + str(epoch))
                best_weight_auc = weight_auc
            print('Current best_weight_auc:', best_weight_auc)


    else:
        start = time.time()
        epoch_start = 1
        num_epochs = 1
        xdl_test = AlaskaDataset(data_type='val', classes_num=4, is_YCrCb=False,
                                 is_albumentations=True, is_one_hot=False)
        test_loader = DataLoader(xdl_test, batch_size=test_batch_size, shuffle=False, num_workers=4)
        test_dataset_len = len(xdl_test)
        print('test_dataset_len:', test_dataset_len)
        eval_model(model, epoch_start, test_loader)
        print('Total time:', time.time() - start)







