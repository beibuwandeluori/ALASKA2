import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import *
import time
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, precision_score, average_precision_score, roc_auc_score, recall_score

from network.model import XuNet, SRNet
from dataset.dataset import AlaskaDataset, shuffle_two_tensor

def eval_model(model, epoch, eval_loader):
    model.eval()
    n_correct = 0
    labels = np.array([])
    preds = np.array([])
    outputs = np.array([])
    with torch.no_grad():
        for i, (img, label) in enumerate(eval_loader):
            img, label = Variable(img.cuda(device_id)), Variable(label.cuda(device_id))
            class_output = model(img)
            class_output = nn.Softmax(dim=1)(class_output)

            pred = torch.max(class_output, 1)
            n_correct += (pred[1] == label).sum().item()

            labels = np.concatenate([labels, label.data.cpu().numpy()], axis=0)
            preds = np.concatenate([preds, pred[1].data.cpu().numpy()], axis=0)
            outputs = np.concatenate([outputs, class_output.data.cpu().numpy()[:, 1]], axis=0)
            if i % 100 == 0:
                print(i+1, 'current_correct:', n_correct, 'current_len:', (i+1) * test_batch_size, len(eval_loader.dataset))

    # print(outputs.shape, outputs[:2], preds[:2])
    loss = log_loss(labels, outputs)  # .clip(0.25, 0.75)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    AP = average_precision_score(labels, outputs)
    AUC = roc_auc_score(labels, outputs)
    print('loss:', loss, 'acc:', acc, 'recall:', recall, 'precision:', precision, 'AP:', AP, 'AUC:', AUC)

    accu = float(n_correct) / len(eval_loader.dataset) * 100
    print('Epoch: [{}/{}], accuracy : {:.4f}%'.format(epoch, num_epochs, accu))
    return accu

def train_model(model, criterion, optimizer, epoch, is_shuffle=False):
    loss_aver = []
    model.train(True)
    start = time.time()
    for i, (XI, label) in enumerate(train_loader):

        if is_shuffle:
            XI, label = shuffle_two_tensor(XI, label)  # 同时打乱tensor

        x = Variable(XI.cuda(device_id))
        label = Variable(torch.LongTensor(label).cuda(device_id))

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
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


if __name__ == '__main__':

    batch_size = 16
    test_batch_size = 128
    device_id = 7
    model_name = 'SRNet'
    writeFile = 'output/logs/' + model_name + '.out'
    store_name = 'output/weights/' + model_name + '.pth'
    model_path = None


    # Load model
    # model, *_ = model_selection(modelname=model_name, num_out_classes=2, dropout=0.5)
    model = SRNet(num_classes=2)

    if model_path is not None:
        # model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    model = model.cuda(device_id)

    is_train = True
    if is_train:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # optimizer = optim.Adam(model.parameters(), lr=0.001)

        xdl = AlaskaDataset(data_type='train', is_shuffle=True)
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=False, num_workers=4)
        train_dataset_len = len(xdl)

        xdl_eval = AlaskaDataset(data_type='val')
        eval_loader = DataLoader(xdl_eval, batch_size=test_batch_size, shuffle=False, num_workers=4)
        eval_dataset_len = len(xdl_eval)
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)

        epoch_start = 0
        num_epochs = 100
        best_acc = 0.5 #eval_model(model, epoch_start, eval_loader)
        for epoch in range(epoch_start, num_epochs):
            train_model(model, criterion, optimizer, epoch, is_shuffle=True)
            acc = eval_model(model, epoch, eval_loader)
            if acc > best_acc:
                torch.save(model.state_dict(), store_name + str(epoch))
                best_acc = acc


    else:
        batch_size = 128
        start = time.time()
        epoch_start = 1
        num_epochs = 1
        xdl_test = AlaskaDataset(data_type='val')
        test_loader = DataLoader(xdl_test, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset_len = len(xdl_test)
        print('test_dataset_len:', test_dataset_len)
        eval_model(model, epoch_start, test_loader)
        print('Total time:', time.time() - start)







