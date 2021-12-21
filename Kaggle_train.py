import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet
from dataset.dataset import AlaskaDataset
from torch.autograd import Variable

H = 512
W = 512

EPOCHS = 10
LR = 1e-4, 1e-3
BATCH_SIZE = 32
VAL_BATCH_SIZE = 128

MODEL_PATH = 'output/weights/efficientnet_model'

def GlobalAvgPooling(x):
    return x.mean(axis=-1).mean(axis=-1)

class ENSModel(nn.Module):
    def __init__(self):
        super(ENSModel, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.avgpool = GlobalAvgPooling
        # self.dense_output = nn.Linear(2560, 1)
        self.dense_output = nn.Linear(1280, 4)
        # self.efn = EfficientNet.from_pretrained('efficientnet-b0')
        self.efn = EfficientNet.from_name('efficientnet-b0')

    def forward(self, x):
        x = x.reshape(-1, 3, H, W)
        feat = self.efn.extract_features(x)
        return self.dense_output(self.avgpool(feat))  # self.sigmoid(self.dense_output(self.avgpool(feat)))

def bce(inp, targ):
    return nn.BCELoss()(inp, targ)

def detach(tensors):
    return [tensor.detach().cpu() for tensor in tensors]

def acc(inp, targ):
    targ_idx = targ.squeeze()
    inp_idx = torch.round(inp).squeeze()
    return (inp_idx == targ_idx).float().sum(axis=0)/len(inp_idx)

if __name__ == '__main__':
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    # create model
    network = ENSModel().to(device)
    network_path = None
    if network_path is not None:
        # model = torch.load(model_path)
        network.load_state_dict(torch.load(network_path, map_location='cpu'))
        print('Model found in {}'.format(network_path))
    else:
        print('No model found, initializing random model.')
    # print(network)
    # load data
    train_set = AlaskaDataset(data_type='train', is_shuffle=False, is_four_classes=True)
    val_set = AlaskaDataset(data_type='val', is_four_classes=True)
    print('train_dataset_len:', len(train_set), 'eval_dataset_len:', len(val_set))
    val_loader = DataLoader(val_set, batch_size=VAL_BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = Adam([{'params': network.efn.parameters(), 'lr': LR[0]},
                      {'params': network.dense_output.parameters(), 'lr': LR[1]}])

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                  patience=2, verbose=True, eps=1e-6)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(1, EPOCHS):
        start = time.time()
        # train
        batch = 1
        for train_batch in train_loader:
            train_img, train_targs = train_batch

            network = network.to(device)
            train_img = train_img.to(device)
            train_targs = train_targs.to(device=device)  # , dtype=torch.float

            network.train()
            train_preds = network.forward(train_img)
            train_loss = criterion(train_preds, train_targs) #bce(train_preds, train_targs)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()


            if batch % 500 == 0 or batch == 1 or batch == len(train_loader):
                print('batch/total:[', batch, '/', len(train_loader), '],train_loss:', train_loss.item(), 'current time:', time.time()-start)
            batch = batch + 1


        # val
        network.eval()
        val_preds, val_targs = [], []
        for val_batch in val_loader:
            img, targ = val_batch
            with torch.no_grad():
                img = img.to(device)
                network = network.to(device)
                pred = network.forward(img)
                val_preds.append(pred);
                val_targs.append(targ)

        val_preds = torch.cat(val_preds, axis=0)
        val_targs = torch.cat(val_targs, axis=0)

        val_targs = val_targs.to(device=device)  # , dtype=torch.float
        val_loss = criterion(val_preds, val_targs)  # bce(val_preds, val_targs)
        val_acc = acc(val_preds, val_targs)

        scheduler.step(val_loss)
        val_acc, val_loss = detach([val_acc, val_loss])
        print('epoch:', epoch, 'val_acc:', val_acc.item(), 'val_loss:', val_loss.item(), 'per epoch time:', time.time()-start)

        if best_acc < val_acc:
            best_acc = val_acc
            path = MODEL_PATH + "_{}.pt".format(epoch)
            torch.save(network.state_dict(), path)


