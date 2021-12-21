import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import *
import time
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import glob

from torch.utils.data import DataLoader
from tqdm import tqdm
from network.model import get_efficientnet, REC_EfficientNet
from dataset.dataset import DatasetSubmissionRetriever, get_valid_transforms
from dataset.utils import Test_time_agumentation

# 12 times
def TTA(model_, img):
    # print('TTA')
    img = Variable(img.cuda(device_id))
    # 1
    outputs = model_(img)
    i = 1
    tta = Test_time_agumentation()
    rot_imgs = tta.tensor_rotation(img)
    # 3 img90, img180, img270
    for rot_img in rot_imgs:
        outputs += model_(rot_img)
        i += 1
    # 2 水平翻转 + 垂直翻转
    flip_imgs = tta.tensor_flip(img)
    for flip_img in flip_imgs:
        outputs += model_(flip_img)
        i += 1
    # 2*3=6
    for flip_img in flip_imgs:
        rot_flip_imgs = tta.tensor_rotation(flip_img)
        for rot_flip_img in rot_flip_imgs:
            outputs += model_(rot_flip_img)
            i += 1

    outputs = outputs/i

    return outputs

def predict(model_):
    test_dataset = DatasetSubmissionRetriever(transforms=get_valid_transforms())
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             drop_last=False)
    tk0 = tqdm(test_loader)
    model_.eval()
    preds = []
    with torch.no_grad():
        for i, im in enumerate(tk0):
            if is_TTA:
                outputs = TTA(model_, im)
                preds.extend(F.softmax(outputs, 1).cpu().numpy())
            elif not is_flip:
                # im = im.to(device)
                im = Variable(im.cuda(device_id))
                if is_rec:
                    outputs, _ = model_(im)
                else:
                    outputs = model_(im)
                outputs = F.softmax(outputs, 1).cpu().numpy()
                preds.extend(outputs)
            else:
                inputs = Variable(im.cuda(device_id))
                # flip vertical
                im = inputs.flip(2)
                if is_rec:
                    outputs, _ = model_(im)
                    # fliplr
                    im = inputs.flip(3)
                    outputs2, _ = model_(im)
                    outputs = (0.25 * outputs + 0.25 * outputs2)
                    outputs3, _ = model_(inputs)
                    outputs = (outputs + 0.5 * outputs3)
                else:
                    outputs = model_(im)
                    # fliplr
                    im = inputs.flip(3)
                    outputs = (0.25 * outputs + 0.25 * model_(im))
                    outputs = (outputs + 0.5 * model_(inputs))

                preds.extend(F.softmax(outputs, 1).cpu().numpy())

    preds = np.array(preds)
    labels = preds.argmax(1)
    new_preds = np.zeros((len(preds),))
    new_preds[labels != 0] = preds[labels != 0, 1:].sum(1)
    new_preds[labels == 0] = 1 - preds[labels == 0, 0]

    return new_preds

def ensamble_csv_by_QF(image_names_npy='/data1/cby/py_project/ALASKA2/dataset/QF_npy/test_all_image_names.npy',
                       QFs_npy='/data1/cby/py_project/ALASKA2/dataset/QF_npy/test_all_QFs.npy'):
    image_names = np.load(image_names_npy)
    QFs = np.load(QFs_npy)
    images_dict = {}
    for i in range(len(image_names)):
        images_dict[image_names[i].split('/')[-1]] = QFs[i]
    # QF = 75
    file_1 = pd.read_csv('/data1/cby/py_project/ALASKA2/output/submission/efficientnet/efficientnet-b4_4c_pth54_flip.csv')
    # QF = 95
    file_2 = pd.read_csv('/data1/cby/py_project/ALASKA2/output/submission/efficientnet/efficientnet-b4_4c_QF95_pth21_flip.csv')
    # QF = 90
    # file_3 = pd.read_csv('/data1/cby/py_project/ALASKA2/output/submission/efficientnet/efficientnet-b4_4c_pth73_flip.csv')
    Ids = file_1['Id']
    Labels_1 = file_1['Label']
    Labels_2 = file_2['Label']
    # Labels_3 = file_3['Label']
    # print(len(Ids), len(Labels_1), len(Labels_2), len(Labels_3), Labels_1[0], Labels_2[0], Labels_3[0])
    new_preds = []
    for i in range(len(Ids)):
        print(images_dict[Ids[i]])
        if images_dict[Ids[i]] == 95:
            new_preds.append(Labels_2[i])
            print(Labels_2[i])
        else:
            new_preds.append(Labels_1[i])
            print('1', Labels_1[i])


    file_1['Label'] = new_preds

    save_csv_name = 'efficientnet-b4_4c_pth54_21QF95_flip.csv'
    file_1.to_csv('output/submission/efficientnet/' + save_csv_name, index=False)

if __name__ == '__main__':
    batch_size = 32
    num_workers = 4
    device_id = 7
    is_ensamble = False
    is_flip = False
    is_rec = False
    is_TTA = True

    if not is_ensamble:
        # efficientnet-b2
        # model_path = '/data1/cby/py_project/ALASKA2/output/weights/efficientnet/efficientnet-b4_4c.pth73'
        model_path = '/data1/cby/py_project/ALASKA2/output/weights/efficientnet/efficientnet-b4_4c.pth54'
        model = get_efficientnet(model_name='efficientnet-b4', num_classes=4)
        # model_path = '/data1/cby/py_project/ALASKA2/output/weights/rec_efficientnet/efficientnet-b2_4c.pth45'
        # model = REC_EfficientNet(num_classes=4)

        # load pretrained weight
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Load weight finished!')
        model = model.cuda(device_id)
        # print(model)


    if not is_ensamble:
        print('Single model prediction!')
        new_preds = predict(model_=model)


    img_root_paths = '/raid/chenby/alaska2/Test'
    test_filenames = np.array(sorted(glob.glob(img_root_paths + '/*')))
    test_df = pd.DataFrame({'ImageFileName': list(test_filenames)}, columns=['ImageFileName'])
    test_df_sub = test_df
    test_df_sub['Id'] = test_df_sub['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])
    test_df_sub['Label'] = new_preds

    test_df_sub = test_df_sub.drop('ImageFileName', axis=1)
    save_csv_name = 'efficientnet-b4_4c_pth54_TTA.csv'
    # save_csv_name = 'efficientnet-b4_4c_QF95_pth21_flip.csv'
    test_df_sub.to_csv('output/submission/efficientnet/' + save_csv_name, index=False)
    print(test_df_sub.head())

    # ensamble csv by QF
    # ensamble_csv_by_QF()
    pass

