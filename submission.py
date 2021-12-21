import warnings
warnings.filterwarnings('ignore')
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
from tqdm import tqdm
from network.model import model_selection, get_efficientnet, REC_EfficientNet
from dataset.dataset import AlaskaTestDataset
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
    test_dataset = AlaskaTestDataset(is_YCrCb=is_YCrCb)
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
                    # flip
                    im = inputs.flip(3)
                    outputs2, _ = model_(im)
                    outputs = (0.25 * outputs + 0.25 * outputs2)
                    outputs3, _ = model_(inputs)
                    outputs = (outputs + 0.5 * outputs3)
                else:
                    outputs = model_(im)
                    # flip
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

if __name__ == '__main__':
    batch_size = 32
    num_workers = 4
    device_id = 5
    is_ensamble = False
    is_flip = False
    is_TTA = True
    is_YCrCb = True
    is_rec = False

    if not is_ensamble:
        # efficientnet-b0 + DCT
        # model_path = '../input/efficientnetb0-dctpth79/efficientnet-b0_DCT.pth79'
        # model, *_ = model_selection(modelname='efficientnet-b0', num_out_classes=2, is_DCT=True)

        # efficientnet-b0 + DCT + 4 classed
        #     model_path = '../input/efficientnetb0-dct-4cpth56/efficientnet-b0_DCT_4c.pth56'
        #     model, *_ = model_selection(modelname='efficientnet-b0', num_out_classes=4, is_DCT=True)

        # efficientnet-b0  + 10 classes
        #     model_path = '../input/efficientnetb0-10cpth40/efficientnet-b0_10c.pth40'
        #     model, *_ = model_selection(modelname='efficientnet-b0', num_out_classes=10, is_DCT=False)

        # resnet18 + DCT + 10 classes
        # model_path = '/data1/cby/py_project/ALASKA2/output/weights/resnet18_DCT_new_YCrCb_10c.pth41'
        # model, *_ = model_selection(modelname='resnet18', num_out_classes=10, is_DCT_new=True)

        # resnet18 + new DCT + 4 classes
        model_path = '/data1/cby/py_project/ALASKA2/output/weights/efficientnet_YCrCb/efficientnet-b4_YCrCb_4c.pth44'
        # model, *_ = model_selection(modelname='resnet18', num_out_classes=4, is_DCT_new=True)
        model = get_efficientnet(model_name='efficientnet-b4')

        # resnet50 + DCT + 10 classes
        # model_path = '/data1/cby/py_project/ALASKA2/output/weights/resnet50_DCT_new_YCrCb_10c.pth49'
        # model, *_ = model_selection(modelname='resnet50', num_out_classes=10, is_DCT_new=True)


        # model_path = 'output/weights/rec_efficientnet_YCrCb/efficientnet-b2_YCrCb_4c.pth25'
        # model = REC_EfficientNet(model_name='efficientnet-b2', num_classes=4)

        # load pretrained weight
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Load weight finished!')
        model = model.cuda(device_id)
        # print(model)

    else:
        # efficientnet-b0 + DCT + 4 classed

        model_path1 = '/data1/cby/py_project/ALASKA2/output/weights/efficientnet-b0_10c.pth40'
        model1, *_ = model_selection(modelname='efficientnet-b0', num_out_classes=10, is_DCT=False)
        # load pretrained weight
        model1.load_state_dict(torch.load(model_path1, map_location='cpu'))
        print('Load model1 weight finished!')
        model1 = model1.cuda(device_id)
        is_YCrCb_model_1 = False

        # resnet18 + DCT + 4 classes
        model_path2 = '/data1/cby/py_project/ALASKA2/output/weights/resnet18_DCT_YCrCb_4c.pth23'
        model2, *_ = model_selection(modelname='resnet18', num_out_classes=4, is_DCT_new=True)
        # load pretrained weight
        model2.load_state_dict(torch.load(model_path2, map_location='cpu'))
        print('Load model2 weight finished!')
        model2 = model2.cuda(device_id)
        is_YCrCb_model_2 = True

        # resnet50 + DCT + 10 classes
        model_path3 = '/data1/cby/py_project/ALASKA2/output/weights/resnet50_DCT_10c.pth16'
        model3, *_ = model_selection(modelname='resnet50', num_out_classes=10, is_DCT=True)
        # load pretrained weight
        model3.load_state_dict(torch.load(model_path3, map_location='cpu'))
        print('Load model3 weight finished!')
        model3 = model3.cuda(device_id)
        is_YCrCb_model_3 = False

    if not is_ensamble:
        print('Single model prediction!', 'is_YCrCb =', is_YCrCb)
        new_preds = predict(model_=model)
    else:
        print('Multi models ensamble prediction!', 'is_YCrCb_model_1 =', is_YCrCb_model_1, 'is_YCrCb_model_2 =',
              is_YCrCb_model_2, 'is_YCrCb_model_3 =', is_YCrCb_model_3)
        preds1 = predict(model_=model1, is_flip=is_flip, is_YCrCb=is_YCrCb_model_1)
        preds2 = predict(model_=model2, is_flip=is_flip, is_YCrCb=is_YCrCb_model_2)
        preds3 = predict(model_=model3, is_flip=is_flip, is_YCrCb=is_YCrCb_model_3)
        #     print(preds1.shape, preds2.shape, preds3.shape)
        #     new_preds = (preds2 + preds3) / 2
        new_preds = (preds1 + preds2 + preds3) / 3

    img_root_paths = '/raid/chenby/alaska2/Test'
    test_filenames = np.array(sorted(glob.glob(img_root_paths + '/*')))
    test_df = pd.DataFrame({'ImageFileName': list(test_filenames)}, columns=['ImageFileName'])
    test_df_sub = test_df
    test_df_sub['Id'] = test_df_sub['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])
    test_df_sub['Label'] = new_preds

    test_df_sub = test_df_sub.drop('ImageFileName', axis=1)
    save_csv_name = 'efficientnet-b4_YCrCb_4c_pth44_TTA.csv'
    # save_csv_name = 'rec_efficientnet-b2_YCrCb_4c_pth25.csv'
    test_df_sub.to_csv('output/submission/' + save_csv_name, index=False)
    print(test_df_sub.head())

