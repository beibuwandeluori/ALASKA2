import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import time
from torch.utils.data import *
import cv2
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import glob
from random import random, choice
from scipy.ndimage.filters import gaussian_filter
import sys
sys.dont_write_bytecode = True  # 设置不生成pyc文件
sys.path.append('/data1/cby/py_project/ALASKA2/dataset')
from jpeg_utils import *

# 数据增强
from sklearn.model_selection import GroupKFold
from torchvision import transforms
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip
)
from catalyst.data.sampler import BalanceClassSampler

# ——————————————自定义数据增强——————————————————#
# 自定的数据增强
def data_augment(img, blur_prob=0.0, blur_sig=[0, 3], jpg_prob=0.5, jpg_qual=[75, 90, 95]):
    img = np.array(img)
    if random() < jpg_prob:
        qual = sample_discrete(jpg_qual)
        img = cv2_jpg(img, qual)
    # if random() < blur_prob:
    #     sig = sample_continuous(blur_sig)
    #     gaussian_blur(img, sig)

    return Image.fromarray(img)

# 连续的采样
def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

# 离散采样
def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

# 高斯模糊
def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

# Jpeg压缩
def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]

# ——————————————自定义数据增强——————————————————#
W, H = 512, 512
default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((W, H)),
        # transforms.Lambda(lambda img: data_augment(img)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0),  # , contrast=0.5, saturation=0.5, hue=0.5
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((W, H)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((W, H)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

albumentations_data_transforms = {
    'train': Compose([
        Resize(H, W),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        # ToFloat(max_value=255),
        ToTensorV2()
    ]),
    'val': Compose([
        Resize(H, W),
        # ToFloat(max_value=255),
        ToTensorV2()
    ]),
    'test': Compose([
        Resize(H, W),
        # ToFloat(max_value=255),
        ToTensorV2()
    ]),
}

# ——————————————制作数据集——————————————————#
# 同时打乱torch tensor
def shuffle_two_tensor(a, b):
    ids = torch.randperm(a.size(0))
    a = a[ids]
    b = b[ids]
    return a, b

# 同步对应打乱两个数组
def shuffle_two_array(a, b, seed=None):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(a)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(b)
    return a, b

# 同步对应打乱四个数组
def shuffle_four_array(a, b, c, d, seed=None):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(a)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(b)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(c)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(d)
    return a, b, c, d

def get_images_by_QF(QF, image_paths_npy='/data1/cby/py_project/ALASKA2/dataset/QF_npy/train_all_image_names.npy',
                     QFs_npy='/data1/cby/py_project/ALASKA2/dataset/QF_npy/train_all_QFs.npy'):
    image_paths = np.load(image_paths_npy)
    QFs = np.load(QFs_npy)
    # class_names = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
    QF_images = []
    for i in range(len(image_paths)):
        if QFs[i] == QF:
            QF_images.append(ROOT_PATH + image_paths[i])

    return QF_images

def get_images_by_class_names(images):
    class_images = {'Cover': [], 'JMiPOD': [], 'JUNIWARD': [], 'UERD': []}
    for img in images:
        class_name = img.split('/')[-2]
        class_images[class_name].append(img)

    return class_images['Cover'], class_images['JMiPOD'], class_images['JUNIWARD'], class_images['UERD']

def select_data_by_QF(QF, image_paths, labels,
                      image_paths_npy='/data1/cby/py_project/ALASKA2/dataset/QF_npy/train_all_image_names.npy',
                      QFs_npy='/data1/cby/py_project/ALASKA2/dataset/QF_npy/train_all_QFs.npy'):
    all_image_paths = np.load(image_paths_npy)
    all_QFs = np.load(QFs_npy)
    all_images_dict = {}
    for i in range(len(all_image_paths)):
        all_images_dict[ROOT_PATH + all_image_paths[i]] = all_QFs[i]

    image_paths_QF = []
    labels_QF = []
    for j in range(len(image_paths)):
        if all_images_dict[image_paths[j]] == QF:
            image_paths_QF.append(image_paths[j])
            labels_QF.append(labels[j])

    return image_paths_QF, labels_QF

ROOT_PATH = '/raid/chenby/alaska2/'
def load_data(Cover_path=ROOT_PATH + 'Cover', JMiPOD_path=ROOT_PATH + 'JMiPOD',
              JUNIWARD_path=ROOT_PATH + 'JUNIWARD', UERD_path=ROOT_PATH + 'UERD',
              is_shuffle=False, classes_num=2, use_one_cover_type=3, QF=-1):
    if QF == -1:
        # Cover
        Cover_images = np.array(sorted(glob.glob(Cover_path + '/*')))#[:10000]
        # Stego
        JMiPOD_images = np.array(sorted(glob.glob(JMiPOD_path + '/*')))#[:10000]
        JUNIWARD_images = np.array(sorted(glob.glob(JUNIWARD_path + '/*')))#[:10000]
        UERD_images = np.array(sorted(glob.glob(UERD_path + '/*')))#[:10000]
        # Test_images = np.array(os.listdir(Test_path))
    else:
        Cover_images, JMiPOD_images, JUNIWARD_images, UERD_images = get_images_by_class_names(images=get_images_by_QF(QF=QF))

    if classes_num == 2:
        if use_one_cover_type == 3:
            Cover_images = np.concatenate([Cover_images, Cover_images, Cover_images])  # UpSample
            Stego_images = np.concatenate([JMiPOD_images, JUNIWARD_images, UERD_images], axis=0)
        else:  # use_one_cover_type = 0, 1, 2
            covers = [JMiPOD_images, JUNIWARD_images, UERD_images]
            Stego_images = covers[use_one_cover_type]
        if is_shuffle:
            Cover_images, Stego_images = shuffle_two_array(a=Cover_images, b=Stego_images, seed=10)
        return Cover_images, Stego_images
    else:
        if is_shuffle:
            Cover_images, JMiPOD_images, JUNIWARD_images, UERD_images = shuffle_four_array(a=Cover_images,
                                                                                           b=JMiPOD_images,
                                                                                           c=JUNIWARD_images,
                                                                                           d=UERD_images, seed=10)

        return Cover_images, JMiPOD_images, JUNIWARD_images, UERD_images

def read_data_from_csv(root_path='/raid/chenby/alaska2/', data_type='train'):

    # ['Normal', 'JMiPOD_75', 'JMiPOD_90', 'JMiPOD_95', 'JUNIWARD_75', 'JUNIWARD_90',
    #  'JUNIWARD_95', 'UERD_75', 'UERD_90', 'UERD_95']

    if data_type == 'train':
        data_df = pd.read_csv('/data1/cby/py_project/ALASKA2/dataset/csv_files/alaska2_train_df.csv')
    else:
        data_df = pd.read_csv('/data1/cby/py_project/ALASKA2/dataset/csv_files/alaska2_val_df.csv')

    img_paths = list(data_df['ImageFileName'].apply(lambda x: os.path.join(root_path, x)))

    labels = list(data_df['Label'])
    # print(len(img_paths), len(labels))
    # print(img_paths[-5:], labels[-5:])

    return img_paths, labels

# ——————————————制作数据集——————————————————#
def read_image_yo_YCrCb(image_path, branches='YCrCb', data_type='train'):
    channel_slice = branch_to_slice(branches)
    tmp = jpeglib.jpeg(image_path, verbosity=0)
    image = decompress(tmp)[:, :, channel_slice]
    image = np.array(image, dtype=np.float32)

    if data_type == 'train':
        # ValueError: some of the strides of a given numpy array are negative.
        # This is currently not supported, but will be added in future releases.
        # np.rot90(image, rot, axes=[0, 1]) -> np.rot90(image, rot, axes=[0, 1]).copy()

        rot = random.randint(0, 3)
        if random.random() < 0.5:
            image = np.rot90(image, rot, axes=[0, 1]).copy()
        else:
            image = np.flip(np.rot90(image, rot, axes=[0, 1]), axis=1).copy()
    return image

class AlaskaDataset(Dataset):
    def __init__(self, data_type='train', is_shuffle=False, is_YCrCb=False, classes_num=2,
                 is_albumentations=False, is_one_hot=False, use_one_cover_type=3, QF=-1, is_RGB_YCrCb=False):
        self.classes_num = classes_num
        self.data_type = data_type
        self.is_YCrCb = is_YCrCb
        self.is_RGB_YCrCb = is_RGB_YCrCb
        self.is_albumentations = is_albumentations
        if is_albumentations or is_RGB_YCrCb:
            self.transform = albumentations_data_transforms[self.data_type]
        else:
            self.transform = default_data_transforms[self.data_type]

        self.img_paths = []
        self.labels = []

        if classes_num == 10:
            self.img_paths, self.labels = read_data_from_csv(data_type=self.data_type)
        else:
            if classes_num == 2:
                cover_paths, stego_paths = load_data(is_shuffle=is_shuffle, classes_num=classes_num,
                                                     use_one_cover_type=use_one_cover_type)
                for i in range(len(cover_paths)):
                    self.img_paths.append(cover_paths[i])
                    self.labels.append(0)
                    self.img_paths.append(stego_paths[i])
                    self.labels.append(1)
            else:
                Cover_paths, JMiPOD_paths, JUNIWARD_paths, UERD_paths = load_data(is_shuffle=is_shuffle, classes_num=classes_num)
                # print(Cover_paths[:2], JMiPOD_paths[:2], JUNIWARD_paths[:2], UERD_paths[:2])
                for i in range(len(Cover_paths)):
                    self.img_paths.append(Cover_paths[i])
                    self.labels.append(0)
                    self.img_paths.append(JMiPOD_paths[i])
                    self.labels.append(1)
                    self.img_paths.append(JUNIWARD_paths[i])
                    self.labels.append(2)
                    self.img_paths.append(UERD_paths[i])
                    self.labels.append(3)
            train_paths, valid_paths, train_labels, valid_labels = train_test_split(self.img_paths, self.labels, test_size=0.15, random_state=2020)
            if data_type == 'train':
                self.img_paths = train_paths
                self.labels = train_labels
            else:
                self.img_paths = valid_paths
                self.labels = valid_labels

            if QF != -1:
                self.img_paths, self.labels = select_data_by_QF(QF, image_paths=self.img_paths, labels=self.labels)

        # for one hot
        self.is_one_hot = is_one_hot


    def __len__(self):
        return len(self.img_paths)

    def get_labels(self):
        return list(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        if self.is_one_hot:
            label = one_hot(size=self.classes_num, target=label)

        img_name = self.img_paths[index]
        # image = cv2.imread(img_name)
        # Revert from BGR
        if self.is_RGB_YCrCb:
            image_YCrCb = read_image_yo_YCrCb(img_name, branches='YCrCb', data_type=self.data_type)
            image_YCrCb = torch.from_numpy(image_YCrCb).permute(2, 0, 1)

            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = image.astype(np.float32)
            image /= 255.0
            sample = {'image': image}
            sample = self.transform(**sample)
            image_RGB = sample['image']

            return image_RGB, image_YCrCb, label

        elif self.is_YCrCb:  # 有小数点
            image = read_image_yo_YCrCb(img_name, branches='YCrCb', data_type=self.data_type)
            image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.is_albumentations:
                image = image.astype(np.float32)
                image /= 255.0
                sample = {'image': image}
                sample = self.transform(**sample)
                image = sample['image']
            else:
                image = self.transform(Image.fromarray(image))

        return image, label

class AlaskaTestDataset(Dataset):
    def __init__(self, img_root_paths='/raid/chenby/alaska2/Test', data_type='test', is_YCrCb=False):
        self.data_type = data_type
        self.is_YCrCb = is_YCrCb
        self.transform = default_data_transforms[self.data_type]
        self.img_paths = np.array(sorted(glob.glob(img_root_paths + '/*')))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]

        # image = cv2.imread(img_name)
        # Revert from BGR
        if self.is_YCrCb:
            # image = cv2.imread(img_name)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            image = read_image_yo_YCrCb(img_name, branches='YCrCb', data_type=self.data_type)
            image = torch.from_numpy(image).permute(2, 0, 1)
            # print(image.shape)
        else:
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(Image.fromarray(image))

        return image
#__________________________for efficientnet__________________________________
def Mix_up(cover, stego, mix_up_type=1):
    if mix_up_type == 0:  # global mix up
        p = np.random.random()
        p = p if p > 0.5 else p + 0.5  # 0.5 < p < 1
        image = (1 - p) * cover + p * stego
    elif mix_up_type == 1:  # grid mix up
        image = np.zeros(shape=(512, 512, 3), dtype=np.float32)
        mix_p = np.random.randint(0, 2, size=(4,))
        image[0: 256, 0:256, :] = cover[0: 256, 0:256, :] if mix_p[0] == 0 else stego[0: 256, 0:256, :]
        image[256: 512, 0:256, :] = cover[256: 512, 0:256, :] if mix_p[1] == 0 else stego[256: 512, 0:256, :]
        image[0: 256, 256: 512, :] = cover[0: 256, 256: 512, :] if mix_p[2] == 0 else stego[0: 256, 256: 512, :]
        image[256: 512, 256: 512, :] = cover[256: 512, 256: 512, :] if mix_p[3] == 0 else stego[256: 512, 256: 512, :]
    else:  # channel mix up
        image = np.zeros(shape=(512, 512, 3), dtype=np.float32)
        mix_p = np.random.randint(0, 2, size=(3,))
        image[:, :, 0] = cover[:, :, 0] if mix_p[0] == 0 else stego[:, :, 0]
        image[:, :, 1] = cover[:, :, 1] if mix_p[1] == 0 else stego[:, :, 1]
        image[:, :, 2] = cover[:, :, 2] if mix_p[2] == 0 else stego[:, :, 2]
        pass
    return image

def get_train_transforms():
    return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return Compose([
            Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def one_hot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class DatasetRetriever(Dataset):

    def __init__(self, data_type='train', is_one_hot=True, transforms=None, classes_num=4, use_one_cover_type=3,
                 QF=-1, is_mix_up=False, is_rec=False):
        super().__init__()
        self.classes_num = classes_num
        self.data_type = data_type
        self.transforms = transforms
        self.is_one_hot = is_one_hot
        self.is_mix_up = is_mix_up
        self.is_rec = is_rec
        self.img_paths = []
        self.labels = []

        if classes_num == 10:
            self.img_paths, self.labels = read_data_from_csv(data_type=self.data_type)
        else:
            if classes_num == 2:
                cover_paths, stego_paths = load_data(is_shuffle=False, classes_num=classes_num,
                                                     use_one_cover_type=use_one_cover_type, QF=-1)
                for i in range(len(cover_paths)):
                    self.img_paths.append(cover_paths[i])
                    self.labels.append(0)
                    self.img_paths.append(stego_paths[i])
                    self.labels.append(1)
            else:
                Cover_paths, JMiPOD_paths, JUNIWARD_paths, UERD_paths = load_data(is_shuffle=False, classes_num=classes_num, QF=-1)
                for i in range(len(Cover_paths)):
                    self.img_paths.append(Cover_paths[i])
                    self.labels.append(0)
                    self.img_paths.append(JMiPOD_paths[i])
                    self.labels.append(1)
                    self.img_paths.append(JUNIWARD_paths[i])
                    self.labels.append(2)
                    self.img_paths.append(UERD_paths[i])
                    self.labels.append(3)
            # split dataset
            train_paths, valid_paths, train_labels, valid_labels = train_test_split(self.img_paths, self.labels,
                                                                                    test_size=0.15, random_state=2020)
            # print('train:', len(train_labels), 'test:', len(valid_labels))
            if data_type == 'train':
                self.img_paths = train_paths
                self.labels = train_labels
            else:
                self.img_paths = valid_paths
                self.labels = valid_labels
            if QF != -1:
                self.img_paths, self.labels = select_data_by_QF(QF, image_paths=self.img_paths, labels=self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        image_name = self.img_paths[index]
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # mix_up
        p = np.random.random()
        if self.data_type == 'train' and self.is_mix_up and label != 0 and p > 0.5:
            # print(image_name, 'mix_up before', image[0, 0, :])
            image = self.mix_up(image, image_name, p)
            # print(image_name, 'mix_up after', image[0, 0, :])


        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        if self.is_one_hot:
            label = one_hot(self.classes_num, label)

        if self.is_rec:
            rec_label = self.get_rec_label(image, image_name)

            return image, label, rec_label

        return image, label

    def __len__(self) -> int:
        return len(self.img_paths)

    def mix_up(self, image, image_name, p):
        # mix_up = (1-p) * real + p * stego, 0.5 < p < 1
        real_path = ROOT_PATH + 'Cover/' + image_name.split('/')[-1]
        real_image = cv2.imread(real_path, cv2.IMREAD_COLOR)
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = Mix_up(cover=real_image, stego=image, mix_up_type=1 if np.random.random() > 0.5 else 2)
        return image

    def get_rec_label(self, image, image_name):
        real_path = ROOT_PATH + 'Cover/' + image_name.split('/')[-1]
        real_image = cv2.imread(real_path, cv2.IMREAD_COLOR)
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        rec_label = image - real_image

        return rec_label

    def get_labels(self):
        return list(self.labels)

class DatasetSubmissionRetriever(Dataset):

    def __init__(self, img_root_paths='/raid/chenby/alaska2/Test', transforms=None):
        super().__init__()
        self.image_names = np.array(sorted(glob.glob(img_root_paths + '/*')))
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image

    def __len__(self) -> int:
        return len(self.image_names)



if __name__ == '__main__':
    # Cover_images, Stego_images = load_data(is_shuffle=True)
    # print(Cover_images.shape, Stego_images.shape)
    # print(Cover_images[:10])
    # print(Stego_images[:10])

    start = time.time()
    xdl = AlaskaDataset(data_type='train', is_shuffle=False, classes_num=4, is_RGB_YCrCb=True, is_one_hot=False)
    print('length:', len(xdl))
    train_loader = DataLoader(xdl, batch_size=16, shuffle=False, num_workers=4,
                              sampler=BalanceClassSampler(labels=xdl.get_labels(), mode="downsampling"))
    for i, (img, img_YCrCb, label) in enumerate(train_loader):
        print(i, img.shape, img_YCrCb.shape, label.shape)
        if i == 10:
            break
        # break

    end = time.time()
    print('end iterate')
    print('DataLoader total time: %fs' % (end - start))

    # read_data_from_csv(data_type='test')

    # start = time.time()
    # xdl = AlaskaTestDataset(data_type='test', is_YCrCb=True)
    # print('length:', len(xdl))
    # train_loader = DataLoader(xdl, batch_size=128, shuffle=False, num_workers=4)
    # for i, img in enumerate(train_loader):
    #     print(i, img.shape)
    #     if i == 10:
    #         break
    #     # break
    #
    # end = time.time()
    # print('end iterate')
    # print('DataLoader total time: %fs' % (end - start))

    # start = time.time()
    # xdl = DatasetRetriever(data_type='train', is_one_hot=False, transforms=get_train_transforms(), classes_num=4, QF=90, is_rec=True)
    # print('length:', len(xdl))
    # train_loader = DataLoader(xdl, batch_size=20, shuffle=False, num_workers=1,
    #                           sampler=BalanceClassSampler(labels=xdl.get_labels(), mode="downsampling"))
    # for i, (img, label, rec_label) in enumerate(train_loader):
    #     print(i, img.shape, label.size(), rec_label.shape)
    #     if i == 10:
    #         break
    #     # break
    #
    # end = time.time()
    # print('end iterate')
    # print('DataLoader total time: %fs' % (end - start))

    # image_paths_npy = './QF_npy/test_all_image_names.npy'
    # QFs_npy = './QF_npy/test_all_QFs.npy'
    # images = get_images_by_QF(QF=75, image_paths_npy=image_paths_npy, QFs_npy=QFs_npy)
    # print(len(images), print(images[:10]))
    # a, b, c, d = get_images_by_class_names(images)
    # print(len(a), len(b), len(c), len(d))

    pass
