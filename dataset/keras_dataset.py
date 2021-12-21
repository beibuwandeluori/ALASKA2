# -*- coding:UTF-8 -*-
from PIL import Image
import os
import numpy as np
from numpy import random
from numpy.random import rand
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import glob
import time


# 同步对应打乱两个数组
def shuffle_two_array(a, b, seed=None):
    state = np.random.get_state()
    if seed != None:
        np.random.seed(seed)
    np.random.shuffle(a)
    np.random.set_state(state)
    if seed != None:
        np.random.seed(seed)
    np.random.shuffle(b)
    return a, b

# 分割训练数据和验证数据
def split_data(cover_files, stego_files, val_size=0.1):
    n1 = int(len(cover_files)*(1-val_size))
    train_cover_files = cover_files[:n1]
    valid_cover_files = cover_files[n1:]
    n2 = int(len(stego_files) * (1 - val_size))
    train_stego_files = stego_files[:n2]
    valid_stego_files = stego_files[n2:]
    return train_cover_files, valid_cover_files, train_stego_files, valid_stego_files

# 加载文件
def load_game_data(cover_dir, stego_dir, train_set=True, val_size=0.1):
    # UpSample
    cover_files = sorted(glob.glob(os.path.join(cover_dir, '*'))) + \
                  sorted(glob.glob(os.path.join(cover_dir, '*'))) + \
                  sorted(glob.glob(os.path.join(cover_dir, '*')))
    stego_files = sorted(glob.glob(os.path.join(stego_dir, '*'))) + \
                  sorted(glob.glob(os.path.join('/raid/chenby/alaska2/JUNIWARD', '*'))) + \
                  sorted(glob.glob(os.path.join('/raid/chenby/alaska2/UERD', '*')))

    # 同步打乱数据
    cover_files, stego_files = shuffle_two_array(cover_files, stego_files, seed=10)

    train_cover_files, valid_cover_files, train_stego_files, valid_stego_files = \
        train_test_split(cover_files, stego_files, test_size=val_size, random_state=0)

    if train_set:
        train_images, train_labels = get_images_labels(train_cover_files, train_stego_files)
        print('train dataset length:', len(train_images))
        data = (train_images, train_labels)  # 训练集
    else:
        valid_images, valid_labels = get_images_labels(valid_cover_files, valid_stego_files)
        print('valid dataset length:', len(valid_images))
        data = (valid_images, valid_labels)  # 验证集
    return data

# 读取图片
def load_batch_image(img_path, channel_index=0):
    # param channel_index: r:0 g:1 b:2 rgb:3
    # img = load_img(img_path) # read rgb channel
    img = Image.open(img_path)
    # img = img.convert('YCbCr')  # 转换为YCbCr通道
    # print('YCbCr:'+str(np.array(img).shape))
    img = np.array(img)
    img = img / 255.0
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    if channel_index != 3:
        img = img[:, :, channel_index:channel_index+1] # - img[:, :, channel_index+1:channel_index+1]  # convertd image to numpy array, only read r channel
    return img

# 获取images_files和对应的labels
def get_images_labels(cover_files, stego_files):
    images_files = []
    labels = []
    # cover-stego交叉处理
    for i in range(len(cover_files)):
        # cover
        images_files.append(cover_files[i])
        labels.append(0)
        # stego
        images_files.append(stego_files[i])
        labels.append(1)
    images_files = np.array(images_files)
    labels = np.array(labels)
    return images_files, labels

# 建立一个数据迭代器
def my_dataset_generator(batch_size, cover_dir, stego_dir, channel_index=0, train_set=True):
    X_samples, Y_samples = load_game_data(cover_dir=cover_dir, stego_dir=stego_dir, train_set=train_set)

    batch_num = int(len(X_samples) / batch_size)
    max_len = batch_num * batch_size
    X_samples = np.array(X_samples[:max_len])
    Y_samples = np.array(Y_samples[:max_len])

    X_batches = np.split(X_samples, batch_num)
    Y_batches = np.split(Y_samples, batch_num)

    i = 0
    n = len(X_batches)
    while True:
        for b in range(len(X_batches)):
            # print('i % n = ' + str(i % n))
            i %= n
            X_batches[i], Y_batches[i] = shuffle_two_array(X_batches[i], Y_batches[i])  # 打乱每个batch数据
            X = np.array(list(map(load_batch_image,  X_batches[i], [channel_index] * len(X_batches[i])))) # 使用map进行调用函数
            # X = random_crop_image(X, crop_size=(512, 512))
            Y = np.array(Y_batches[i]) # 剪切图片
            # print(X.shape)
            i += 1

            if train_set:
                # 数据增强
                rot = random.randint(0, 3)
                # r = random.randint(0, 4)
                # if r < 1:
                X = np.rot90(X, rot, axes=[1, 2])
                # elif r < 2:
                # if random.random() < 0.5:
                #     X = np.flip(X, axis=2)
            yield X, keras.utils.to_categorical(Y)

if __name__ == '__main__':

    generator = my_dataset_generator(batch_size=64,
                                     cover_dir='/raid/chenby/alaska2/Cover',
                                     stego_dir='/raid/chenby/alaska2/JMiPOD',
                                     channel_index=3,
                                     train_set=True)
    print('start iterate')
    start = time.time()
    for i in range(100):
        x, y = next(generator)
        # print(x[0])
        print(str(i) + '-- x.shape:' + str(x.shape), ' y.shape:' + str(y.shape))
        print(y[:6])
    end = time.time()
    print('end iterate')
    print('my_generator iterate time: %fs' % ((end - start) / 100.))


    pass
