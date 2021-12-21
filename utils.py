import numpy as np
import torch
import matplotlib.pyplot as plt  # plt 用于显示图片
import cv2

def multi2binary(y_pred, label):
    for i in range(2, y_pred.shape[1]):
        y_pred[:, 1] += y_pred[:, i]
    label[:, 1] = 1 - label[:, 0]
    return y_pred[:, :2], label[:, :2]

def show_img(num, image_name=None,image=None):
    if image is None:
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 4, num)
    plt.imshow(image)  # 显示图片

    return image

if __name__ == '__main__':
    # y_pred = torch.tensor([[0.1, 0.7, 0.2, 0.1], [0.6, 0.1, 0.2, 0.1]])
    # label = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0]])
    # print(y_pred, label)
    # y_pred, label = multi2binary(y_pred, label)
    # print(y_pred, label)
    plt.figure(figsize=(10, 10))
    image_0 = show_img(image_name='dataset/img_demo/Cover/00001.jpg', num=1)
    image_1 = show_img(image_name='dataset/img_demo/JMiPOD/00001.jpg', num=2)
    image_2 = show_img(image_name='dataset/img_demo/JUNWARD/00001.jpg', num=3)
    image_3 = show_img(image_name='dataset/img_demo/UERD/00001.jpg', num=4)
    show_img(image=image_0-image_0, num=5)
    show_img(image=image_1 - image_0, num=6)
    show_img(image=image_2 - image_0, num=7)
    show_img(image=image_3 - image_0, num=8)
    plt.show()


    pass