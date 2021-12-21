# -*- coding: utf-8 -*-
# @Time    : 2020/1/10 12:20
# @Author  : Mingxing Li
# @FileName: fusion.py
# @Software: PyCharm
import torch

class Test_time_agumentation(object):

    def __init__(self, is_rotation=True):
        self.is_rotation = is_rotation

    def __rotation(self, img):
        """
        clockwise rotation 90 180 270
        """
        img90 = img.rot90(-1, [2, 3]) # 1 逆时针； -1 顺时针
        img180 = img.rot90(-1, [2, 3]).rot90(-1, [2, 3])
        img270 = img.rot90(1, [2, 3])
        return [img90, img180, img270]

    def __inverse_rotation(self, img90, img180, img270):
        """
        anticlockwise rotation 90 180 270
        """
        img90 = img90.rot90(1, [2, 3]) # 1 逆时针； -1 顺时针
        img180 = img180.rot90(1, [2, 3]).rot90(1, [2, 3])
        img270 = img270.rot90(-1, [2, 3])
        return img90, img180, img270

    def __flip(self, img):
        """
        Flip vertically and horizontally
        """
        return [img.flip(2), img.flip(3)]

    def __inverse_flip(self, img_v, img_h):
        """
        Flip vertically and horizontally
        """
        return img_v.flip(2), img_h.flip(3)

    def tensor_rotation(self, img):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__rotation(img)

    def tensor_inverse_rotation(self, img_list):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_rotation(img_list[0], img_list[1], img_list[2])

    def tensor_flip(self, img):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__flip(img)

    def tensor_inverse_flip(self, img_list):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_flip(img_list[0], img_list[1])


if __name__ == "__main__":
    a = torch.tensor([[0, 1], [2, 3]]).unsqueeze(0).unsqueeze(0)
    print(a, a.size())
    tta = Test_time_agumentation()
    # a = tta.tensor_rotation(a)
    a = tta.tensor_flip(a)
    print(a)
    a = tta.tensor_inverse_flip(a)
    print(a)

