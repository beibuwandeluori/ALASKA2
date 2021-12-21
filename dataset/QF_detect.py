import os
import numpy as np
import glob

def get_image_name(class_name, path):
    _, image_name = os.path.split(path)
    image_name = class_name + '/' + image_name
    return image_name

def QF_detect(class_name):
    image_paths = np.array(sorted(glob.glob(root_path + class_name + '/*')))
    QFs = []
    image_names = []
    i = 0
    for img_path in image_paths:
        command = "identify -verbose " + img_path + " | grep Quality"
        # print(command)
        # result = os.system(command)
        result = os.popen(command).read()
        QF = int(result[11:13])
        QFs.append(QF)
        image_name = get_image_name(class_name, img_path)
        image_names.append(image_name)
        if i % 1000 == 0:
            print(i, image_name, 'QF:', QF)
        i += 1

    return image_names, QFs

def analyze(QFs):
    QF75 = 0
    QF90 = 0
    QF95 = 0
    for QF in QFs:
        if QF == 75:
            QF75 += 1
        elif QF == 90:
            QF90 += 1
        else:
            QF95 += 1

    print('QF75:', QF75, 'QF90:', QF90, 'QF95:', QF95)


if __name__ == '__main__':
    root_path = '/pubdata/chenby/dataset/alaska2/'
    class_names = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
    # class_names = ['Test']
    all_image_names = []
    all_QFs = []
    for i in range(len(class_names)):
        image_names, QFs = QF_detect(class_name=class_names[i])
        all_image_names += image_names
        all_QFs += QFs
        print(class_names[i], len(image_names), len(QFs))
        analyze(QFs)


    print('All', len(all_image_names), len(all_QFs))

    image_names_save_path = './QF_npy/train_all_image_names.npy'
    QFs_save_path = './QF_npy/train_all_QFs.npy'
    # image_names_save_path = './QF_npy/test_all_image_names.npy'
    # QFs_save_path = './QF_npy/test_all_QFs.npy'
    # np.save(image_names_save_path, np.array(all_image_names))
    # np.save(QFs_save_path, np.array(all_QFs))
