from collections import Counter

import sys
import os
import shutil
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# 原始路径
image_original_path = "./dataset/lungData/images/"
label_original_path = "./dataset/lungData/labels/"

cur_path = os.getcwd()

# 训练集路径
train_image_path = os.path.join(cur_path, "dataset/images/train/")
train_label_path = os.path.join(cur_path, "dataset/labels/train/")

# 测试集路径
test_image_path = os.path.join(cur_path, "dataset/images/test/")
test_label_path = os.path.join(cur_path, "dataset/labels/test/")

# 训练集目录
list_train = os.path.join(cur_path, "dataset/train.txt")
list_test = os.path.join(cur_path, "dataset/test.txt")

train_percent = 0.9
test_percent = 0.1

def del_file(path):
    for i in os.listdir(path):
        file_data = os.path.join(path, i)  # 使用 os.path.join 进行路径拼接
        os.remove(file_data)

def mkdir():
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    else:
        del_file(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)
    else:
        del_file(train_label_path)

    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    else:
        del_file(test_image_path)
    if not os.path.exists(test_label_path):
        os.makedirs(test_label_path)
    else:
        del_file(test_label_path)

def clearfile():
    if os.path.exists(list_train):
        os.remove(list_train)
    if os.path.exists(list_test):
        os.remove(list_test)

def cut():
    mkdir()
    clearfile()

    file_train = open(list_train, 'w')
    file_test = open(list_test, 'w')

    total_txt = os.listdir(label_original_path)
    num_txt = len(total_txt)
    list_all_txt = range(num_txt)

    num_train = int(num_txt * train_percent)

    random.shuffle(total_txt)  # 打乱文件名列表
    train = random.sample(list_all_txt, num_train)
    test = [i for i in list_all_txt if i not in train]

    for i in list_all_txt:
        name = total_txt[i][:-4]

        srcImage = os.path.join(image_original_path, name + '.jpg')
        srcLabel = os.path.join(label_original_path, name + ".txt")

        if i in train:
            # 训练集目标路径 train_image_path
            dst_train_Image = os.path.join(train_image_path, name + '.jpg')
            dst_train_Label = os.path.join(train_label_path, name + '.txt')
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
            file_train.write(dst_train_Image + '\n')
        else:
            # 测试集目标路径 test_image_path
            dst_test_Image = os.path.join(test_image_path, name + '.jpg')
            dst_test_Label = os.path.join(test_label_path, name + '.txt')
            shutil.copyfile(srcImage, dst_test_Image)
            shutil.copyfile(srcLabel, dst_test_Label)
            file_test.write(dst_test_Image + '\n')

    file_train.close()
    file_test.close()

cut()