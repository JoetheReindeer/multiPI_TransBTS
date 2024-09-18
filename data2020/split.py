
import os
import shutil


trainImage_dir = "/home/multiPI_TransBTS/data2020/train_data/"
trainGT_dir = "/home/multiPI_TransBTS/data2020/train_gt/"
valImage_dir = "/home/multiPI_TransBTS/data2020/val_data/"
valGT_dir = "/home/multiPI_TransBTS/data2020/val_gt/"
testImage_dir = "/home/multiPI_TransBTS/data2020/test_data/"
testGT_dir = "/home/multiPI_TransBTS/data2020/test_gt/"

train_txt="/home/multiPI_TransBTS/data2020/train_list.txt"
val_txt="/home/multiPI_TransBTS/data2020/val_list.txt"
test_txt="/home/multiPI_TransBTS/data2020/test_list.txt"

with open(train_txt, "r") as train_file:
    train_list = [line.strip() for line in train_file.readlines()]

with open(val_txt, "r") as val_file:
    val_list = [line.strip() for line in val_file.readlines()]

with open(test_txt, "r") as test_file:
    test_list = [line.strip() for line in test_file.readlines()]


if not os.path.exists(valImage_dir):
    os.mkdir(valImage_dir)

if not os.path.exists(valGT_dir):
    os.mkdir(valGT_dir)

if not os.path.exists(testImage_dir):
    os.mkdir(testImage_dir)

if not os.path.exists(testGT_dir):
    os.mkdir(testGT_dir)

for image_name in val_list:
    image_name = image_name + ".npy"
    image_path = os.path.join(trainImage_dir, image_name)
    gt_path = os.path.join(trainGT_dir, image_name)

    val_image_dest = os.path.join(valImage_dir, image_name)
    val_gt_dest = os.path.join(valGT_dir, image_name)

    shutil.move(image_path, val_image_dest)
    shutil.move(gt_path, val_gt_dest)


for image_name in test_list:
    image_name = image_name + ".npy"
    image_path = os.path.join(trainImage_dir, image_name)
    gt_path = os.path.join(trainGT_dir, image_name)

    test_image_dest = os.path.join(testImage_dir, image_name)
    test_gt_dest = os.path.join(testGT_dir, image_name)
    
    shutil.move(image_path, test_image_dest)
    shutil.move(gt_path, test_gt_dest)





