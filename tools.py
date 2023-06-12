import os
import numpy as np
from PIL import Image
from skimage import io
from skimage import measure, color
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter


def cal_mean_std():

    filepath = './train'  # 数据集目录
    pathDir = os.listdir(filepath)  # 数据集目录下图片
    num = len(pathDir)  

    print("Computing mean...")
    print("Computing var...")
    data_mean = [0]*3
    data_std = [0]*3
    for idx in range(len(pathDir)): #数据集中的每一张图片的索引
        filename = pathDir[idx] 
        img = Image.open(os.path.join(filepath, filename)) #数据集中的每张图片
        img = np.array(img) / 256.0 

        for i in range(3): # 取三维矩阵中三维的所有数据
            data_mean[i] += np.mean(img[i])  
            data_std[i] += np.std(img[i])

    data_mean = np.array(data_mean) / num
    data_std = np.array(data_std) / num
    print("mean:{}".format(data_mean))
    print("std:{}".format(data_std))

def cal_weightmap():

    gt = cv2.imread('F:/code/wanghai/minerals/train_label/11.png',0)
    gt = 1 * (gt > 100)

    c_weights = np.zeros(2)
    c_weights[0] = 1.0 / ((gt == 0).sum())
    c_weights[1] = 1.0 / ((gt == 1).sum())
    
    # 【2】归一化
    c_weights /= c_weights.max()
    # 【3】得到 class_weight map(cw_map)
    cw_map = np.where(gt==0, c_weights[0], c_weights[1])
    cells = measure.label(gt, connectivity=2)
    cells_color = color.label2rgb(cells , bg_label = 0,  bg_color = (0, 0, 0)) 
    w0 = 5
    sigma = 5
    dw_map = np.zeros_like(gt)
    maps = np.zeros((gt.shape[0], gt.shape[1], cells.max()))
    if cells.max() >= 2:
        for i in range(1, cells.max() + 1):
            maps[:,:,i-1] =  cv2.distanceTransform(1- (cells == i).astype(np.uint8), cv2.DIST_L2, 3)
        maps = np.sort(maps, axis = 2)
        d1 = maps[:,:,0]
        d2 = maps[:,:,1]
        dis = ((d1 + d2)**2) / (2 * sigma * sigma) * 0.1
        dw_map = 1+ w0 * np.exp(-dis) * (cells == 0) 
    plt.figure(figsize = (10,10))
    plt.imshow(dw_map, cmap = 'jet')
    plt.colorbar(fraction = 0.035)
    plt.show()


def cut_and_save(root):

    '''cut the img_2560*1920 and 1280*960 into  img_640*480,the sampling interval is 0'''

    root_files = os.listdir(root) #把地址中的文件转换成列表的形式
    print(root_files)
    count = 0
    for file in root_files: 
        img_path = os.path.join(root, file) #把地址中的每个文件的绝对地址输出来
        img = cv2.imread(img_path)

        x_shape = img.shape[0]
        y_shape = img.shape[1]
     
        for x in range(0, x_shape, 480):
            for y in range(0, y_shape, 640): 
                img_cut = img[x : x + 480, y : y + 640] #
                save_dir = f"C:/Users/wanghai/Desktop/new2/{file[:8]}_{count}.png"
                cv2.imwrite(save_dir, img_cut)
                count += 1

        if img.shape[0] == 1920:
            x, y = 0, 0
            for x in range(0, 1920, 480):
                for y in range(0, 2560, 640):
                    img_cut = img[x : x + 480, y : y + 640]
                    save_dir = f"F:/code/wanghai/minerals_seg/train_label_split/{file[:8]}_{count}.png"
                    cv2.imwrite(save_dir, img_cut)
                    count += 1

def merge1(root):
    root_files = os.listdir(root)
    merge_img = np.zeros(((1920,2560,3)))
    x, y = 0, 0
    idx = 0
    while idx < 16:
        for x in range(0, 1920, 480):
            for y in range(0, 2560, 640):
                img_path = os.path.join(root, root_files[idx])
                img = cv2.imread(img_path)
                merge_img[x: x + 480, y: y + 640, :] = img
                idx += 1

    save_dir = ("F:/code/wanghai/minerals_seg/merge_img/{}.png".format(root_files[0][:8]))
    cv2.imwrite(save_dir, merge_img)

import re
def merge2(root):
    root_files = os.listdir(root)
    merge_img = np.zeros(((10560, 11520,3)))

    x, y = 0, 0
    idx = 0
    while idx < (10560/480 * 11520/640):
        for x in range(0, 10560, 480):
            for y in range(0, 11520, 640):
                img_path = os.path.join(root, root_files[idx])
                img = cv2.imread(img_path)
                merge_img[x: x + 480, y: y + 640, :] = img
                idx += 1

    save_dir = ("C:/Users/wanghai/Desktop/new/{}.png".format(root_files[0][:8]))
    cv2.imwrite(save_dir, merge_img)

def fill_0(root):
    file_list = os.listdir(root)

    for file in file_list:
        if file.endswith('.png'):
            src = os.path.join(root, file)
            print(str(file[9:-4]))
            dst = os.path.join(root, file[9:-4].zfill(3) + '.png')
            print(dst)
            os.rename(src, dst)



def cal_area_size(root):
    img = cv2.imread(root)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray == 128)
    dst = np.array(mask * 255, dtype=np.uint8) #注意，格式一定要是np.uint8才能显示出图片

    contours, hierarchy = cv2.findContours(dst ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #轮廓检测函数
    cv2.drawContours(dst,contours,-1,(120,0,0),2)  #绘制轮廓
    count=0 #磁铁矿总数
    ares_avrg=0  #磁铁矿平均
    length_avrg = 0
    #遍历找到的所有磁铁矿
    for cont in contours:
        ares = cv2.contourArea(cont) #计算包围性状的面积
        length = cv2.arcLength(cont, closed=True)
        if ares<50:   #过滤面积小于10的形状
            continue
        count += 1    #总体计数加1
        ares_avrg += ares
        length_avrg += length
        print("{}-磁铁矿面积:{}".format(count,ares), end="  ") #打印出每个磁铁矿的面积
        print("磁铁矿周长:{}".format(round(length, 2)), end="  ")
        rect = cv2.boundingRect(cont) #提取矩形坐标,返回（x, y, w, h）表示的是外界矩形的左上角坐标和矩形的宽和高
        print("x:{} y:{}".format(rect[0],rect[1]))#打印坐标
        cv2.rectangle(dst, rect, (255,0,0), 1)#绘制矩形
        y = 10 if rect[1] < 10 else rect[1] #防止编号到图片之外
        cv2.putText(dst , str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 0), 1) #在磁铁矿左上角写上编号

    print("磁铁矿平均面积:{}".format(round(ares_avrg/count, 2))) #打印出每个磁铁矿的面积
    print("磁铁矿平均周长:{}".format(round(length_avrg/count, 2)))

    cv2.imshow("img2", dst)
    cv2.waitKey()
    cv2.destroyAllWindows

def cal_cls_number(root):
    file_list = os.listdir(root)
    count = [0] * 10
    for file in file_list:
        img_path = os.path.join(root, file)
        img_BGR = cv2.imread(img_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        r,g,b = cv2.split(img_RGB)
        r = np.array(r, np.uint16)
        g = np.array(g, np.uint16)
        b = np.array(b, np.uint16)
        sum_rgb = r * 1 + g*2 + b * 3
        sum_rgb = sum_rgb.flatten()
        res = np.bincount(sum_rgb, minlength=1600)
        val = [768, 891, 1061, 1275, 765, 1190, 976, 1530, 909, 0]
        for i in range(len(count)):
            count[i] += res[val[i]]
    print(count)
    idx = ['Mag', 'Ol', 'Px', 'Pn', 'Cp', 'Hbl', 'Bio', 'Pl', 'Po', 'bg']
    Acc = np.array([94.5, 93.2, 55.6, 56.3, 92.3, 64.5, 56.3, 74.5, 89.5, 23]) 
    plt.subplot(211)
    plt.plot(idx, Acc, color='orange')
    plt.ylabel("Class Accuracy %")
    plt.subplot(212)
    plt.bar(idx, count)
    plt.ylabel('Pixel numbers of minerals')
    plt.tight_layout()
    plt.show()

def check_prams_in_pth(file):
    net = torch.load(file)
    print(type(net))
    print(len(net))
    for key, val in net.items():
        print(key, val.size(), sep=" ")

def visalize_params():
    writer = SummaryWriter(comment='test_tensorboard')
    for x in range(100):

        writer.add_scalar('y=2x', x*2, x)
        writer.add_scalar('y=pow(2,x)', 2 ** x, x)

        writer.add_scalars('data/scalar_group', {'xsinx': x * np.sin(x), 'xcosx': x * np.cos(x), 'arctanx': np.arctan(x)}, x)
    writer.close()

if __name__ == "__main__":
    root = "C:/Users/wanghai/Desktop/new3/"
    # merge2(root)
    merge2(root)