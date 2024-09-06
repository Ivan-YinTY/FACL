import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import os

dataset_dir = '../vein_databases/synthetic_samples_fv'
output_dir = "../vein_databases/synthetic_fv_hist"

def hist(input_img_path, output_img_path):
    #全局直方图均衡化
    img = cv.imread(input_img_path, 0)
    img_equalize = cv.equalizeHist(img)
    cv.imwrite(output_img_path, img_equalize)
    # cv.imshow("img",img)
    # cv.imshow("img_equalize",img_equalize)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img_equalize

def gamma(input_img_path, output_img_path):
    # 伽马变换
    img = cv.imread(input_img_path, 0)
    img_norm = img / 255.0  # 注意255.0得采用浮点数
    img_gamma = np.power(img_norm, 0.4) * 255.0
    img_gamma = img_gamma.astype(np.uint8)
    cv.imwrite(output_img_path, img_gamma)
    # cv.imshow("img", img)
    # cv.imshow("img_gamma", img_gamma)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

def log(c, img):
    output = c * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    return output

def logtrans(input_img_path, output_img_path):
    #对数变换
    img = cv.imread(input_img_path, 0)
    output = log(42, img)
    cv.imwrite(output_img_path, output)
    # cv.imshow("img", img)
    # cv.imshow("img_gamma", img_gamma)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

if __name__ == '__main__':
    # 获得需要转化的图片路径并生成目标路径
    image_filenames = [(os.path.join(dataset_dir, x), os.path.join(output_dir, x))
                       for x in os.listdir(dataset_dir)]
    # 转化所有图片
    for path in image_filenames:
        # hist(path[0], path[1])
        # gamma(path[0], path[1])
        logtrans(path[0], path[1])
