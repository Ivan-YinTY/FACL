import math
from PIL import Image
import cv2
from skimage import exposure
from scipy.ndimage import convolve
import numpy as np
import numpy


def imfilter(a, b, gpu=False, conv=True):
    """imfilter function based on MATLAB implementation."""
    if (a.dtype == np.uint8):
        a = a.astype(np.float64)/255     # convert to a numpy float64
    M, N = a.shape
    if conv == True:
        b = np.rot90(b, k=2)     # rotate the image by 180 degree
        return convolve(a, b, mode='nearest')


class Finger_crop():
    def __init__(self,
        mask_h = 4,  # Height of the mask
        mask_w = 20,  # Width of the mask
        heq = False,
        padding_offset = 5,  # Always the same
        padding_threshold = 0.2,  # 0 for UTFVP database (high quality), 0.2 for VERA database (low quality)
        gpu = False,
        color_channel = 'gray',  # the color channel to extract from colored images, if colored images are in the database
        output_w=144,
        output_h=64,
        dataset='SDMULA',
        ** kwargs):  # parameters to be written in the __str__ method

        self.mask_h = mask_h
        self.mask_w = mask_w
        self.heq = heq
        self.padding_offset = padding_offset
        self.padding_threshold = padding_threshold
        self.gpu = gpu
        self.color_channel = color_channel
        self.output_w = output_w
        self.output_h = output_h
        self.dataset = dataset

    def __correct_edge__(self, edge, position, thred=5):
        edge_diff = np.array([s - t for s, t in zip(edge, edge[1:])])
        if position == 'up':
            ind_1 = np.where(edge_diff < -1 * thred)[0]
            ind_2 = np.where(edge_diff > thred)[0]
            if len(ind_1) > 0:
                for ind in ind_1[::-1]:
                    # edge[0:ind_1[-1] + 1] = edge[ind_1[-1] + 1]
                    if ind < len(edge) *0.8:
                        edge[0:ind + 1] = edge[ind + 1]
                        break
            if len(ind_2) > 0:
                for ind in ind_2:
                    if ind > len(edge)*0.2:
                        edge[ind + 1:] = edge[ind]
                        break
                # edge[ind_2[0] + 1:] = edge[ind_2[0]]
        elif position == 'down':
            ind_1 = np.where(edge_diff > thred)[0]
            ind_2 = np.where(edge_diff < -1 * thred)[0]
            if len(ind_1) > 0:
                for ind in ind_1[::-1]:
                    # edge[0:ind_1[-1] + 1] = edge[ind_1[-1] + 1]
                    if ind < len(edge) * 0.8:
                        edge[0:ind + 1] = edge[ind + 1]
                        break
            if len(ind_2) > 0:
                for ind in ind_2:
                    if ind > len(edge) * 0.2:
                        edge[ind + 1:] = edge[ind]
                        break
                # edge[ind_2[0] + 1:] = edge[ind_2[0]]
        return edge

    def __leemask__(self, image):
        img_h, img_w = image.shape

        # Determine lower half starting point vertically
        if numpy.mod(img_h, 2) == 0:
            half_img_h = img_h // 2 + 1
        else:
            half_img_h = numpy.ceil(img_h / 2)

        # Determine lower half starting point horizontally
        if numpy.mod(img_w, 2) == 0:
            half_img_w = img_w // 2 + 1
        else:
            half_img_w = numpy.ceil(img_w / 2)

        # Construct mask for filtering
        mask = numpy.zeros((self.mask_h, self.mask_w))
        mask[0:self.mask_h // 2, :] = -1
        mask[self.mask_h // 2:, :] = 1

        img_filt = imfilter(image, mask, self.gpu, conv=True)
        # Upper part of filtred image
        img_filt_up = img_filt[0:half_img_h - 1, :]
        y_up = img_filt_up.argmax(axis=0)
        # for SDMULA and FVUSM, no need for MMCBNU_6000, some images of MMCBNU_6000 need to be manually correct edge
        # y_up = self.__correct_edge__(y_up, position='up')

        # Lower part of filtred image
        img_filt_lo = img_filt[half_img_h - 1:, :]
        y_lo = img_filt_lo.argmin(axis=0)
        # for SDMULA and FVUSM, no need for MMCBNU_6000, some images of MMCBNU_6000 need to be manually correct edge
        # y_lo = self.__correct_edge__(y_lo, position='down')

        img_filt = imfilter(image, mask.T, self.gpu, conv=True)
        # Left part of filtered image
        # img_filt_lf = img_filt[:, 0:half_img_w]
        # y_lf = img_filt_lf.argmax(axis=1)

        # Right part of filtred image
        img_filt_rg = img_filt[:, half_img_w:]
        y_rg = img_filt_rg.argmin(axis=1)

        finger_mask = np.zeros(image.shape, dtype=np.bool)
        for i in range(0, y_up.size):
            finger_mask[y_up[i]:y_lo[i] + img_filt_lo.shape[0] + 1, i] = True

        # Left region
        # for i in range(0, y_lf.size):
        #     finger_mask[i, 0:y_lf[i] + 1] = False
        # for i in range(0, y_lf.size):
        #     finger_mask[:, 0:int(numpy.median(y_lf[i]))] = False

        # Right region has always the finger ending, crop the padding with the meadian
        if self.dataset == 'FVUSM':
            finger_mask[:, int(numpy.median(y_rg)) + img_filt_rg.shape[1]:] = False

        # Extract y-position of finger edges
        edges = numpy.zeros((2, img_w))
        edges[0, :] = y_up
        # edges[0, 0:int(round(numpy.mean(y_lf))) + 1] = edges[0, int(round(numpy.mean(y_lf))) + 1]

        edges[1, :] = y_lo + img_filt_lo.shape[0]
        # edges[1, 0:int(round(numpy.mean(y_lf))) + 1] = edges[1, int(round(numpy.mean(y_lf))) + 1]

        return (finger_mask, edges)

    def __leemaskMATLAB__(self, image):
        img_h, img_w = image.shape

        # Determine lower half starting point
        if numpy.mod(img_h, 2) == 0:
            half_img_h = img_h // 2 + 1
        else:
            half_img_h = numpy.ceil(img_h / 2)

        # Construct mask for filtering
        mask = numpy.zeros((self.mask_h, self.mask_w))
        mask[0:self.mask_h // 2, :] = -1
        mask[self.mask_h // 2:, :] = 1

        img_filt = imfilter(image, mask, self.gpu, conv=True)

        # Upper part of filtred image
        img_filt_up = img_filt[0:img_h // 2, :]
        y_up = img_filt_up.argmax(axis=0)

        # Lower part of filtred image
        img_filt_lo = img_filt[half_img_h - 1:, :]
        y_lo = img_filt_lo.argmin(axis=0)

        for i in range(0, y_up.size):
            img_filt[y_up[i]:y_lo[i] + img_filt_lo.shape[0], i] = 1

        finger_mask = numpy.ndarray(image.shape, numpy.bool)
        finger_mask[:, :] = False

        finger_mask[img_filt == 1] = True

        # Extract y-position of finger edges
        edges = numpy.zeros((2, img_w))
        edges[0, :] = y_up
        edges[1, :] = numpy.round(y_lo + img_filt_lo.shape[0])

        return (finger_mask, edges)

    def __huangnormalization__(self, image, mask, edges):
        img_h, img_w = image.shape

        bl = (edges[0, :] + edges[1, :]) / 2  # Finger base line
        x = numpy.arange(0, img_w)
        A = numpy.vstack([x, numpy.ones(len(x))]).T

        # Fit a straight line through the base line points
        w = numpy.linalg.lstsq(A, bl)[0]  # obtaining the parameters

        angle = -1 * math.atan(w[0])  # Rotation
        tr = img_h / 2 - w[1]  # Translation
        scale = 1.0  # Scale

        # Affine transformation parameters
        sx = sy = scale
        cosine = math.cos(angle)
        sine = math.sin(angle)

        a = cosine / sx
        b = -sine / sy
        # b = sine/sx
        c = 0  # Translation in x

        d = sine / sx
        e = cosine / sy
        f = tr  # Translation in y
        # d = -sine/sy
        # e = cosine/sy
        # f = 0

        g = 0
        h = 0
        # h=tr
        i = 1

        T = numpy.matrix([[a, b, c], [d, e, f], [g, h, i]])
        Tinv = numpy.linalg.inv(T)
        Tinvtuple = (Tinv[0, 0], Tinv[0, 1], Tinv[0, 2], Tinv[1, 0], Tinv[1, 1], Tinv[1, 2])

        img = Image.fromarray(image)
        image_norm = img.transform(img.size, Image.AFFINE, Tinvtuple, resample=Image.BICUBIC)
        # image_norm = img.transform(img.size, Image.AFFINE, (a,b,c,d,e,f,g,h,i), resample=Image.BICUBIC)
        image_norm = numpy.array(image_norm)

        finger_mask = numpy.zeros(mask.shape)
        finger_mask[mask == True] = 1

        img_mask = Image.fromarray(finger_mask)
        mask_norm = img_mask.transform(img_mask.size, Image.AFFINE, Tinvtuple, resample=Image.BICUBIC)
        # mask_norm = img_mask.transform(img_mask.size, Image.AFFINE, (a,b,c,d,e,f,g,h,i), resample=Image.BICUBIC)
        mask_norm = numpy.array(mask_norm)

        mask[:, :] = False
        mask[mask_norm == 1] = True

        return (image_norm, mask)

    def crop_finger(self, image):

        if self.heq:
            image = exposure.equalize_hist(image)
        else:
            image = image

        # finger_mask, finger_edges = self.__leemaskMATLAB__(image_eq)
        finger_mask, finger_edges = self.__leemask__(image)
        # finger_mask, finger_edges = self.__leemaskMATLAB__(image)
        ori_mask = (finger_mask *255).astype('uint8')
        image_norm, finger_mask_norm = self.__huangnormalization__(image, finger_mask, finger_edges)

        mask = (finger_mask_norm * 255).astype(np.uint8)
        rect = cv2.boundingRect(mask)
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        # output = image_norm[y:y + h, x:x + w]
        output = image_norm[y:y + h, x:x + w]
        output = cv2.resize(output, (self.output_w, self.output_h))

        cv2.imshow('original', image)
        cv2.imshow('normalized image', image_norm)
        cv2.imshow('normalized mask', mask)
        cv2.imshow('ori mask', ori_mask)
        cv2.imshow('finger crop', output)
        cv2.waitKey(0)
        # cv2.imwrite('original_image.png', image)
        # cv2.imwrite('normalized_image.png', image_norm)
        # cv2.imwrite('normalized_mask.png', mask)
        # cv2.imwrite('original_mask.png', ori_mask)
        # cv2.imwrite('finger_crop.png', output)
        return output

    def extract_edges_HKPU(self, mask):
        h, w = mask.shape
        half_h = h // 2
        half_w = w // 2
        upper_edge_ind = np.argmax(mask, axis=0)
        lower_edge_ind = h - 1 - np.argmax(mask[::-1, :], axis=0)
        right_edge_ind = w - 1 - np.argmax(mask[:, half_w:][:, ::-1], axis=1)
        mask[:, int(np.median(right_edge_ind)):] = 0
        # msk = np.zeros(mask.shape, np.uint8)
        # for i in range(w):
        #     msk[int(upper_edge_ind[i]):int(lower_edge_ind[i]),i] = 255
        # cv2.imshow("test",msk)
        edges = np.zeros((2, w))
        edges[0, :] = upper_edge_ind
        edges[1, :] = lower_edge_ind
        return mask, edges

    def crop_finger_HKPU(self, image, mask):
        finger_mask, finger_edges = self.extract_edges_HKPU(mask)
        image_norm, finger_mask_norm = self.__huangnormalization__(image, finger_mask.astype(np.bool), finger_edges)
        finger_mask_norm = (finger_mask_norm * 255).astype(np.uint8)
        rect = cv2.boundingRect(finger_mask_norm)
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        output = image_norm[y:y + h, x:x + w]
        output = cv2.resize(output, (self.output_w, self.output_h))

        # cv2.imshow('original', image)
        # cv2.imshow('normalized image', image_norm)
        # cv2.imshow('ori mask', mask)
        # cv2.imshow('normalized mask', finger_mask_norm)
        # cv2.imshow('finger crop', output)
        # cv2.waitKey(0)
        # cv2.imwrite('original_image.png', image)
        # cv2.imwrite('normalized_image.png', image_norm)
        # cv2.imwrite('normalized_mask.png', mask)
        # cv2.imwrite('original_mask.png', ori_mask)
        # cv2.imwrite('finger_crop.png', output)
        return output


def extract_roi_FVUSM():
    fc = Finger_crop(output_w=144, output_h=64, dataset='FVUSM')
    import os
    from PIL import Image
    # first session
    src = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/FV-USM-RAW/1st_session/raw_data'
    dst = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/FV-USM-processed'
    if not os.path.exists(dst):
        os.mkdir(dst)

    class_dirs = os.listdir(src)
    class_dirs.sort()
    for finger in class_dirs:
        if os.path.isdir(os.path.join(src, finger)):
            for i in range(1, 7):
                img_path = os.path.join(src, finger, '%02d.jpg' % (i))
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.T
                img = img[100:540, 0:440]
                # img = img[30:240, 0:280]
                # img = cv2.resize(img, (336, 190))
                output_img = fc.crop_finger(img)
                output_img = Image.fromarray(output_img)
                output_name = os.path.join(dst, finger + '_' + '%02d.bmp' % (i))
                output_img.save(output_name, 'bmp')

    # second session
    src = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/FV-USM-RAW/2nd_session/raw_data'
    class_dirs = os.listdir(src)
    class_dirs.sort()
    for finger in class_dirs:
        if os.path.isdir(os.path.join(src, finger)):
            for i in range(1, 7):
                img_path = os.path.join(src, finger, '%02d.jpg' % (i))
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.T
                img = img[100:540, 0:440]
                # img = cv2.resize(img, (336, 190))
                output_img = fc.crop_finger(img)
                output_img = Image.fromarray(output_img)
                output_name = os.path.join(dst, finger + '_' + '%02d.bmp' % (i + 6))
                output_img.save(output_name, 'bmp')


def test_FVUSM():
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/FV-USM-RAW/1st_session/raw_data/003_1/01.jpg')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/FV-USM-RAW/1st_session/raw_data/004_1/06.jpg')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/FV-USM-RAW/1st_session/raw_data/022_2/01.jpg')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/FV-USM-RAW/1st_session/raw_data/063_4/04.jpg')
    x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/FV-USM-RAW/2nd_session/raw_data/094_3/06.jpg')
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = x.T
    cv2.imshow('original image (full size)', x)
    x = x[100:540, 0:440]
    # x = cv2.resize(x, (336, 190))
    fc = Finger_crop(output_w=144, output_h=64, dataset='FVUSM')
    output = fc.crop_finger(x)
    # cv2.imshow('output', output)
    # cv2.imshow('flip', output[::-1, :])
    # cv2.waitKey(0)


def extract_roi_SDMULA():
    fc = Finger_crop(output_w=144, output_h=64, dataset='SDMULA')

    import os
    from PIL import Image
    src = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW'
    dst = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-processed-64_144_15_10'
    if not os.path.exists(dst):
        os.mkdir(dst)

    people_dirs = os.listdir(src)
    people_dirs.sort()
    for person in people_dirs:
        hand_dirs = os.listdir(os.path.join(src, person))
        hand_dirs.sort()
        for hand in hand_dirs:
            img_list = os.listdir(os.path.join(src, person, hand))
            img_list.sort()
            for img_name in img_list:
                if img_name.endswith('.bmp'):
                    img_path = os.path.join(src, person, hand, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img[30:240, 0:280]
                    # img = img[40:230, 40:280]
                    # img = cv2.resize(img, (336, 190))
                    output_img = fc.crop_finger(img)
                    output_img = Image.fromarray(output_img)
                    output_name = os.path.join(dst, person + '_' + hand + '_' + img_name)
                    output_img.save(output_name, 'bmp')


def test_SDMULA():
    x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/001/left/index_1.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/009/right/ring_4.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/019/right/ring_2.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/035/right/index_5.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/074/left/ring_5.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/106/left/middle_6.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/087/left/index_1.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/102/right/index_6.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/097/right/ring_4.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/097/right/middle_3.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/026/right/middle_1.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/058/right/ring_5.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/025/right/middle_5.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/041/right/ring_6.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/002/left/index_4.bmp')
    # x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/SDMULA-RAW/040/left/ring_5.bmp')
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    cv2.imshow('original image (full size)', x)
    x = x[30:240, 0:280]
    # x = cv2.resize(x, (336, 190))
    fc = Finger_crop()
    output = fc.crop_finger(x)
    # cv2.imshow('output', output)
    # cv2.imshow('flip', output[::-1, :])
    # cv2.waitKey(0)


def extract_roi_MMCNBU_6000():
    fc = Finger_crop(output_w=144, output_h=64, dataset='MMCBNU_6000')

    import os
    from PIL import Image
    src = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/MMCBNU_6000/MMCBNU_6000/Captured images'
    dst = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/MMCBNU_6000-processed'
    if not os.path.exists(dst):
        os.mkdir(dst)

    people_dirs = os.listdir(src)
    people_dirs.sort()
    for person in people_dirs:
        finger_dirs = os.listdir(os.path.join(src, person))
        finger_dirs.sort()
        for finger in finger_dirs:
            img_list = os.listdir(os.path.join(src, person, finger))
            img_list.sort()
            for img_name in img_list:
                if img_name.endswith('.bmp'):
                    img_path = os.path.join(src, person, finger, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img[0:480, 0:640]
                    # img = img[40:230, 40:280]
                    # img = cv2.resize(img, (336, 190))
                    output_img = fc.crop_finger(img)
                    output_img = Image.fromarray(output_img)
                    output_name = os.path.join(dst, person + '_' + finger + '_' + img_name)
                    output_img.save(output_name, 'bmp')


def test_MMCBNU_6000():
    x = cv2.imread("/home/weifeng/Desktop/datasets/FingerVeinDatasets/MMCBNU_6000/MMCBNU_6000/Captured images/075/L_Fore/01.bmp")
    x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/MMCBNU_6000/MMCBNU_6000/Captured images/074/L_Ring/07.bmp')
    x = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/MMCBNU_6000-processed/075_R_Ring_03.bmp')
    x = cv2.imread("/home/weifeng/Desktop/datasets/FingerVeinDatasets/MMCBNU_6000-processed/097_R_Ring_06.bmp")
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    cv2.imshow('original image (full size)', x)
    x = x[0:480, 0:640]
    # x = cv2.resize(x, (336, 190))
    fc = Finger_crop()
    output = fc.crop_finger(x)
    cv2.imshow('output', output)
    # cv2.imshow('flip', output[::-1, :])
    cv2.waitKey(0)


def extract_roi_HKPU():
    fc = Finger_crop(output_w=144, output_h=64, dataset='PolyU')
    import os
    from PIL import Image
    src = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original'
    src_mask = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Processed/Mask'
    dst = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-processed'
    if not os.path.exists(dst):
        os.mkdir(dst)
    people_dirs = os.listdir(src)
    people_dirs.sort()
    for person in people_dirs:
        if int(person) <= 9999:  # process all the persons
            finger_dirs = os.listdir(os.path.join(src, person))
            finger_dirs.sort()
            for finger in finger_dirs:
                session_dirs = os.listdir(os.path.join(src, person, finger))
                session_dirs.sort()
                for session in session_dirs:
                    img_list = os.listdir(os.path.join(src, person, finger, session))
                    img_list.sort()
                    count = 1
                    for img_name in img_list:
                        if img_name.endswith('.bmp'):
                            img_path = os.path.join(src, person, finger, session, img_name)
                            img = cv2.imread(img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            mask_path = os.path.join(src_mask, img_name)
                            mask = cv2.imread(mask_path)
                            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                            img = cv2.bitwise_and(img, img, mask=mask)
                            output_img = fc.crop_finger_HKPU(img, mask)
                            output_img = Image.fromarray(output_img)
                            output_name = os.path.join(dst, "%03d" % int(person) + '_' + finger + '_' + session + "_%02d.bmp" % count)
                            output_img.save(output_name, 'bmp')
                            count += 1


def test_HKPU():
    # img = cv2.imread("/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/98/f2/2/98_6_f2_2.bmp")
    # mask = cv2.imread("/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Processed/Mask/98_6_f2_2.bmp")
    img = cv2.imread('/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/86/f1/2/86_5_f1_2.bmp')
    mask = cv2.imread("/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Processed/Mask/86_5_f1_2.bmp")

    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/82/f2/1/82_5_f2_1.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/82/f2/1/82_6_f2_1.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/82/f1/1/82_4_f1_1.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/71/f1/2/71_1_f1_2.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/71/f1/2/71_5_f1_2.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/71/f1/2/71_6_f1_2.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/66/f1/2/66_4_f1_2.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/66/f1/2/66_5_f1_2.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/37/f1/2/37_2_f1_2.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/37/f1/2/37_3_f1_2.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/37/f1/2/37_6_f1_2.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/37/f1/1/37_2_f1_1.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/21/f1/1/21_1_f1_1.bmp'
    '/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-RAW/FingerVein/Original/14/f2/1/14_2_f2_1.bmp'

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("aa", img)
    # cv2.waitKey(0)
    fc = Finger_crop()
    output = fc.crop_finger_HKPU(img, mask)


def extract_roi_FTVein90DB():
    fc = Finger_crop(output_w=144, output_h=64, dataset='FTVein90DB')
    import os
    from PIL import Image
    src = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/FT_Vein_90DB/FT_Vein_90DB'
    dst = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/FTVein90DB-processed'
    if not os.path.exists(dst):
        os.mkdir(dst)

    img_list = os.listdir(src)
    img_list.sort()
    for img_name in img_list:
        if img_name.endswith('.bmp'):
            img_path = os.path.join(src, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[::-1, :].T
            output_img = fc.crop_finger(img)
            output_img = Image.fromarray(output_img)
            output_name = os.path.join(dst, img_name)
            output_img.save(output_name, 'bmp')


def test_FTVein90DB():
    x = cv2.imread("/home/weifeng/Desktop/datasets/FingerVeinDatasets/FT_Vein_90DB/FT_Vein_90DB/88800004.F03.6.000.bmp")
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = x[::-1, :].T
    cv2.imshow('original image (full size)', x)
    fc = Finger_crop(output_w=144, output_h=64, dataset='FTVein90DB')
    output = fc.crop_finger(x)
    cv2.waitKey(0)


if __name__ == '__main__':
    # extract_roi_SDMULA()
    # test_SDMULA()
    # extract_roi_FVUSM()
    # test_FVUSM()
    # extract_roi_MMCNBU_6000()
    # test_MMCBNU_6000()
    # extract_roi_HKPU()
    # test_HKPU()
    # extract_roi_FTVein90DB()
    test_FTVein90DB()
