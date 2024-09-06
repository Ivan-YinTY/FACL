import os


def expand_classes(dataset_path):
    from PIL import Image
    outdir = dataset_path + '-expandClasses-all'
    img_list = os.listdir(dataset_path)
    img_list.sort()
    # augment class by flipping
    for i in range(len(img_list)):
        img = Image.open(os.path.join(dataset_path, img_list[i]))
        img_1 = img.transpose(Image.FLIP_TOP_BOTTOM)
        # img_1 = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_1.save(os.path.join(outdir, 'expand_' + img_list[i]))
        img.save(os.path.join(outdir, img_list[i]))


def generate_aug_class_list(original_list, sample_per_class):
    f_name = original_list.split('.')[0] + '_augClass' + '.txt'
    f_aug_class = open(f_name, 'w')
    ori_list = open(original_list).readlines()
    num_classes = len(ori_list) // sample_per_class
    for i in ori_list:
        f_aug_class.writelines("%s" % i)
    for i in ori_list:
        img_name = i.split(' ')[0]
        label = int(i.split(' ')[1])
        f_aug_class.writelines("expand_%s %d\n" % (img_name, label + num_classes))


def split_openset(dataset_name, dataset_path, samples_per_class):
    import random
    img_list = os.listdir(dataset_path)
    img_list.sort()
    num_class = len(img_list) // samples_per_class
    train_classes = random.sample(range(0, num_class), num_class//2)
    train_classes.sort()
    test_classes = [i for i in range(0, num_class) if i not in train_classes]
    trainlist_name = dataset_name + '_trainlist.txt'
    testlist_name = dataset_name + '_testlist.txt'
    f_train = open(trainlist_name, 'w')
    f_test = open(testlist_name, 'w')
    label = 0
    for i in train_classes:
        class_samples = img_list[i*samples_per_class: (i+1)*samples_per_class]
        for j in class_samples:
            f_train.writelines("%s %d\n" % (j, label))
        label += 1
    label = 0
    for i in test_classes:
        class_samples = img_list[i*samples_per_class: (i+1)*samples_per_class]
        for j in class_samples:
            f_test.writelines("%s %d\n" % (j, label))
        label += 1
    f_train.close()
    f_test.close()
    generate_aug_class_list(trainlist_name, samples_per_class)
    generate_aug_class_list(testlist_name, samples_per_class)


def split_openset_hkpu(dataset_path):
    import random
    img_list = os.listdir(dataset_path)
    img_list.sort()
    num_class = 210
    train_classes = random.sample(range(0, num_class), num_class//2)
    train_classes.sort()
    test_classes = [i for i in range(0, num_class) if i not in train_classes]
    trainlist_name = 'PolyU_trainlist.txt'
    testlist_name = 'PolyU_testlist.txt'
    f_train = open(trainlist_name, 'w')
    f_test = open(testlist_name, 'w')
    label = 0
    for i in train_classes:
        class_samples = img_list[i*12: (i+1)*12]
        for j in class_samples:
            f_train.writelines("%s %d\n" % (j, label))
        label += 1
    label = 0
    for i in test_classes:
        class_samples = img_list[i*12: (i+1)*12]
        for j in class_samples:
            f_test.writelines("%s %d\n" % (j, label))
        label += 1
    f_train.close()
    f_test.close()
    generate_aug_class_list(trainlist_name, 12)
    generate_aug_class_list(testlist_name, 12)

    ori_list = open(trainlist_name).readlines()
    f = open(trainlist_name.split('.')[0] + '_with_last_51_subjects' + '.txt', 'w')
    for i in ori_list:
        f.writelines(i)
    label = 105
    for i in range(0, 102):
        class_samples = img_list[210*12 + i * 6: 210*12 + (i + 1) * 6]
        for j in class_samples:
            f.writelines("%s %d\n" % (j, label))
        label += 1

    ori_list = open(testlist_name).readlines()
    f = open(testlist_name.split('.')[0] + '_with_last_51_subjects' + '.txt', 'w')
    for i in ori_list:
        f.writelines(i)
    label = 105
    for i in range(0, 102):
        class_samples = img_list[210*12 + i * 6: 210*12 + (i + 1) * 6]
        for j in class_samples:
            f.writelines("%s %d\n" % (j, label))
        label += 1


# def split_openset_hkpu(dataset, dataset_path):
#     img_list = os.listdir(dataset_path)
#     img_list.sort()
#     f_all = open(dataset + '_filelist.txt', 'w')
#     for i in range(len(img_list)):
#         if i < 2520:
#             label = i // 12
#         else:
#             label = 210 + ((i - 2520) // 6)
#         f_all.writelines(img_list[i] + ' ' + str(label) + '\n')
#     f_all.close()
#
#     filelist = open(dataset + '_filelist.txt', 'r').readlines()
#     f_train = open(dataset + '_trainlist_openset.txt', 'w')
#     f_test = open(dataset + '_testlist_openset.txt', 'w')
#     f_train.writelines(filelist[0:1260])
#     f_test.writelines(filelist[1260:])


def split_closeset(dataset, dataset_path):
    img_list = os.listdir(dataset_path)
    img_list.sort()

    if dataset == 'SDMULA':
        f_all = open(dataset + '_filelist.txt', 'w')
        label = 0
        for i in range(len(img_list)):
            if i % 6 == 0 and i != 0:
                label += 1
            f_all.writelines(img_list[i] + ' ' + str(label) + '\n')
        f_all.close()

        filelist = open(dataset + '_filelist.txt', 'r').readlines()
        f_train = open(dataset + '_trainlist_closeset.txt', 'w')
        f_test = open(dataset + '_testlist_closeset.txt', 'w')
        for i in range(len(filelist) // 6):
            f_train.writelines(filelist[i * 6])
            f_train.writelines(filelist[i * 6 + 2])
            f_train.writelines(filelist[i * 6 + 4])
            f_train.writelines(filelist[i * 6 + 5])
            f_test.writelines(filelist[i * 6 + 1])
            f_test.writelines(filelist[i * 6 + 3])

    elif dataset == 'FVUSM':
        f_all = open(dataset + '_filelist.txt', 'w')
        label = 0
        for i in range(len(img_list)):
            if i % 12 == 0 and i != 0:
                label += 1
            f_all.writelines(img_list[i] + ' ' + str(label) + '\n')
        f_all.close()

        filelist = open(dataset + '_filelist.txt', 'r').readlines()
        f_train = open(dataset + '_trainlist_closeset.txt', 'w')
        f_test = open(dataset + '_testlist_closeset.txt', 'w')

        for i in range(len(filelist) // 2):
            f_train.writelines(filelist[i * 2])
            f_test.writelines(filelist[i * 2 + 1])


def get_filelist_DB90():
    dataset = '/home/weifeng/Desktop/datasets/FingerVeinDatasets/FTVein90DB-processed'
    img_list = os.listdir(dataset)
    img_list.sort()
    f = open('FTVein90DB_testlist_openset.txt', 'w')
    for i, img_name in enumerate(img_list):
        f.writelines(img_name + ' ' + str(i//3) + '\n')


def gen_list():
    # flist = open('PolyU_SecHalf_trainlist_openset.txt', 'r').readlines()
    # f_out = open('PolyU_SecHalf_trainlist_openset_1.txt', 'w')
    # flist = open('FVUSM_testlist_openset.txt', 'r').readlines()
    # f_out = open('FVUSM_SecHalf_trainlist_openset.txt', 'w')
    flist = open('MMCBNU_6000_testlist_openset.txt', 'r').readlines()
    f_out = open('MMCBNU_6000_SecHalf_trainlist_openset.txt', 'w')

    img_list = [i.split()[0] for i in flist]
    label = 0
    samples_per_class = 10
    for i in range(len(img_list)):
        if i % samples_per_class == 0 and i != 0:
            label += 1
        f_out.writelines(img_list[i] + ' ' + str(label) + '\n')
    f_out.close()


if __name__ == '__main__':
    # split_openset('FVUSM', '/home/weifeng/Desktop/datasets/FingerVeinDatasets/FV-USM-processed', 12)
    # split_openset('MMCBNU_6000', '/home/weifeng/Desktop/datasets/FingerVeinDatasets/MMCBNU_6000-processed', 10)
    # split_openset_hkpu('/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-processed')
    # expand_classes('/home/weifeng/Desktop/datasets/FingerVeinDatasets/MMCBNU_6000-processed')
    # expand_classes('/home/weifeng/Desktop/datasets/FingerVeinDatasets/PolyU-processed')

    flist = os.listdir("/home/weifeng/Downloads/PyTorch-GAN-master/data/unpaired_veins_FVUSM/train_4_methods/C_fake")
    flist.sort()
    f = open('FVUSM_synthetic.txt', 'w')
    for i in range(len(flist)):
        f.writelines(flist[i] + ' ' + str(i) + '\n')
