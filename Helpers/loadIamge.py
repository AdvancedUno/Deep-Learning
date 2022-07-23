import os
import glob

from PIL import Image
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


import scipy.io as sio

file_path = 'D:/WorkspaceForC/a/final/unlabeled_images/SemiSeg-AEL/finaldat'
file_path = 'D:/WorkspaceForC/U2PL/finaldata/final/labeled_images'
with open("Filenames.txt", mode='w', newline='') as fp:
    i = 0
    for file in os.listdir(file_path):

        i += 1
        print(i)

        f = os.path.join(file_path, file).replace("D:/WorkspaceForC/U2PL/finaldata/final/labeled_images\\", "")
        print(f)
        temp = str(f)
        temp.replace("\\", "/")
        print(temp + os.linesep)
        fp.write(temp + os.linesep)


# files = glob.glob("final/unlabels/*")
# for name in files:
#     #print(name)
#     # if not os.path.isdir(name):
#     #     src = os.path.splitext(name)
#     print(name)
#         #os.rename(name,src[0]+'.png')
#
#     with open(name, "rb") as f:
#         print(1)
#
#         img = Image.open(f)


# df = pd.read_csv('D:/WorkspaceForC/U2PL/finaldata/sample.csv')
# dfc = pd.read_csv('D:/WorkspaceForC/U2PL/sample_submission.csv')
# cnt = []
#
# for i in range(len(df)):
#   if df['file_name'][i] != dfc['file_name'][i]:
#     print(df['file_name'][i], dfc['file_name'][i], i)


# def helper(mask, img_shape, className):
#     canvas = np.zeros(img_shape).T
#
#     if(className == "prostate"):
#         canvas[tuple(zip(*mask))] = 1
#     elif (className == "spleen"):
#         canvas[tuple(zip(*mask))] = 1
#     elif (className == "largeintestine"):
#         canvas[tuple(zip(*mask))] = 1
#     elif (className == "lung"):
#         canvas[tuple(zip(*mask))] = 1
#     elif (className == "kidney"):
#         canvas[tuple(zip(*mask))] = 1
#     # This is the Equivalent for loop of the above command for better understanding.
#     # for pos in range(len(p_loc)):
#     #   canvas[pos[0], pos[1]] = 1
#
#     return canvas
#
#
# def get_mask(rle_string, img_shape, className):
#     rle = [int(i) for i in rle_string.split(' ')]
#     pairs = list(zip(rle[0::2], rle[1::2]))
#
#     p_loc = []
#
#     for start, length in pairs:
#         for p_pos in range(start, start + length):
#             p_loc.append((p_pos % img_shape[1], p_pos // img_shape[0]))
#
#
#     return helper(p_loc, img_shape,className)
#
# df = pd.read_csv('D:/data/hubmap-organ-segmentation/train.csv')
# df.head()
# df['image_path'] = 'D:/data/hubmap-organ-segmentation//train_images/'
# df['image_path'] = df['image_path'].str.cat(df['id'].astype(str))
# df['image_path'] = df['image_path'] + '.tiff'
# df.head()
#
# for part in df['image_path'].unique():
#     x = df[df['image_path'] == part]
#     index_list = x.index
#     idx = index_list[np.random.randint(0, x.shape[0])]
#
#     class_of_scan = df.loc[idx, 'organ']
#     image_path = df.loc[idx, 'image_path']
#     id = df.loc[idx, 'id']
#
#     image = np.array(Image.open(image_path)) / 255
#     k = (df.loc[idx, 'img_height'], df.loc[idx, 'img_width'])
#
#     rle_string = df.loc[idx, 'rle']
#     mask = get_mask(rle_string, k, class_of_scan)
#
#
#     im = Image.fromarray(mask)
#     im = im.convert("L")
#     #im = im.resize((400, 400))
#
#
#
#     im.save("checkpoints/" + str(id) + ".png")


# for part in df['image_path'].unique():
#     x = df[df['image_path'] == part]
#     index_list = x.index
#     idx = index_list[np.random.randint(0, x.shape[0])]
#
#     class_of_scan = df.loc[idx, 'organ']
#     image_path = df.loc[idx, 'image_path']
#     id = df.loc[idx, 'id']
#
#     image = Image.open(image_path)
#
#
#     image = image.convert("RGB")
#     image = image.resize((400,400))
#
#
#     image.save("checkpoints/" + str(id) + ".png")


    # fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    # ax[0].set_title(f'Image : {id}')
    # ax[0].imshow(image)
    #
    # ax[1].set_title(f'Mask : {id}')
    # ax[1].imshow(mask)
    #
    # ax[2].set_title(f'{class_of_scan} Segmented : {id}')
    # ax[2].imshow(np.dstack((mask, np.zeros(mask.shape), np.argmax(image, axis=-1))))
    # plt.show()