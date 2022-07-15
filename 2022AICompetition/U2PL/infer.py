import logging
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from PIL import Image
from tqdm import tqdm

from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.utils import (
    AverageMeter,
    check_makedirs,
    colorize,
    load_state,
    convert_state_dict,
    intersectionAndUnion,
)
import pandas as pd

import cv2






def contour_remove_png(image, color, noTruck3):

    # img1 = cv2.imread(f'/content/drive/MyDrive/SIA/unlabel_images_label/{name}.png')


    #img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #print(image)

    image = Image.fromarray((image).astype(np.uint8))

    #image = np.array(image)
    image = np.array(image)


    img2 = image



    # ret, img_binary = cv2.threshold(image, 0, 255, 0)
    #
    # hierarchy, contours = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    ret, img_binary = cv2.threshold(img2, 0, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #print(len(contours))



    contours = sorted(contours, key=cv2.contourArea)

    idx2 =0

    m = 0
    idx = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])

        if area > m:
            m = area
            idx2 = idx
            idx = i

    # print(contours[0][0][0][1])
    # print(contours[0][1][0][1])

    area = cv2.contourArea(contours[idx])
    if(color == 4 and len(contours) > 1):
        if(cv2.contourArea(contours[idx2]) > area*0.5 and contours[idx2][0][0][1] > contours[idx][0][0][1]):
            #print(area)
            #print(cv2.contourArea(contours[idx2]))
            idx = idx2






    # m = 0
    # idx = 0
    # for i in range(len(contours)):
    #     area = cv2.contourArea(contours[i])
    #     if area > m:
    #         m = area
    #         idx = i

    mask = np.zeros(img2.shape[:2], dtype=image.dtype)
    #cv2.drawContours(mask, [contours[idx]], 0, (int(color)), -1)
    cv2.drawContours(mask, [contours[idx]], 0, (int(color)), -1)







    #result = cv2.bitwise_and(image, image, mask=mask)

    # maxContour = max(contours, key=cv2.contourArea)
    #
    # mask = np.zeros(image.shape[:2], np.uint8)
    # cv2.drawContours(mask, [maxContour], -1, 255, -1)
    #
    # image = np.zeros(img2.shape[:3], np.uint8)
    # height = img2.shape[:2][1]
    # image[:, 0:height] = (255, 255, 255)
    #
    # locations = np.where(mask != 0)
    # image[locations[0], locations[1]] = img2[locations[0], locations[1]]




    # m = 0
    # idx = 0
    # for i in range(len(contours)):
    #     area = cv2.contourArea(contours[i])
    #     if area > m:
    #         m = area
    #         idx = i
    #
    # mask = np.zeros(img2.shape[:2], dtype=image.dtype)
    #
    # #print(mask.shape)
    #
    # cv2.drawContours(mask, [contours[idx]], 0, (255), -1)
    # result = cv2.bitwise_and(image, image, mask=mask)


    return mask



def submit(result):
      noTruck3 = False
      #print(result)
      k = -1
      a, b = np.unique(result, return_counts=True)
      if len(a) == 1:
        sh = result.shape
        w, h = sh[0], sh[1]
        return 'ship', f'{int(w * h / 2)} 5'

      a, b = a[1:], b[1:]
      #print(result)







      noTruck2 = None

      #noTruck3 = (result == [3,3,3])



      # for i in range(len(result)):
      #   if np.array_equal(result[i], 3):
      #       noTruck3 = True

     # noTruck3 = np.where((result == 3))



      m = a[b.argmax()]

      #print(m)
      # if(m == 1 and noTruck3 == True):
      #       m = 3





      label = ''

      if m == 4:
        label = 'ship'
      elif m == 1:
        label = 'container_truck'
      elif m == 2:
        label = 'forklift'
      elif m == 3:
        label = 'reach_stacker'

      #result = np.expand_dims(result, axis=0)


      result = contour_remove_png(result, m, noTruck3)

      mask = result

      result = np.reshape(result, -1)


      start = True
      ch = False
      idx = []
      cnt = 0
      for i in range(len(result)):

        if result[i] == m:
          cnt += 1
          if start:
            idx.append(str(i))
            start = False
            ch = True
        elif result[i] != m and ch:
          start = True
          ch = False
          idx.append(str(cnt))
          cnt = 0

      if result[-1] == m and start == False:
        idx.append(str(len(result) - 1 - int(idx[-1])))

      #print(mask)
      #print(label)

      return label, mask, ' '.join(idx)



# Setup Parser
def get_parser():
    parser = ArgumentParser(description="PyTorch Evaluation")
    parser.add_argument("--config", type=str, default="finaldata/config.yaml")
    parser.add_argument(
        "--model_path",
        type=str,
        default="finaldata/checkpoints/ckpt.pth",
        help="evaluation model path",
    )
    parser.add_argument(
        "--save_folder", type=str, default="finaldata/checkpoints/", help="results save folder"
    )
    return parser


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger, cfg
    args = get_parser().parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = get_logger()
    logger.info(args)

    cfg_dset = cfg["dataset"]
    mean, std = cfg_dset["mean"], cfg_dset["std"]
    num_classes = cfg["net"]["num_classes"]
    crop_size = cfg_dset["val"]["crop"]["size"]
    crop_h, crop_w = crop_size

    assert num_classes > 1

    os.makedirs(args.save_folder, exist_ok=True)
    gray_folder = os.path.join(args.save_folder, "gray")
    os.makedirs(gray_folder, exist_ok=True)
    color_folder = os.path.join(args.save_folder, "color")
    os.makedirs(color_folder, exist_ok=True)

    cfg_dset = cfg["dataset"]
    data_root, f_data_list = cfg_dset["val"]["data_root"], cfg_dset["val"]["data_list"]

    #print(f_data_list)
    data_list = []

    if "cityscapes" in data_root:
        for line in open(f_data_list, "r"):
            arr = [
                line.strip(),
                "gtFine/" + line.strip()[12:-15] + "gtFine_labelTrainIds.png",
            ]
            arr = [os.path.join(data_root, item) for item in arr]
            data_list.append(arr)

    else:
        for line in open(f_data_list, "r"):
            arr = [
                #line.strip().replace("final/", ""),
                #line.strip().replace("final/", ""),
                "test/" + line.strip(),
                "test/" + line.strip(),
                #(line.strip().replace("final/", "").replace("labeled_images", "labels")).replace(".jpg",".png"),
            ]
            arr = [os.path.join(data_root, item) for item in arr]
            data_list.append(arr)

    # Create network.
    args.use_auxloss = True if cfg["net"].get("aux_loss", True) else False
    logger.info("=> creating model from '{}' ...".format(args.model_path))

    cfg["net"]["sync_bn"] = False
    model = ModelBuilder(cfg["net"])

    checkpoint = torch.load(args.model_path)





    key = "teacher_state" if "teacher_state" in checkpoint.keys() else "model_state"
    key = "teacher_state"
    logger.info(f"=> load checkpoint[{key}]")

    saved_state_dict = convert_state_dict(checkpoint[key])
    #model.load_state_dict(saved_state_dict, strict=False)
    load_state(cfg["saver"]["pretrain"], model, key="teacher_state")
    model.cuda()
    logger.info("Load Model Done!")

    input_scale = [914, 850]
    colormap = create_pascal_label_colormap()

    df = pd.read_csv("finaldata/sample_submission.csv")
    files = df['file_name']

    labels = ('background', 'container_truck', 'forklift', 'reach_stacker', 'ship')
    classes = []
    index = []
    names = []
    j = 0





    model.eval()
    for image_path, label_path in tqdm(data_list):
        image_name = image_path.split("/")[-1]
        j += 1


        #csvName = image_name.replace("labeled_images\\", "")
        csvName = image_name.replace("test\\", "")
        #print(csvName)

        image = Image.open(image_path).convert("RGB")

        #input_scale[0] = image[0]
        print(input_scale[0])
        print(image.size)

        original_image = image
        i = csvName



        image = np.asarray(image).astype(np.float32)
        h, w, _ = image.shape
        image = (image - mean) / std

        image = torch.Tensor(image).permute(2, 0, 1)
        image = image.unsqueeze(dim=0)


        image = F.interpolate(image, input_scale, mode="bilinear", align_corners=True)



        output = net_process(model, image)

        output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)





        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        #print(mask)

        #print(mask.dtype)

        # cv2.IMREAD_UNCHANGED
        # mask = np.where(mask == 1, 1, mask)
        # mask = np.where(mask == 2, 1, mask)
        # mask = np.where(mask == 3, 1, mask)
        # mask = np.where(mask == 4, 1, mask)
        # mask = np.where(mask == 5, 1, mask)
        # mask = np.where(mask == 6, 1, mask)
        # mask = np.where(mask == 7, 1, mask)
        # mask = np.where(mask == 8, 1, mask)
        # mask = np.where(mask == 9, 1, mask)
        # mask = np.where(mask == 11, 1, mask)
        # mask = np.where(mask == 12, 1, mask)
        # mask = np.where(mask == 13, 1, mask)
        # mask = np.where(mask == 17, 1, mask)
        # mask = np.where(mask == 18, 1, mask)



        #mask = torch.from_numpy(mask)

        #mask = Image.fromarray(mask)

        label, mask, idx = submit(mask)



        color_mask = Image.fromarray(colorful(mask, colormap))


        color_mask = np.array(color_mask)
        #print(color_mask.shape)




        #color_mask = np.moveaxis(color_mask, -1, 0)
        #print(color_mask.shape)
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
        #color_mask = np.moveaxis(color_mask, -1, 0)


        original_image = np.array(original_image)
        original_image = original_image.astype(np.uint8)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)


        #result = cv2.bitwise_and(original_image, color_mask)
        #print(color_mask.size)
        #color_mask.save((os.path.join(color_folder, image_name)).replace("labeled_images\\", ""))


        #color_mask.save((os.path.join(color_folder, image_name)).replace("test\\", ""))


        cv2.imwrite((os.path.join(color_folder, image_name)).replace("test\\", ""), original_image)


        classes.append(label)
        index.append(idx)
        names.append(i)
        #print(len(names))

        sub = pd.DataFrame({'file_name': names, 'class': classes, 'prediction': index})
        sub.to_csv(f'finaldata/sample_submission.csv', index=False)

    mask = Image.fromarray((mask).astype(np.uint8))
    #mask = Image.fromarray(mask)


        #mask.save((os.path.join(gray_folder, image_name)).replace("labeled_images\\", ""))


def colorful(mask, colormap):

    #print(mask.shape)
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])

    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]
        #color_mask[mask == i] = [100,100,100]

    return np.uint8(color_mask)


def create_pascal_label_colormap():
    """Creates a label colormap used in Pascal segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = 255 * np.ones((5, 3), dtype=np.uint8)
    colormap[0] = [255, 255, 255]
    colormap[1] = [255, 0, 0]
    colormap[2] = [0, 255, 0]
    colormap[3] = [0, 0, 255]
    colormap[4] = [255, 255, 0]

    #colormap = np.zeros((256, 3), dtype=np.uint8)
    # colormap[0] = [255, 255, 255]
    # colormap[1] = [255, 255, 255]
    # colormap[2] = [255, 255, 255]
    # colormap[3] = [255, 255, 255]
    # colormap[4] = [255, 255, 255]
    # colormap[5] = [255, 255, 255]
    # colormap[6] = [255, 255, 255]
    # colormap[7] = [255, 255, 255]
    # colormap[8] = [255, 255, 255]
    # colormap[9] = [255, 255, 255]
    # colormap[10] = [255, 0, 0]
    # colormap[11] = [255, 255, 255]
    # colormap[12] = [255, 255, 255]
    # colormap[13] = [255, 255, 255]
    # colormap[14] = [0, 255, 0]
    # colormap[15] = [0, 0, 255]
    # colormap[16] = [0, 255, 255]
    # colormap[17] = [255, 255, 255]
    # colormap[18] = [255, 255, 255]

    return colormap


@torch.no_grad()
def net_process(model, image):
    input = image.cuda()
    output = model(input)["pred"]
    return output


if __name__ == "__main__":
    main()
