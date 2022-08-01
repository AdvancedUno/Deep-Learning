import cv2
from matplotlib import pyplot as plt
from utils import plot_examples
import numpy as np
import albumentations as A
from PIL import Image

image = Image.open("images/test.jpeg")
mask = Image.open("images/mask.jpeg")


transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p = 0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5),
            ], p =1.0
        ),
    ]
)

images_list = [image]
image = np.array(image)

mask = np.array(mask)
for i in range(15):
    augmentations = transform(image=image, mask =mask)
    augmented_img = augmentations["image"]
    augmented_mask = augmentations["mask"]
    images_list.append(augmented_img)
    images_list.append(augmented_mask)



