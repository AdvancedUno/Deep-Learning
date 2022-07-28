import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customDataset import CustomDatasetForCSV
from torchvision.utils import save_image


transformUno = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])





dataset = CustomDatasetForCSV(csv_file= 'test.csv', root_dir='test/', transform=transforms.ToTensor())

img_num = 0
for img, label in dataset:
    save_image(img, 'img' + str(img_num) + '.png')
    img_num += 1
