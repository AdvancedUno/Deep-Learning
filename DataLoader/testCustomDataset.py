import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from customDataset import CustomDatasetForCSV


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



dataset = CustomDatasetForCSV(csv_file= 'test.csv', root_dir='test/', transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [100, 10])
train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)
