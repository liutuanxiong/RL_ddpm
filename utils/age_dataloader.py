import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from pathlib import Path
import numpy as np

""""""
# Define the transformations
# transform = transforms.Compose([
#     transforms.RandomRotation(degrees=10),
#     transforms.RandomCrop(512),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
# ])
#对于多个图像进行随机操作，随机图像增强
class AgeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # self.transform = transform
        self.directory=Path(root_dir)
        self.image_folders = [folder.name for folder in self.directory.iterdir()]
        #listdir对应路径下所有文件与文件的名称，os.path.isdir(os.path.join(root_dir, folder)表示是否为文件名称

    def __len__(self):
        return len(self.image_folders)

    def __getitem__(self, idx):
        source_image_name= self.image_folders[idx]
        folder_path = self.root_dir

        # # Get the list of image filenames in the folder
        # image_filenames = [f"{i}.jpg" for i in range(0, 101, 10)]
        image_filenames = self.image_folders

        # Pick two random assets from the folder
        target_image_name = random.sample(image_filenames, 1)[0]
        # source_image_name, target_image_name = '20.jpg', '80.jpg'
        """"""

        source_age = int(Path(source_image_name).stem[-2:]) / 100
        target_age = int(Path(target_image_name).stem[-2:]) / 100

        # Randomly select two assets from the folder
        source_image_path = os.path.join(folder_path, source_image_name)
        target_image_path = os.path.join(folder_path, target_image_name)
        '''
        source_image = Image.open(source_image_path).convert('RGB')
        target_image = Image.open(target_image_path).convert('RGB')
        #打开图像转换为RGB通道

        # Apply the same random crop and augmentations to both assets# 对两个资产应用相同的随机裁剪和增强
        if self.transform:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
            torch.manual_seed(seed)
            source_image = self.transform(source_image)
            torch.manual_seed(seed)
            target_image = self.transform(target_image)
        '''
        source_image= np.load(source_image_path)
        target_image = np.load(target_image_path)
        origin_s=source_image
        origin_t=target_image
        source_image = np.expand_dims(source_image, axis=0)
        target_image = np.expand_dims(target_image, axis=0)
        # source_image = np.repeat(source_image0, repeats=3, axis=0)
        # target_image = np.repeat(target_image0, repeats=3, axis=0)
        # source_image = np.transpose(source_image, (1, 2, 0))
        # target_image = np.transpose(target_image, (1, 2, 0))
        source_image =  torch.tensor(source_image)
        target_image =  torch.tensor(target_image)
        source_image = source_image.to(torch.float32)
        target_image =target_image.to(torch.float32)
        # source_image = source_image.permute(2,0,1)
        # target_image = target_image.permute(2,0,1)
        # source_image = transforms.Resize(input_size, interpolation=Image.NEAREST, antialias=True)(source_image)  # 裁剪为目标尺寸
        # target_image = transforms.Resize(input_size, interpolation=Image.NEAREST,antialias=True)(target_image)
        origin_t = np.expand_dims(origin_t, axis=0)
        origin_t = torch.tensor(origin_t)
        origin_t = origin_t.to(torch.float32)
        source_age_channel = torch.full_like(source_image[:1, :, :], 0)
        target_age_channel = torch.full_like(source_image[:1, :, :],0)
        # source_age_channel_origin = torch.full_like(origin_t[:1, :, :], source_age)
        # target_age_channel_origin = torch.full_like(origin_t[:1, :, :], target_age)
        # Concatenate the age channels with the source_image
        source_image = torch.cat([source_image, source_age_channel, target_age_channel], dim=0)
        # origin_t = torch.cat([origin_t, source_age_channel_origin, target_age_channel_origin], dim=0)
        return source_image,source_age*100