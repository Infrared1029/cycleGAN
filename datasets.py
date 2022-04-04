import os
import random


import torch 
from PIL import Image
from torchvision import transforms




# len(os.listdir('apple2orange/apple2orange/trainB'))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_A_path, images_B_path, training=True):
        self.process_image = transforms.Compose(
            [
            transforms.Resize(size=(286, 286)) if training else transforms.Resize(size=(256, 256)),
            transforms.RandomCrop(size=(256, 256)) if training else transforms.Lambda(lambda img: img),
            transforms.RandomHorizontalFlip() if training else transforms.Lambda(lambda img: img),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             
        ]
        )
        
        self.images_A = [os.path.join(images_A_path, img) for img in 
                          os.listdir(images_A_path)]
        self.images_B = [os.path.join(images_B_path, img) for img in 
                          os.listdir(images_B_path)]
        
        
        self.size_A = len(self.images_A)
        self.size_B = len(self.images_B)
        
                              
    def __getitem__(self, index):
        index_A = index % self.size_A
        index_B = random.randint(0, self.size_B-1)
        image_A_path, image_B_path = self.images_A[index_A], self.images_B[index_B] 
        image_A, image_B = Image.open(image_A_path).convert('RGB'), Image.open(image_B_path).convert('RGB')
        return self.process_image(image_A), self.process_image(image_B)
    
    def __len__(self):
        return max(self.size_A, self.size_B)
        
        