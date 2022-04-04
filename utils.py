import random
import numpy as np
import torch


def tensor2array(tensor):
    if tensor.is_cuda:
        tensor = tensor.detach().cpu()
    else:
        tensor = tensor.detach()
        
    np_arr = tensor.squeeze().permute(1,2,0).numpy()
    np_arr = ((np_arr + 1)/2*255.0).astype(np.uint8)
    return np_arr




class ImageBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.images = []
        self.num_images = 0
        
    def query(self, images):
        return_images = []        
        for image in images:
            image = torch.unsqueeze(image.detach(), 0)
            
            if self.num_images < self.buffer_size:
                self.images.append(image)
                return_images.append(image)
                self.num_images += 1
                
            else:
                if random.uniform(0, 1) > 0.5:
                    random_index = random.randint(0, self.buffer_size-1)
                    old_image = self.images[random_index].clone()
                    self.images[random_index] = image
                    return_images.append(old_image)
                    
                else:
                    return_images.append(image)
                
        return torch.cat(return_images, 0)