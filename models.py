from utils import tensor2array
from utils import ImageBuffer

import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.functional import F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 use_activation=True,
                use_instance_norm=True,
                padding_mode='reflect'):
        
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode),
            nn.InstanceNorm2d(out_channels) if use_instance_norm else nn.Identity(),
            nn.ReLU(True) if use_activation else nn.Identity()
        
        )
        
    def forward(self, x):
        return self.conv_block(x)
        
        

class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, padding_mode='reflect'):
        super().__init__()
        self.res_block = nn.Sequential(
        ConvBlock(in_channels, in_channels, kernel_size, stride, padding, padding_mode=padding_mode),
#         ConvBlock(in_channels, in_channels, kernel_size, stride, padding, padding_mode=padding_mode),
        ConvBlock(in_channels, in_channels, kernel_size, stride, padding, padding_mode=padding_mode, use_activation=False)    
        
        )
        
    def forward(self, x):
        return x + self.res_block(x)


def init_weights(m):
    class_name = m.__class__.__name__
    if hasattr(m, 'weight') and class_name.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)   

        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)



class Generator(nn.Module):
    def __init__(self, n_resblocks=9):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, padding=3, padding_mode='reflect', stride=1),
            
            #Downsampling layers
            ConvBlock(64, 128, kernel_size=3, padding=1, padding_mode='zeros', stride=2),
            ConvBlock(128, 256, kernel_size=3, padding=1, padding_mode='zeros', stride=2),
            
            #Resblocks
            *[ResBlock(256, kernel_size=3, padding=1, padding_mode='reflect', stride=1) for i in range(n_resblocks)],
            
            #Upsampling layers
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(256, 128, kernel_size=3, padding=1, padding_mode='reflect',stride=1),
 
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(128, 64, kernel_size=3, padding=1, padding_mode='reflect',stride=1),
            
            ConvBlock(64, 3, kernel_size=7, padding=3,  padding_mode='reflect', 
                      stride=1, use_activation=False, use_instance_norm=False),
            
            nn.Tanh()
            
            
        )
        
        
        #init weights from a normal dist N~(0.0, 0.002)
        self.apply(init_weights)
        
        
    def forward(self, x):
        return self.model(x)
    
    

class DiscriminatorConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
        use_activation=True,
    use_instance_norm=True,
    padding_mode='reflect'):

        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode),
            nn.InstanceNorm2d(out_channels) if use_instance_norm else nn.Identity(),
            
            #only difference between generator conv blocks and discriminator
            nn.LeakyReLU(0.2, True) if use_activation else nn.Identity()
        
        )
        
    def forward(self, x):
        return self.conv_block(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            DiscriminatorConvBlock(3, 64, kernel_size=4, padding=1, padding_mode='zeros', stride=2, 
                                   use_instance_norm=False),
            
            DiscriminatorConvBlock(64, 128, kernel_size=4, padding=1, padding_mode='zeros', stride=2),
            DiscriminatorConvBlock(128, 256, kernel_size=4, padding=1, padding_mode='zeros', stride=2),
            
            DiscriminatorConvBlock(256, 512, kernel_size=4, padding=1, padding_mode='zeros', stride=1),
            
            DiscriminatorConvBlock(512, 1, kernel_size=4, padding=1, padding_mode='zeros', stride=1,
                                  use_instance_norm=False, use_activation=False)
            
        
        )
        
        
        #init weights from a normal dist N~(0.0, 0.002)
        self.apply(init_weights)
        
    def forward(self, x):
        return self.model(x)



from itertools import chain

class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.G_AtoB = Generator()
        self.G_BtoA = Generator()
        
        self.D_A = Discriminator()
        self.D_B = Discriminator()
        
        # optimizers
        self.optimizer_G = torch.optim.Adam(chain(self.G_AtoB.parameters(), self.G_BtoA.parameters()),
                                            lr=0.0002, betas=(0.5, 0.999))
        
        self.optimizer_D = torch.optim.Adam(chain(self.D_A.parameters(), self.D_B.parameters()),
                                            lr=0.0002, betas=(0.5, 0.999))
        
        # images buffer
        self.fake_A_buffer = ImageBuffer(50)
        self.fake_B_buffer = ImageBuffer(50)
        
        
        # useful when converting model to GPU
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.register_buffer('real_label', torch.tensor(1.0))
        
        self.lambda_BtoA = 10.0
        self.lambda_AtoB = 10.0
        self.lambda_idt = 0.5
        
        
        
        
        
    def forward(self, images_A, images_B):
#         images_A, images_B = images
        
    

    
        self.images_A = images_A
        self.images_B = images_B
        
        self.fake_B = self.G_AtoB(images_A)
        self.rec_A = self.G_BtoA(self.fake_B)
        
        self.fake_A = self.G_BtoA(images_B)
        self.rec_B = self.G_AtoB(self.fake_A)
        
#         return fake_B, fake_A, rec_B, rec_A
    
    
    
    def generator_backward(self):
        
        
        # adverserial loss
        
        d_B_preds = self.D_B(self.fake_B)
        self.loss_AtoB = F.mse_loss(d_B_preds, self.real_label.expand_as(d_B_preds))
        
        d_A_preds = self.D_A(self.fake_A)
        self.loss_BtoA = F.mse_loss(d_A_preds, self.real_label.expand_as(d_A_preds))
        
        
        #cycle loss
        self.cycle_loss_AtoBtoA = F.l1_loss(self.rec_A, self.images_A) * self.lambda_BtoA
        self.cycle_loss_BtoAtoB = F.l1_loss(self.rec_B, self.images_B) * self.lambda_AtoB
        
        #identity_loss
        self.idt_loss_A = F.l1_loss(self.G_BtoA(self.images_A), self.images_A) * self.lambda_BtoA * self.lambda_idt
        self.idt_loss_B = F.l1_loss(self.G_AtoB(self.images_B), self.images_B) * self.lambda_AtoB * self.lambda_idt
        
        
        
        gen_loss =  (self.loss_AtoB + self.loss_BtoA + self.cycle_loss_BtoAtoB + self.cycle_loss_AtoBtoA +
                     self.idt_loss_A + self.idt_loss_B)
        
        gen_loss.backward()
    
    
    def discriminator_backward(self):
        
        real_A_preds = self.D_A(self.images_A)
        d_A_real_loss = F.mse_loss(real_A_preds, self.real_label.expand_as(real_A_preds))
        
        fake_A = self.fake_A_buffer.query(self.fake_A)
        fake_A_preds = self.D_A(fake_A.detach())
        d_A_fake_loss = F.mse_loss(fake_A_preds, self.fake_label.expand_as(fake_A_preds))
        
        self.d_A_loss = (d_A_fake_loss + d_A_real_loss) * 0.5
        
        self.d_A_loss.backward()
        
        
        
        real_B_preds = self.D_B(self.images_B)
        d_B_real_loss = F.mse_loss(real_B_preds, self.real_label.expand_as(real_B_preds))
        
        
        fake_B = self.fake_B_buffer.query(self.fake_B)
        fake_B_preds = self.D_B(fake_B.detach())
        d_B_fake_loss = F.mse_loss(fake_B_preds, self.fake_label.expand_as(fake_B_preds))
        
        self.d_B_loss = (d_B_fake_loss + d_B_real_loss) * 0.5
        
        self.d_B_loss.backward()

        
        
    
    def optimize_parameters(self, images_A, images_B):
        
#         images_A, images_B = images
        
        self.forward(images_A, images_B)
        
        # update generators
        #########################################
        
        # zero out previous gradients
        self.optimizer_G.zero_grad()
        
        # freeze Discriminators weights
        self.set_requires_grad([self.D_A, self.D_B], False)
        
        # calculate generators' gradients 
        self.generator_backward()
        
        # apply gradient descent
        self.optimizer_G.step()
        
        # update discriminators
        #####################################
        
        # zero out previous gradients
        self.optimizer_D.zero_grad()
        
        # unfreeze Discriminators
        self.set_requires_grad([self.D_A, self.D_B], True)
        
        # calculate discriminators' gradients
        self.discriminator_backward()
        
        # apply gradient descent
        self.optimizer_D.step()

        
        
        
        
    def set_requires_grad(self, nets, requires_grad):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
                
    
    def visualize_results(self):
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        
        axs[0, 0].imshow(tensor2array(self.images_A))
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(tensor2array(self.fake_B))
        axs[0, 1].axis('off')
    
        axs[0, 2].imshow(tensor2array(self.rec_A))
        axs[0, 2].axis('off')
        
        axs[1, 0].imshow(tensor2array(self.images_B))
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(tensor2array(self.fake_A))
        axs[1, 1].axis('off')
        
        axs[1, 2].imshow(tensor2array(self.rec_B))
        axs[1, 2].axis('off')
        
        plt.show()


    def print_losses(self):
        print(f'''gan_AtoB; {self.loss_AtoB}, gan_BtoA: {self.loss_BtoA},
        idtA: {self.idt_loss_A}, cycleAtoBtoA: {self.cycle_loss_AtoBtoA},
        idtB: {self.idt_loss_B}, cycleBtoAtoB: {self.cycle_loss_BtoAtoB},
        ''')
        