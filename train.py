from models import CycleGAN
from datasets import Dataset

import torch 



TOTAL_EPOCHS = 200
DECAY_START_EPOCH = 100

cycleGAN = CycleGAN().cuda()
# try:
#     cycleGAN.load_state_dict(torch.load('../models/horse2zebra_cycleGAN.pth'))
#     print('Model was loaded')
# except:
#     print('Could not load model')

# lr_scheduler, starts to linearly decay learning rate at DECAY_START_EPOCH
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(cycleGAN.optimizer_G,
                                                        lr_lambda=(lambda epoch: 1 - max(0, epoch - DECAY_START_EPOCH)
                                                        /(TOTAL_EPOCHS - DECAY_START_EPOCH)))
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(cycleGAN.optimizer_D,
                                                        lr_lambda=(lambda epoch: 1 - max(0, epoch - DECAY_START_EPOCH)
                                                        /(TOTAL_EPOCHS - DECAY_START_EPOCH)))



train_dataset = Dataset('../facades/trainA', '../facades/trainB') #hardcoded atm
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

for epoch in range(1, TOTAL_EPOCHS+1):
    
    for i, (img_A, img_B) in enumerate(train_data_loader, 1):
        img_A, img_B = img_A.cuda(), img_B.cuda()
        cycleGAN.optimize_parameters(img_A, img_B)
        
            
        if i % 200 == 0:
            torch.save(cycleGAN.state_dict(), '../models/facades_cycleGAN.pth') #hardcoded also
            cycleGAN.print_losses()
            print(f'epoch: {epoch}, batch: {i} DONE')
            print('MODEL SAVED')
            

    lr_scheduler_G.step()
    lr_scheduler_D.step()
    print(f'END OF EPOCH, setting learning rate to {lr_scheduler_G.get_last_lr()[0]:e}')
        
        
        
        