# Stacked denoising autoencoder
# A stacked contractive denoising auto-encoder for ECG signal denoising, Xiong et al.

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib as plt
from datasheet import AE_Input_train, AE_Input_test, AE_Label

print("Data imported")

# Variable setting based on data imported from datasheet.py
Tensor_Train = Variable(torch.from_numpy(AE_Input_train).float())
Tensor_Label = Variable(torch.from_numpy(AE_Label).float())

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 15000
LR = 0.0005
#BATCH_SIZE = 32


# Autoencoder Model Structure
class StackedAutoEncoder(nn.Module):
    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Linear(270, 135),
            nn.Tanh(),
            nn.Linear(135, 70),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(70, 135),
            nn.Tanh(),
            nn.Linear(135, 270)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(270, 135),
            nn.Tanh(),
            nn.Linear(135, 70),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(70, 135),
            nn.Tanh(),
            nn.Linear(135, 270)
        )

    def forward(self, x):
        en1 = self.encoder1(x)
        de1 = self.decoder1(en1)
        en2 = self.encoder2(de1)
        de2 = self.decoder2(en2)
        return en1, de1, en2, de2

print("Step 1: Model Setup & Data Import Done")

# Setting of Loss function and optimizer
SAE = StackedAutoEncoder()
optimizer = torch.optim.Adam(SAE.parameters(), lr=LR)
loss_func = nn.MSELoss()
train_loss = []
print("Step 2: Loss function & Optimizer Done")

# Train the model
for epoch in range(EPOCH):

    en1, de1, en2, de2 = SAE(Tensor_Train)

    loss = loss_func(de2, Tensor_Label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])
    train_loss.append((epoch, loss.data[0]))

print("Step 3: Model Training Finished")

# Save trained Parameters
# FileName = /ae_att_lossfunc_epoc.pt
torch.save(SAE.encoder1, '/Users/WoochanH/python/ecgproject/main/trainedparams/sae2_GnB_encoder1_MSE_5000_Tanh_pretrain.pt')
torch.save(SAE.decoder1, '/Users/WoochanH/python/ecgproject/main/trainedparams/sae2_GnB_decoder1_MSE_5000_Tanh_pretrain.pt')
torch.save(SAE.encoder2, '/Users/WoochanH/python/ecgproject/main/trainedparams/sae2_GnB_encoder2_MSE_5000_Tanh_pretrain.pt')
torch.save(SAE.decoder2, '/Users/WoochanH/python/ecgproject/main/trainedparams/sae2_GnB_decoder2_MSE_5000_Tanh_pretrain.pt')
np.save('sae_15000_train_loss.npy',train_loss)

print("Model Saved")

plt.figure(figsize = (10,4))
plt.plot(train_loss[:,1], color='k', linewidth=0.4, linestyle='-', label = 'MSE loss');
plt.legend(loc = 2);
plt.title("Loss over Epoch");

plt.show()

print("End of Session")
