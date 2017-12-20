import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from datasheet import AE_Input_train, AE_Input_test, AE_Label

print("Data imported")

# Variable setting based on data imported from datasheet.py
Tensor_Train = Variable(torch.from_numpy(AE_Input_train).float())
Tensor_Label = Variable(torch.from_numpy(AE_Label).float())

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10000
LR = 0.002
#BATCH_SIZE = 32


# Autoencoder Model Structure
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1000, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1000),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

print("Step 1: Model Setup & Data Import Done")

# Setting of Loss function and optimizer
autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()
train_loss = []
print("Step 2: Loss function & Optimizer Done")

# Train the model
for epoch in range(EPOCH):

    encoded, decoded = autoencoder(Tensor_Label)

    loss = loss_func(decoded, Tensor_Label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])
    train_loss.append((epoch, loss.data[0]))

print("Step 3: Model Training Finished")

# Save trained Parameters
# FileName = /ae_att_lossfunc_epoc.pt
torch.save(autoencoder.encoder, '/Users/WoochanH/python/ecgproject/main/trainedparams/ae_encoder_MSE_10000_Tanh_pretrain1.pt')
torch.save(autoencoder.decoder, '/Users/WoochanH/python/ecgproject/main/trainedparams/ae_decoder_MSE_10000_Tanh_pretrain1.pt')
np.savez('train_loss.npz',train_loss)

print("Model Saved")
