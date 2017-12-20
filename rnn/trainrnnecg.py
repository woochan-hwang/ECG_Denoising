# Train Deep Recurrent Denoising Autoencoder (DRDA) _ PyTorch
# Based on works of AL Maas, "Recurrent Neural Networks for Noise Reduction in Robust ASR"
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/

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
EPOCH = 10
LR = 0.002
INPUT_SIZE = 100

# Autoencoder Model Structure
class DRDA(nn.Module):
    def __init__(self):
        super(DRDA, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=512,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 100),
        )

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.decoder(r_out[:, -1, :])
        return out

print("Step 1: Model Setup & Data Import Done")


# Setting of Loss function and optimizer
drda = DRDA()
print(drda)
optimizer = torch.optim.Adam(drda.parameters(), lr=LR)
loss_func = nn.MSELoss()
train_loss = []
print("Step 2: Loss function & Optimizer Done")


# Train the model
for epoch in range(EPOCH):
    print("Loop working")
    decoded = drda(Tensor_Train[epoch+1,:])
    print('Epoch: ', epoch, ' | ', decoded)

    loss = loss_func(decoded, Tensor_Label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])
    train_loss.append((epoch, loss.data[0]))

print("Step 3: Model Training Finished")

# Save trained Parameters
# FileName = /ae_att_lossfunc_epoc.pt
torch.save(DRDA.rnn, '/Users/WoochanH/python/ecgproject/main/trainedparams/ae_encoder_MSE_10000_Tanh_pretrain.pt')
torch.save(DRDA.decoder, '/Users/WoochanH/python/ecgproject/main/trainedparams/ae_decoder_MSE_10000_Tanh_pretrain.pt')
np.savez('train_loss.npz',train_loss)

print("Model Saved")
