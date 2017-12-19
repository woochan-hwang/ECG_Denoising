import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from datasheet import AE_Input_train, AE_Input_test, AE_Label
import matplotlib.pyplot as plt

# Load test variable and trained parameters
encoder1 = torch.load('/Users/WoochanH/python/ecgproject/main/trainedparams/sae2_GnB_encoder1_MSE_5000_Tanh_pretrain.pt')
decoder1 = torch.load('/Users/WoochanH/python/ecgproject/main/trainedparams/sae2_GnB_decoder1_MSE_5000_Tanh_pretrain.pt')
encoder2 = torch.load('/Users/WoochanH/python/ecgproject/main/trainedparams/sae2_GnB_encoder2_MSE_5000_Tanh_pretrain.pt')
decoder2 = torch.load('/Users/WoochanH/python/ecgproject/main/trainedparams/sae2_GnB_decoder2_MSE_5000_Tanh_pretrain.pt')

Tensor_trainset = Variable(torch.from_numpy(AE_Input_train).float())
Tensor_testset = Variable(torch.from_numpy(AE_Input_test).float())
Tensor_label = Variable(torch.from_numpy(AE_Label).float())

key = input("Will you test on testset(1) or trainset(2)?: ")
batch = input("Which batch will you plot?: ")

if key == 1:
    Input_tensor = Tensor_testset
    dirty = np.reshape(AE_Input_test, (2400,270))
    dirtyarray = dirty[batch]
elif key == 2:
    Input_tensor = Tensor_trainset
    dirty = np.reshape(AE_Input_train, (43200, 270))
    dirtyarray = dirty[batch]
else:
    print("Invalid input")

# Run Model on Input data
En1 = encoder1(Input_tensor)
De1 = decoder1(En1)
En2 = encoder2(De1)
De2 = decoder2(En2)


# Reshape into numpy array
denoised = De2.data[batch]
denoisedarray = denoised.numpy()


# Plot to compare
plt.figure(figsize = (10,4))
plt.plot(denoisedarray, color='blue', linewidth=0.2, linestyle='-', label = 'Denoised');
plt.plot(dirtyarray, color='k', linewidth=0.4, linestyle='-', label = 'Noisy');
plt.legend(loc = 2);
plt.title("Comparison between original and denoised(Gaussian and Baseline)");

plt.show()

# Plot Loss over EPOCH
train_loss = np.load('sae_15000_train_loss.npy')
plt.figure(figsize = (10,4))
plt.plot(train_loss[1000:14999,1], color='k', linewidth=0.4, linestyle='-', label = 'MSE loss');
plt.legend(loc = 2);
plt.title("Loss over Epoch");

plt.show()
