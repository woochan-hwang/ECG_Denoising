# Evaluating Convolutional denoising autoencoder (2 layers)
# Written by Woochan H.

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from chicken_selects import *

inputdat = np.genfromtxt("/Users/WoochanH/Desktop/Year 4/Group Project/codebase/ECG_Denoising/main/LaureBreastFilteredLowSR.csv", delimiter = ',')[0:900]
print("Step 1: Data Imported")
print(inputdat)

# Define Model Structure. Should be same as one used for Training
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Input = b, 1, 4, 300
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (4,3), stride=1, padding=(0,1)), # b, 8, 1, 300
            nn.Tanh(),
            nn.MaxPool2d((1,2), stride=2), # b, 8, 1, 150
            nn.Conv2d(8, 4, 3, stride=1, padding=1), # b, 8, 1, 150
            nn.Tanh(),
            nn.MaxPool2d((1,2), stride=2) # b, 4, 1, 75
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=(0,1)), # b, 8, 1, 150
            nn.Tanh(),
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=(0,1), output_padding=1), # b, 8, 4, 300
            nn.Tanh(),
            nn.ConvTranspose2d(8, 1, 3, stride=1, padding=1), # b, 1, 4, 300
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Loads model and model parameters
params_dir = "/Users/WoochanH/Desktop/Year 4/Group Project/codebase/ECG_Denoising/main/Trained_Params/Convolutional Autoencoder/case1_h2_Interrupted_3000/model.pth"
model_params = torch.load(params_dir)

mymodel = ConvAutoEncoder()
mymodel.load_state_dict(model_params['state_dict'])
print("Step 2: Model Structure and Parameter Loaded")

# Additional information about trained instance
trained_data = model_params['data_setting']
trained_epochs = model_params['epoch']
trained_optim = model_params['optimizer']
trained_lossf = model_params['loss_function']
trained_lr = model_params['learning_rate']

# Call data into numpy array format. Check soure code for additional input specifications
def format2(input_arr, feature_len):
    l = int(np.shape(input_arr)[1])
    k = int(feature_len)
    sample_num = int(l / feature_len)
    output = np.zeros((sample_num, 4, k))
    for i in range(0,int(sample_num)):
        output[i] = input_arr[:, k*(i):k*(i+1)]
    return np.array(output)

def undo_format2(npver, feature_len):
    k = int(feature_len)
    sample_num = np.shape(npver)[0]
    sig_len = sample_num*k
    output = np.zeros((4, sample_num*k))
    for i in range(1,sample_num):
        output[:, k*i:k*(i+1)] = npver[i]
    return np.array(output)

clean_acc = np.random.randn(3, 900)*0.05 # N(0,0.05)


print(np.shape(inputdat))
testdata = np.expand_dims(inputdat,axis =1)
testdata = testdata.transpose()
print(testdata)
print(type(testdata[0,10]))
print(np.shape(testdata))
print(np.shape(clean_acc))
data = np.vstack((testdata, clean_acc))
print(np.shape(data))
reformed = format2(data, 300)
print(np.shape(reformed))

t_input = np.expand_dims(reformed, axis = 1)
print(np.shape(t_input))

# Generate tensors for training / validation
t_x = Variable(torch.from_numpy(t_input).float())

# Evaluate model on data
pred_t_y = mymodel(t_x)
print("Step 3: Model Evaluation Finished")

print("Step 4: Plotting Results")
# Change tensor output to numpy and undo reformatting
pred_t_y = undo_format2(pred_t_y.data.numpy(), 300)

# Plot Results
plot = str(input("Plot results(y/n)?: "))
print("Available data lenght: {}".format(np.shape(pred_t_y)[1]))

while plot == 'y':
    t0 = int(input("Plotting | Start time?: "))
    tf = t0 + int(input("Plotting | Duration?: "))

    denoised = pred_t_y[0,t0:tf]
    original = testdata[0,t0:tf]

    fig, (ax1, ax2) = plt.subplots(2, sharey=False)
    ax1.plot(denoised, color='b', linewidth=0.4, linestyle='-')
    ax1.set(title='Model Output', ylabel='Denoised')

    ax2.plot(original, color='b', linewidth=0.4, linestyle='-')
    ax2.set(xlabel ='time(s, {} to {})'.format(t0,tf), ylabel='Original')

    plt.show()

    plot = str(input("Plot again(y/n)?: "))

print("Session Terminated")
