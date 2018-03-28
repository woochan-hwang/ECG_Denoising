# Sample Stacked denoising autoencoder (2 layers)
# Note use of chicken_selects module and how to process data imports
# Written by Woochan H. (Last modified: 26/03/18)

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from chicken_selects import *

import matplotlib
if str(input("x11-backend?(y/n): ")) == 'y':
    matplotlib.use('GTKAgg')
    print("GTKAgg backend in use")
import matplotlib.pyplot as plt

# Object Data('model type', 'motion', noiselevel, cuda = False)
data = Data('Stacked Autoencoder', 'flexion_extension', 1)

if torch.cuda.is_available == True:
    data.cuda_on()

# Specify directory if you have changed folder name / dir
data.set_ecg_filepath()
data.set_emg_filepath()
data.set_acc_filepath()

# Call data into numpy array format. Check soure code for additional input specifications
clean_ecg = data.pull_all_ecg(t0 = 0, tf = 30000)
noisy_ecg = data.pull_all_emg()
acc_dat = data.pull_all_acc()

# Acc data modified to fit that of noisy emg
acc_dat = np.array(list(acc_dat[:, 0:3000].transpose())*140).transpose()
# Clean generated from gaussian dist, N(0, 0.05)
clean_acc = np.random.randn(np.shape(acc_dat)[0], np.shape(acc_dat)[1]) *0.05

# Add ACC data onto clean/noisy ecg data
input_dat = np.vstack((noisy_ecg, acc_dat))
label_dat = np.vstack((clean_ecg, clean_acc))

# Reformat to shape that can be imported to neural net
input_dat = data.reformat(input_dat, feature_len = 200, data_form = 1)
label_dat = data.reformat(label_dat, feature_len = 200, data_form = 1)
print("Input Data shape: {}".format(np.shape(input_dat)))
print("Label Data shape: {}".format(np.shape(label_dat)))
print("CUDA: {}", data.cuda)
print("Step 0: Data Import Done")

#input_dat = data.undo_reformat(input_dat)
#label_dat = data.undo_reformat(label_dat)

if str(input("Continue(y/n)?: ")) == 'n':
    quit()

# Change numpy array to tensor variable
Tensor_Input = Variable(torch.from_numpy(input_dat).float())
Tensor_Label = Variable(torch.from_numpy(label_dat).float())

# Hyper Parameters
EPOCH = int(input("Epochs?: "))
LR = float(input("Learning rate?: "))
BATCH_SIZE = int(input("Batch size?: "))

# Autoencoder Model Structure
class StackedAutoEncoder(nn.Module):
    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Linear(800, 270),
            nn.Tanh(),
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
            nn.Linear(135, 270),
            nn.Tanh(),
            nn.Linear(270, 800)
        )

    def forward(self, x):
        en1 = self.encoder1(x)
        de1 = self.decoder1(en1)
        en2 = self.encoder2(de1)
        de2 = self.decoder2(en2)
        return en1, de1, en2, de2

print("Step 1: Model Setup Done")

# Setting of Loss function and optimizer
SAE = StackedAutoEncoder()
optimizer = torch.optim.Adam(SAE.parameters(), lr=LR)
loss_func = nn.MSELoss()
train_loss = []


def save_params(save_name):
    dir = '{}/Trained_Params/{}/{}'.format(data.filepath, data.model, save_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(SAE.encoder1, dir + '/encoder1_{}.pt'.format(EPOCH))
    torch.save(SAE.decoder1, dir + '/decoder1_{}.pt'.format(EPOCH))
    torch.save(SAE.encoder2, dir + '/encoder2_{}.pt'.format(EPOCH))
    torch.save(SAE.decoder2, dir + '/decoder2_{}.pt'.format(EPOCH))
    np.save(dir + '/trainloss_{}.npy'.format(EPOCH),train_loss)
    print("Step 3: Model Saved")


# Train the model
try:
    if torch.cuda.is_available() == True:
        SAE = SAE.cuda()
        Tensor_Input = Tensor_Input.cuda()
        Tensor_Label = Tensor_Label.cuda()

    print("Step 2: Model Training Start")

    for epoch in range(EPOCH):

        en1, de1, en2, de2 = SAE(Tensor_Input)

        loss = loss_func(de2, Tensor_Label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: ', epoch + 1, '| train loss: %.4f' % loss.data[0])
        train_loss.append(loss.data[0])

    print("Step 2: Model Training Finished")

    # Plot Loss
    if max(train_loss) > min(train_loss)*100:
        train_loss = train_loss[int(len(train_loss)/100):end]
    plt.figure(figsize = (10,4));
    plt.plot(train_loss, color='k', linewidth=0.4, linestyle='-', label = 'MSE loss');
    plt.legend(loc = 2);
    plt.title("Training Loss({} | {} | LR:{})".format(data.model, data.motion, LR));
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.show()

    # Save trained Parameters
    if str(input("Save Parameters?(y/n): ")) == 'y':
        save_name = str(input("Save parameters as?: ")) + '_Interrupted'
        save_params(save_name)
        print("End of Session")
    else:
        print("Session Terminated. Parameters not saved")


except KeyboardInterrupt:

    if str(input("Save Parameters?(y/n): ")) == 'y':
        save_name = str(input("Save parameters as?: ")) + '_Interrupted'
        save_params(save_name)
    else:
        print("Session Terminated. Parameters not saved")
