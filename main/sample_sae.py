# Sample Stacked denoising autoencoder (2 layers)
# Note use of chicken_selects module and how to process data imports
# Written by Woochan H. (Last modified: 26/03/18)

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.utils.data as loader
from chicken_selects import *

import matplotlib
if str(input("x11-backend?(y/n): ")) == 'y':
    matplotlib.use('GTKAgg')
    print("GTKAgg backend in use")
import matplotlib.pyplot as plt

# Object Data('model type', 'motion', noiselevel, cuda = False)
data = Data('Stacked Autoencoder', 'flexion_extension', 1)

if torch.cuda.is_available() == True:
    data.cuda_on()
else:
    print("Cuda Not Detected")

# Specify directory if you have changed folder name / dir
data.set_ecg_filepath()
data.set_emg_filepath()
data.set_acc_filepath()

# Call data into numpy array format. Check soure code for additional input specifications
clean_ecg = data.pull_all_ecg()
noisy_ecg = data.pull_all_emg()
acc_dat = data.pull_all_acc()

print(data.opened_acc)
print(data.opened_emg)
print(type(data.opened_emg))

# Acc data modified to fit that of noisy emg
acc_dat = np.array(list(acc_dat[:, 0:6000].transpose())*int(108*data.opened_emg/data.opened_acc)).transpose()
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

train_set, test_set = data.data_splitter(input_dat, label_dat, shuffle = True, ratio = 4)

print("CUDA: {}".format(data.cuda))
print("Step 0: Data Import Done")

#input_dat = data.undo_reformat(input_dat)
#label_dat = data.undo_reformat(label_dat)

if str(input("Continue(y/n)?: ")) == 'n':
    quit()

# Hyper Parameters
EPOCH = int(input("Epochs?: "))
LR = float(input("Learning rate?: "))
BATCH_SIZE = int(input("Batch size?: "))

# Generate tensors for training / testing
train_set = torch.from_numpy(train_set).float()
test_set = torch.from_numpy(test_set).float()


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
train_loss, test_loss = [], []


def save_model(save_name, optim, loss_f, lr, epoch = EPOCH):
    dir = '{}/Trained_Params/{}/{}_{}'.format(data.filepath, data.model, save_name, epoch)
    if not os.path.exists(dir):
        os.makedirs(dir)
    SAE.cpu()
    data.cuda_off()
    torch.save({'data_setting': data,
                'state_dict': SAE.state_dict(),
                'epoch': epoch,
                'optimizer': optim,
                'loss_function': loss_f,
                'learning_rate': lr
                },
               dir + '/model.pth')
    np.save(dir + '/trainloss.npy',train_loss)
    np.save(dir + '/testloss.npy',test_loss)
    print("Step 3: Model Saved")


# Train the model
try:
    # Moves data and model to gpu if available
    if data.cuda == True:
        SAE.cuda()
        train_set.cuda()
        test_set.cuda()

    # Generates mini_batchs for training. Loads data for testing.
    train_loader = loader.DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)
    t_x, t_y = Variable(test_set[:,0,:]).cuda(), Variable(test_set[:,1,:]).cuda()

    print("Step 2: Model Training Start")

    for epoch in range(EPOCH):
        for step, train_data in enumerate(train_loader):

            b_x = Variable(train_data[0]).cuda()
            b_y = Variable(train_data[1]).cuda()

            en1, de1, en2, de2 = SAE(b_x)

            loss = loss_func(de2, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluates current model state on test data set
        _, _, _, pred_y = SAE(t_x)
        loss_test_set = loss_func(pred_y, t_y)
        print('Epoch: {} | train loss: {:.4f} | test loss: {:.4f}'.format(epoch+1, loss.data[0], loss_test_set.data[0]))
        train_loss.append(loss.data[0])
        test_loss.append(loss_test_set.data[0])

    print("Step 2: Model Training Finished")

    # Save trained Parameters
except KeyboardInterrupt:

    if str(input("Save Parameters?(y/n): ")) == 'y':
        save_name = str(input("Save parameters as?: ")) + '_Interrupted'
        save_model(save_name, 'Adam', 'MSELoss', LR)
    else:
        print("Session Terminated. Parameters not saved")

else:
    if str(input("Save Parameters?(y/n): ")) == 'y':
        save_name = str(input("Save parameters as?: "))
        save_model(save_name, 'Adam', 'MSELoss', LR)
    else:
        print("Parameters not saved")

    # Plot Loss
    threshold_train = 20*min(train_loss)
    threshold_test = 20*min(test_loss)
    for i in range(len(train_loss)):
        if train_loss[i] > threshold_train:
            train_loss[i] = threshold_train
        if test_loss[i] > threshold_test:
            test_loss[i] = threshold_test
    plt.figure(figsize = (10,4));
    plt.plot(train_loss, color='k', linewidth=0.4, linestyle='-', label = 'train_set loss');
    plt.plot(test_loss, color='b', linewidth=0.4, linestyle='-', label = 'test_set loss')
    plt.legend(loc = 2);
    plt.title("Training Loss({} | {} | LR:{})".format(data.model, data.motion, LR));
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.show()
