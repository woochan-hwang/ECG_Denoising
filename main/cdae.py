# Convolutional denoising autoencoder (2 layers)
# Written by Woochan H.

# In this model I changed the denoising objective to just reconstruct the ecg
# without the clean acc as in previous models.

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
data = Data('Convolutional Autoencoder', 'flexion_extension', 1)

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

# Acc data modified to fit that of noisy emg
acc_dat = np.array(list(acc_dat[:, 0:6000].transpose())*int(108*data.opened_emg/data.opened_acc)).transpose()
# Clean generated from gaussian dist, N(0, 0.05)
clean_acc = np.random.randn(np.shape(acc_dat)[0], np.shape(acc_dat)[1]) *0.05

# Add ACC data onto clean/noisy ecg data
input_dat = np.vstack((noisy_ecg, acc_dat))
label_dat = np.vstack((clean_ecg, clean_acc))

# Note Use of data_form = 2, which gives a 2D output for each training sample
input_dat = data.reformat(input_dat, feature_len = 200, data_form = 2)
label_dat = data.reformat(label_dat, feature_len = 200, data_form = 2)
print("Input Data shape: {}".format(np.shape(input_dat)))
print("Label Data shape: {}".format(np.shape(label_dat)))

train_set, test_set = data.data_splitter(input_dat, label_dat, shuffle = True, ratio = 4)
print(np.shape(train_set))
print(np.shape(test_set))

print("CUDA: {}".format(data.cuda))
print("Step 0: Data Import Done")

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
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=2), # B,C,H,W = b, 16, 4, 200
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # b, 16, 2, 100
            nn.Conv2d(16, 8, 3, stride=2, padding=1), # b, 8, 2, 100
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1) # b, 8, 1, 50
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2), # b, 16, 2, 100
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1), # b, 8, 4, 200
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=1, padding=1), # b, 1, 4, 200
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

print("Step 1: Model Setup Done")

# Setting of Loss function and optimizer
CAE = ConvAutoEncoder()
optimizer = torch.optim.Adam(CAE.parameters(), lr=LR, weight_decay=1e-5)
loss_func = nn.MSELoss()
train_loss, test_loss = [], []


def save_model(save_name, optim, loss_f, lr, epoch = EPOCH):
    dir = '{}/Trained_Params/{}/{}_{}'.format(data.filepath, data.model, save_name, epoch)
    if not os.path.exists(dir):
        os.makedirs(dir)
    CAE.cpu()
    data.cuda_off()
    torch.save({'data_setting': data,
                'state_dict': CAE.state_dict(),
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
    # Generates mini_batchs for training. Loads data for testing.
    train_loader = loader.DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)
    t_x, t_y = Variable(test_set[:,0:1,:,:]), Variable(test_set[:,1:2,:,:])
    print("t_x", t_x.size())

    # Moves data and model to gpu if available
    if torch.cuda.is_available() == True:
        CAE.cuda()
        t_x = t_x.cuda()
        t_y = t_y.cuda()

    print("Step 2: Model Training Start")

    for epoch in range(EPOCH):
        for step, train_data in enumerate(train_loader):

#            print("train_data", train_data.size())

            if torch.cuda.is_available() == True:
                b_x = Variable(train_data[:,0:1,:,:]).cuda()
                b_y = Variable(train_data[:,1:2,:,:]).cuda()
            else:
                b_x = Variable(train_data[:,0:1,:,:])
                b_y = Variable(train_data[:,1:2,:,:])

#            print("b_x:", b_x.size())
            de = CAE(b_x)

#            print("de:", de.size())
            loss = loss_func(de, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluates current model state on test data set
        pred_y = CAE(t_x)
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
    threshold_train = 100*min(train_loss)
    threshold_test = 100*min(test_loss)
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
