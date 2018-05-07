# Convolutional denoising autoencoder (2 layers)
# Written by Woochan H.

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy
import torch.utils.data as loader
from chicken_selects import *
import matplotlib.pyplot as plt

print(torch.__version__)

#noiselevel = int(input("EMG noise level?: "))
noiselevel = 1
# Object Data('model type', 'motion', noiselevel, cuda = False)
data = Data('Convolutional Autoencoder', 'mixed', noiselevel = noiselevel)

# Specify directory if you have changed folder name / dir
data.set_ecg_filepath()
data.set_emg_filepath(filepath = 'emgdata_final')
data.set_acc_filepath(filepath = 'accdata_final')

# Call data into numpy array format. Check soure code for additional input specifications
clean_ecg = data.pull_all_ecg(tf = 240000) # Total of 14 recordings
emg_noise = data.pull_all_emg(tf = 10000) # 10,000 data points * 3 motions * 2 trials * 4 subjects
acc_dat = data.pull_all_acc(tf = 10000) # equiv to emg

# Remove mean, normalize to range (-1,1), adjust for noiselevel setting.
clean_ecg[0,:] -= np.mean(clean_ecg[0,:])
clean_ecg[0,:] = clean_ecg[0,:]/max(abs(clean_ecg[0,:]))

emg_noise[0,:] -= np.mean(emg_noise[0,:])
emg_noise[0,:] = (emg_noise[0,:]/max(abs(emg_noise[0,:])))*data.noiselevel

for i in range(0,3):
    acc_dat[i,:] -= np.mean(acc_dat[i,:])
    acc_dat[i,:] = (acc_dat[i,:]/max(abs(acc_dat[i,:])))*float(data.noiselevel**(0.5))
# Repeat the emg noise to each ecg recording
repeats = np.shape(clean_ecg)[1]/np.shape(emg_noise)[1]
emg_noise = np.array(list(emg_noise.transpose())*int(repeats)).transpose()
acc_dat = np.array(list(acc_dat.transpose())*int(repeats)).transpose()

clean_acc = np.random.randn(np.shape(acc_dat)[0], np.shape(acc_dat)[1])*0.05 # N(0,0.05)

# Generate noisy ECG by adding EMG noise
noisy_ecg = clean_ecg + emg_noise

# Add ACC data onto clean/noisy ecg data
input_dat = np.vstack((noisy_ecg, acc_dat))
label_dat = np.vstack((clean_ecg, clean_acc))

# Note Use of data_form = 2, which gives a 2D output for each training sample
input_dat = data.reformat(input_dat, feature_len = 300, data_form = 2)
label_dat = data.reformat(label_dat, feature_len = 300, data_form = 2)
print("Input Data shape: {}".format(np.shape(input_dat)))
print("Label Data shape: {}".format(np.shape(label_dat)))

train_set, val_set = data.data_splitter(input_dat, label_dat, shuffle = True, ratio = 4)

print("Step 0: Data Import Done")

#if str(input("Continue(y/n)?: ")) == 'n':
#    quit()

# Hyper Parameters
#EPOCH = int(input("Epochs?: "))
#LR = float(input("Learning rate?: "))
#BATCH_SIZE = int(input("Batch size?: "))
EPOCH = 5000
LR = 0.0003
BATCH_SIZE = 128

cuda = True if torch.cuda.is_available() else False
print(cuda)

# Generate tensors for training / validation
train_set = torch.from_numpy(train_set).float()
val_set = torch.from_numpy(val_set).float()

# Autoencoder Model Structure
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Zero padding is almost the same as average padding in this case
        # Input = b, 1, 4, 300
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (4,3), stride=1, padding=(0,1)), # b, 8, 1, 300
            nn.Tanh(),
            nn.MaxPool2d((1,2), stride=2), # b, 8, 1, 150
            nn.Conv2d(8, 4, 3, stride=1, padding=1), # b, 4, 1, 150
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
        # Evaluates confidence based on bottleneck features
        self.critic = nn.Sequential(
            nn.Conv2d(4, 1, 3, stride=1, padding=1), # b, 1, 1, 75
            nn.Tanh(),
            nn.Linear(75,30)
        )

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        c = self.critic(x)
        return y, c

print("Step 1: Model Setup Done")

# Initialize network
CAE1 = ConvAutoEncoder() # Ensemble part 1 : Smoothing reconstruction
CAE2 = ConvAutoEncoder() # Ensemble part 2 : Peak Reconstruction
if cuda:
    CAE.cuda()
L1Loss = nn.L1Loss()

# Define custom loss function based on morphological ROI
def customloss(x, y):
    # Laplacian of Gaussian filter to define ROI
    LoG = scipy.ndimage.filters.gaussian_laplace(x, sigma = 5)
    print(LoG)
    print(LoG.shape())
    loss = "ROI1"
    return loss

train_loss1, val_loss1, trainloss2, val_loss2, train_loss3, val_loss3 = [],[],[],[],[],[]

# Set optimizer
optimizer1 = torch.optim.Adam(CAE1.parameters(), lr=LR, weight_decay=1e-5)
optimizer2 = torch.optim.Adam(CAE2.parameters(), lr=LR, weight_decay=1e-5)


def save_model(save_name, optim, loss_f, lr, epoch = EPOCH):
    dir = '{}/Trained_Params/{}/{}_{}'.format(data.filepath, data.model, save_name, epoch)
    if not os.path.exists(dir):
        os.makedirs(dir)
    CAE1.cpu(), CAE2.cpu()
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
    np.save(dir + '/valloss.npy',val_loss)
    print("Step 3: Model Saved")


# Train the model
try:
    # Generates mini_batchs for training. Loads data for validation.
    train_loader = loader.DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)
    v_x, v_y = Variable(val_set[:,0:1,:,:]), Variable(val_set[:,1:2,:,:])

    # Moves data and model to gpu if available
    if cuda:
        v_x, v_y = v_x.cuda(), v_y.cuda()

    print("Step 2: Model Training Start")

    for epoch in range(EPOCH):
        for step, train_data in enumerate(train_loader):

            b_x = Variable(train_data[:,0:1,:,:]).cuda() if cuda else Variable(train_data[:,0:1,:,:])
            b_y = Variable(train_data[:,1:2,:,:]).cuda() if cuda else Variable(train_data[:,1:2,:,:])

            smooth_recon, confidence1 = CAE1(b_x)
            optimizer1.zero_grad()
            loss1 = customloss(smooth_recon, b_y)
            loss1.backward()
            optimizer1.step()

            peak_recon, confidence2 = CAE2(b_x)
            optimizer2.zero_grad()
            loss2 = customloss(peak_recon, b_y)
            loss2.backward()
            optimizer2.step()

        # Evaluates current model state on val data set
        pred1 = CAE1(v_x)
        loss_val_set1 = customloss(pred1, v_y)
        train_loss1.append(loss1.data[0])
        val_loss1.append(loss_val_set1.data[0])
        print('Epoch: {} | part 1 | train loss: {:.4f} | val loss: {:.4f}'.format(epoch+1, loss.data[0], loss_val_set.data[0]))

        pred2 = CAE2(v_x)
        loss_val_set2 = customloss(pred2, v_y)
        train_loss2.append(loss1.data[0])
        val_loss2.append(loss_val_set1.data[0])
        print('Epoch: {} | part 2 | train loss: {:.4f} | val loss: {:.4f}'.format(epoch+1, loss.data[0], loss_val_set.data[0]))

        # Update both submodels based on ensemble loss every 10 epochs
        if epoch % 10 == 0:
            for step, train_data in enumerate(train_loader):

                b_x = Variable(train_data[:,0:1,:,:]).cuda() if cuda else Variable(train_data[:,0:1,:,:])
                b_y = Variable(train_data[:,1:2,:,:]).cuda() if cuda else Variable(train_data[:,1:2,:,:])

                smooth_recon, confidence1 = CAE1(b_x)
                peak_recon, confidence2 = CAE2(b_x)
                c_ratio1 = confidence1/(confidence1 + confidence2)
                c_ratio2 = confidence2/(confidence1 + confidence2)
                for i in range(len(confidence1)):
                    final_recon[:,:,:,i*10:(i+1)*10] = (smooth_recon[:,:,:,i*10:(i+1)*10]*c_ratio1[:,:,:,i]
                                                        + peak_recon[:,:,:,i*10:(i+1)*10]*c_ratio2[:,:,:,i])

                ensemble_loss = L1Loss(final_recon, b_y)

                optimizer1.zero_grad()
                ensemble_loss.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                ensemble_loss.backward()
                optimizer2.step()

            pred = (CAE1(v_x) + CAE2(v_x))/2
            ensemble_loss_val_set = L1Loss(pred, v_y)
            train_loss3.append(ensemble_loss.data[0])
            val_loss3.append(ensemble_loss_val_set1.data[0])
            print('Ensemble Loss | train loss: {:.4f} | val loss: {:.4f}'.format(loss.data[0], loss_val_set.data[0]))

    print("Step 2: Model Training Finished")

    # Save trained Parameters
except KeyboardInterrupt:

    if str(input("Save Parameters?(y/n): ")) == 'y':
        save_name = str(input("Save parameters as?: ")) + '_Interrupted'
        save_model(save_name, 'Adam', 'L1Loss', LR)
    else:
        print("Session Terminated. Parameters not saved")

else:
    print("entering else statement")
    save_model('Ensemble_cdae', 'Adam', 'L1Loss', LR)
    print(os.listdir(os.getcwd()))
    print(os.getcwd())
