# Convolutional denoising autoencoder for ECGs
# Written by Woochan H.

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.utils.data as loader
from chicken_selects import *
import matplotlib.pyplot as plt

params_dir = '{}/Trained_Params/Convolutional Autoencoder/newdata1_5000/model.pth'.format(os.getcwd())
print(params_dir)
cuda = True if torch.cuda.is_available() else False
print("CUDA: ", cuda)


# Autoencoder Model Structure
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

# Load Trained Params
model_params = torch.load(params_dir)

# Initialize network
CAE = ConvAutoEncoder()
CAE.load_state_dict(model_params['state_dict'])
loss_func = nn.L1Loss()
train_loss, val_loss = [], []
if cuda:
    CAE.cuda()
    loss_func.cuda()

# Hyper Parameters
EPOCH = 2000
LR = 0.0003
BATCH_SIZE = 128
noiselevel = 3

# Set optimizer
optimizer = torch.optim.Adam(CAE.parameters(), lr=LR, weight_decay=1e-5)

# Additional information about trained instance
trained_data = model_params['data_setting']
print("Step 1: Model Setup Done")

# Load data in the same setting apart from using harder EMG samples
# Call data into numpy array format. Check soure code for additional input specifications
trained_data.default_filepath()
trained_data.set_ecg_filepath()
trained_data.set_emg_filepath(filepath = 'emgdata_final')
trained_data.set_acc_filepath(filepath = 'accdata_final')

# Call data into numpy array format. Check soure code for additional input specifications
clean_ecg = trained_data.pull_all_ecg(tf = 240000) # Total of 14 recordings
emg_noise = trained_data.pull_all_emg(tf = 10000) # 10,000 data points * 3 motions * 2 trials * 4 subjects
acc_dat = trained_data.pull_all_acc(tf = 10000) # equiv to emg

# Remove mean, normalize to range (-1,1), adjust for noiselevel setting.
clean_ecg[0,:] -= np.mean(clean_ecg[0,:])
clean_ecg[0,:] = clean_ecg[0,:]/max(abs(clean_ecg[0,:]))

emg_noise[0,:] -= np.mean(emg_noise[0,:])
emg_noise[0,:] = (emg_noise[0,:]/max(abs(emg_noise[0,:])))*noiselevel

for i in range(0,3):
    acc_dat[i,:] -= np.mean(acc_dat[i,:])
    acc_dat[i,:] = (acc_dat[i,:]/max(abs(acc_dat[i,:])))*float(noiselevel**(0.5))
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

# Reformat to shape that can be imported to neural net
input_dat = trained_data.reformat(input_dat, feature_len = trained_data.feature_len, data_form = trained_data.format)
label_dat = trained_data.reformat(label_dat, feature_len = trained_data.feature_len, data_form = trained_data.format)

train_set, val_set = trained_data.data_splitter(input_dat, label_dat, shuffle = True, ratio = 4)

# Generate tensors for training / validation
train_set = torch.from_numpy(train_set).float()
val_set = torch.from_numpy(val_set).float()
print("Step 2: Data Import Done")

# Save model function
def save_model(save_name, optim, loss_f, lr, epoch = EPOCH):
    dir = '{}/Trained_Params/{}/{}_{}'.format(trained_data.filepath, trained_data.model, save_name, epoch)
    if not os.path.exists(dir):
        os.makedirs(dir)
    CAE.cpu()
    torch.save({'data_setting': trained_data,
                'state_dict': CAE.state_dict(),
                'epoch': epoch,
                'optimizer': optim,
                'loss_function': loss_f,
                'learning_rate': lr
                },
               dir + '/model.pth')
    np.save(dir + '/trainloss.npy',train_loss)
    np.save(dir + '/valloss.npy',val_loss)
    print("Step 4: Model Saved")


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

            de = CAE(b_x)
            loss = loss_func(de, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluates current model state on val data set
        pred = CAE(v_x)
        loss_val_set = loss_func(pred, v_y)
        print('Epoch: {} | train loss: {:.4f} | val loss: {:.4f}'.format(epoch+1, loss.data[0], loss_val_set.data[0]))
        train_loss.append(loss.data[0])
        val_loss.append(loss_val_set.data[0])

    print("Step 2: Model Training Finished")

    # Save trained Parameters
except KeyboardInterrupt:

    if str(input("Save Parameters?(y/n): ")) == 'y':
        save_name = str(input("Save parameters as?: ")) + '_Interrupted'
        save_model(save_name, 'Adam', 'MSELoss', LR)
    else:
        print("Session Terminated. Parameters not saved")

else:
    print("entering else statement")
    save_model('newdata1_ft_nl3', 'Adam', 'L1Loss', LR)
    print(os.listdir(os.getcwd()))
    print(os.getcwd())
