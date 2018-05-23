# Convolutional denoising autoencoder (2 layers)
# Written by Woochan H.
'''
This method incorporates the concept of EMD(Empirical Mode Decomposition).
Implementation is based on vanilla EMD with Cauchy Convergence.

Parallel Conv Nets are trained to denoise each IMF component after applying EMD
to the original signal. Then the final result is reconstructed from the clean IMFs.

For the Conv Net, Version 3 is applied here with NL 3 as initial starting.
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.utils.data as loader
from chicken_selects import *
import matplotlib.pyplot as plt
from PyEMD import EMD

print("Torch version: ", torch.__version__)

#noiselevel = int(input("EMG noise level?: "))
noiselevel = 3
# Object Data('model type', 'motion', noiselevel, cuda = False)
data = Data('EMD Convolutional Autoencoder', 'mixed', noiselevel = noiselevel)

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

# Split data into train and validation set
train_set, val_set = data.data_splitter(input_dat, label_dat, shuffle = True, ratio = 4)

# Load IMF outputs of EMD on the trainset
emd_train = np.load(str(data.filepath) + '/EMDs_{}/EMDs_train.npy'.format(noiselevel))
emd_val = np.load(str(data.filepath) + '/EMDs_{}/EMDs_val.npy'.format(noiselevel))
missing_train = np.load(str(data.filepath) + '/EMDs_{}/short_list_train.npy'.format(noiselevel))
missing_val = np.load(str(data.filepath) + '/EMDs_{}/short_list_val.npy'.format(noiselevel))
print("Data load complete: EMD file shape | {}".format(np.shape(emd_train)))
print("Samples with less than 5 EMDs: ", np.shape(missing_train)[0] + np.shape(missing_val)[0])
print("EMD_ECG size match: ", np.shape(train_set)[0] == np.shape(emd_train)[0]
      and np.shape(val_set)[0] == np.shape(emd_val)[0])
print("Step 0: Data Import Done")

EPOCH = 5000
LR = 0.0003
BATCH_SIZE = 128

cuda = True if torch.cuda.is_available() else False
print("CUDA: ", cuda)

# Generate tensors for training / validation
train_set = torch.from_numpy(train_set).float()
val_set = torch.from_numpy(val_set).float()
emd_train = torch.from_numpy(emd_train).float()
emd_val = torch.from_numpy(emd_val).float()

# Autoencoder Model Structure
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Zero padding is almost the same as average padding in this case
        # Input = b, 1, 4, 300
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (4,7), stride=1, padding=(0,3)), # b, 8, 1, 300
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

print("Step 1: Model Setup Done")

# Initialize network
CAE1 = ConvAutoEncoder() # IMF 1
CAE2 = ConvAutoEncoder() # IMF 2
CAE3 = ConvAutoEncoder() # IMF 3
loss_func = nn.L1Loss()
train_loss, val_loss = [], [] # Train loss is loss of IMF_CAEs. Val loss is reconstruction loss

if cuda:
    CAE1.cuda()
    CAE2.cuda()
    CAE3.cuda()
    loss_func.cuda()

# Set optimizer
optimizer1 = torch.optim.Adam(CAE1.parameters(), lr=LR, weight_decay=1e-5)
optimizer2 = torch.optim.Adam(CAE2.parameters(), lr=LR, weight_decay=1e-5)
optimizer3 = torch.optim.Adam(CAE3.parameters(), lr=LR, weight_decay=1e-5)

# Define model save function
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
    np.save(dir + '/valloss.npy',val_loss)
    print("Step 3: Model Saved")

# Train the model
try:
    # Generates mini_batchs for training. Loads data for validation.
    train_loader = loader.DataLoader(dataset = emd_train, batch_size = BATCH_SIZE, shuffle = True)

    # Moves data and model to gpu if available
    if cuda:
        train_set.cuda()
        val_set.cuda()
        emd_train.cuda()
        emd_val.cuda()

    print("Step 2: Model Training Start")

    for epoch in range(EPOCH):
        for step, train_data in enumerate(train_loader):

            # Load IMFs from pre-run EMD
            x_IMFs = train_data[:,0:4,:,:] # Adjust to new EMD
            y_IMFs = train_data[:,4:10,:,:]

            # Construct Tensor Variable for each IMF order
            b_x1 = Variable(x_IMFs[:,0:1,:,:]).cuda() if cuda else Variable(x_IMFs[:,0:1,:,:])
            b_y1 = Variable(y_IMFs[:,0:1,:,:]).cuda() if cuda else Variable(y_IMFs[:,0:1,:,:])

            b_x2 = Variable(x_IMFs[:,1:2,:,:]).cuda() if cuda else Variable(x_IMFs[:,1:2,:,:])
            b_y2 = Variable(y_IMFs[:,1:2,:,:]).cuda() if cuda else Variable(y_IMFs[:,1:2,:,:])

            b_x3 = Variable(x_IMFs[:,2:3,:,:]).cuda() if cuda else Variable(x_IMFs[:,2:3,:,:])
            b_y3 = Variable(y_IMFs[:,2:3,:,:]).cuda() if cuda else Variable(y_IMFs[:,2:3,:,:])

            de1, de2, de3 = CAE1(b_x1), CAE2(b_x2), CAE3(b_x3)

            loss1 = loss_func(de1, b_y1)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = loss_func(de2, b_y2)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            loss3 = loss_func(de3, b_y3)
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

        print('Epoch loss: {} | IMF 1: {:.4f} | IMF 2: {:.4f} | IMF 3: {:.4f}'.format(epoch+1, loss1.data[0], loss2.data[0], loss3.data[0]))
        train_loss.append((loss1.data[0], loss2.data[0], loss3.data[0]))

        # Evaluates current model state by reconstructing IMFs every 10 epochs
        if epoch % 10 == 0:
            train_pred = (CAE1(Variable(emd_train[:,0:1,:,:])).data + CAE2(Variable(emd_train[:,1:2,:,:])).data
                          + CAE3(Variable(emd_train[:,2:3,:,:])).data + emd_train[:,3:4,:,:])
            val_pred = (CAE1(Variable(emd_val[:,0:1,:,:])).data + CAE2(Variable(emd_val[:,1:2,:,:])).data
                        + CAE3(Variable(emd_val[:,2:3,:,:])).data + emd_val[:,3:4,:,:])
            train_recon_loss = torch.mean(torch.sum(torch.abs(torch.add(-train_pred[:,0,0,:], train_set[:,1,0,:])), dim = 1))
            val_recon_loss = torch.mean(torch.sum(torch.abs(torch.add(-val_pred[:,0,0,:], val_set[:,1,0,:])), dim = 1))
            print('Recon loss: {} | Train set: {} | Val set: {}'.format(epoch + 1, train_recon_loss, val_recon_loss))
            val_loss.append((train_recon_loss,val_recon_loss))

    print("Step 2: Model Training Finished")

    # Save trained Parameters
except KeyboardInterrupt:

    if str(input("Save Parameters?(y/n): ")) != 'n':
        save_name = str(input("Save parameters as?: ")) + '_Interrupted'
        save_model(save_name, 'Adam', 'L1Loss', LR)
    else:
        print("Session Terminated. Parameters not saved")

else:
    print("entering else statement")
    save_model('EMD_CDAE_nl{}'.format(noiselevel), 'Adam', 'L1Loss', LR)
    print(os.listdir(os.getcwd()))
    print(os.getcwd())
