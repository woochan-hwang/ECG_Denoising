# Evaluating Convolutional denoising autoencoder (2 layers)
# Written by Woochan H.

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from chicken_selects import *
import matplotlib.pyplot as plt


# Checks file for appropirate model and prints availabe trained parameters
if str(input("Load Conv Autoencoder model?(y/n): ")) == 'y':
    dir = "{}/Trained_Params/{}".format(os.getcwd(), 'Convolutional Autoencoder')
    if not os.path.exists(dir):
        print("Model does NOT exist. Please check")
        quit()
items = os.listdir(dir)
print("Following Trained parameters available:{}".format(items))

# Sets parameter directory
save_name = str(input("Which would you like to load?: "))
params_dir = '{}/{}/model.pth'.format(dir, save_name)

# Enforce correct directory selection
while os.path.exists(params_dir) == False:
    print("Following Trained parameters available:{}".format(items))
    save_name = str(input("[Try Again] Which would you like to load?: "))
    params_dir = '{}/{}/model.pth'.format(dir, save_name)

cdaever = str(input("CDAE version?(1 or 2): "))
# Define Model Structure. Should be same as one used for Training
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Input = b, 1, 4, 300
        if  cdaever == '1':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, (4,3), stride=1, padding=(0,1)), # b, 8, 1, 300
                nn.Tanh(),
                nn.MaxPool2d((1,2), stride=2), # b, 8, 1, 150
                nn.Conv2d(8, 4, 3, stride=1, padding=1), # b, 8, 1, 150
                nn.Tanh(),
                nn.MaxPool2d((1,2), stride=2) # b, 4, 1, 75
            )
        elif cdaever == '2':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, (4,3), stride=1, padding=(0,1)), # b, 8, 1, 300
                nn.Tanh(),
                nn.MaxPool2d((1,2), stride=2), # b, 8, 1, 150
                nn.Conv2d(8, 8, 3, stride=1, padding=1), # b, 8, 1, 150
                nn.Tanh(),
                nn.MaxPool2d((1,2), stride=2), # b, 8, 1, 75
                nn.Conv2d(8, 4, 3, stride=1, padding=1), # b, 4, 1, 150
                nn.Tanh(),
            )
        elif cdaever == '3':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, (4,7), stride=1, padding=(0,3)), # b, 8, 1, 300
                nn.Tanh(),
                nn.MaxPool2d((1,2), stride=2), # b, 8, 1, 150
                nn.Conv2d(8, 4, 3, stride=1, padding=1), # b, 8, 1, 150
                nn.Tanh(),
                nn.MaxPool2d((1,2), stride=2) # b, 4, 1, 75
            )
        elif cdaever == '4':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, (4,7), stride=1, padding=(0,3)), # b, 8, 1, 300
                nn.Tanh(),
                nn.MaxPool2d((1,2), stride=2), # b, 8, 1, 150
                nn.Conv2d(8, 4, 3, stride=1, padding=1), # b, 8, 1, 150
                nn.Tanh(),
                nn.MaxPool2d((1,2), stride=2) # b, 4, 1, 75
            )
        else:
            print("No such version exists")

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
model_params = torch.load(params_dir)
train_loss = np.load('{}/{}/trainloss.npy'.format(dir,save_name))
val_loss = np.load('{}/{}/valloss.npy'.format(dir,save_name))
val_loss = np.reshape(val_loss, (np.shape(val_loss)[0], 1))

mymodel = ConvAutoEncoder()
mymodel.load_state_dict(model_params['state_dict'])
print("Step 0: Model Structure and Parameter Loaded")

# Additional information about trained instance
trained_data = model_params['data_setting']
trained_epochs = model_params['epoch']
trained_optim = model_params['optimizer']
trained_lossf = model_params['loss_function']
trained_lr = model_params['learning_rate']

# Load data in the same setting used for training
# Call data into numpy array format. Check soure code for additional input specifications
trained_data.default_filepath()
trained_data.set_ecg_filepath()
trained_data.set_emg_filepath(filepath = 'emgdata_final')
trained_data.set_acc_filepath(filepath = 'accdata_final')

# Call data into numpy array format. Check soure code for additional input specifications
clean_ecg = trained_data.pull_all_ecg(tf = 240000) # Total of 14 recordings
emg_noise = trained_data.pull_all_emg(tf = 10000) # 10,000 data points * 3 motions * 2 trials * 4 subjects
acc_dat = trained_data.pull_all_acc(tf = 10000) # equiv to emg

noiselevel = int(input("EMG noise level?: "))

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


'''
if emgfile == 'H2_emgdata':
    clean_ecg = trained_data.pull_all_ecg(tf = 576000)
    acc_dat = trained_data.pull_acc("overall_acc")
    k = 6
    acc = np.array(list(acc_dat[:, 0:24000].transpose()))
else:
    clean_ecg = trained_data.pull_all_ecg()
    acc_dat = trained_data.pull_acc("upsampled_acc")
    k = 27
    acc = np.array(list(acc_dat[:, 0:6000].transpose()))
noisy_ecg = trained_data.pull_all_emg(tf = 576000)
print("noisy_ecg shape: {}".format(np.shape(noisy_ecg)))
# Acc data modified to fit that of noisy emg. Adjust for increasing EMG ratio
acc_dat = np.array((list(acc) + list(acc*1.4) +list(acc*1.7)+ list(acc*2))*int(k*trained_data.opened_emg/trained_data.opened_acc)).transpose()
print("acc shape: {}".format(np.shape(acc_dat)))

# Clean generated from gaussian dist, N(0, 0.05)
clean_acc = np.random.randn(np.shape(acc_dat)[0], np.shape(acc_dat)[1]) *0.05
'''
# Add ACC data onto clean/noisy ecg data
input_dat = np.vstack((noisy_ecg, acc_dat))
label_dat = np.vstack((clean_ecg, clean_acc))

# Reformat to shape that can be imported to neural net
input_dat = trained_data.reformat(input_dat, feature_len = trained_data.feature_len, data_form = trained_data.format)
label_dat = trained_data.reformat(label_dat, feature_len = trained_data.feature_len, data_form = trained_data.format)

train_set, val_set = trained_data.data_splitter(input_dat, label_dat, shuffle = True, ratio = 4)
print("Step 1: Data Import Done")

if str(input("Continue(y/n)?: ")) == 'n':
    quit()

# Generate tensors for training / validation
i_x, i_y = Variable(torch.from_numpy(train_set[:,0:1,:,:]).float()), train_set[:,1:2,:,:]
t_x, t_y = Variable(torch.from_numpy(val_set[:,0:1,:,:]).float()), val_set[:,1:2,:,:]

# Evaluate model on train data
pred_i_y = mymodel(i_x)
# Evaluate model on val data
pred_t_y = mymodel(t_x)
print("Step 2: Model Evaluation Finished")


print("Step 3: Plotting Results")
# Change tensor output to numpy and undo reformatting
pred_i_y = trained_data.undo_reformat(pred_i_y.data.numpy())
pred_t_y = trained_data.undo_reformat(pred_t_y.data.numpy())
i_x = trained_data.undo_reformat(i_x.data.numpy())
t_x = trained_data.undo_reformat(t_x.data.numpy())
i_y = trained_data.undo_reformat(i_y)
t_y = trained_data.undo_reformat(t_y)

avg_train_loss = np.average(train_loss[-50:])
avg_val_loss = np.average(val_loss[-50:])

# Plot Results
plot = str(input("Plot results(y/n)?: "))
print("Available data lenght: {}".format(np.shape(t_y)))

while plot == 'y':
    t0 = int(input("Plotting | Start time?: "))
    tf = t0 + int(input("Plotting | Duration?: "))

    i_x_1, i_y_1, pred_i_y_1 = i_x[0,t0:tf], i_y[0,t0:tf], pred_i_y[0,t0:tf]
    t_x_1, t_y_1, pred_t_y_1 = t_x[0,t0:tf], t_y[0,t0:tf], pred_t_y[0,t0:tf]

    fig, (ax1, ax2) = plt.subplots(2, sharey=True)
    ax1.plot(pred_i_y_1, color='b', linewidth=0.4, linestyle='-', label = 'denoised ecg')
    ax1.plot(i_y_1, color='k', linewidth=0.4, linestyle='-', label = 'clean ecg')
    ax1.plot(i_x_1, color='r', linewidth=0.2, linestyle='-', label = 'noisy ecg')
    ax1.set(title='Model Output | after epochs: {} | train_loss: {:.4f}'.format(trained_epochs, avg_val_loss),
            ylabel='train set')
    ax1.legend(loc = 2)

    ax2.plot(pred_t_y_1, color='b', linewidth=0.4, linestyle='-', label = 'denoised ecg')
    ax2.plot(t_y_1, color='k', linewidth=0.4, linestyle='-', label = 'clean ecg')
    ax2.plot(t_x_1, color='r', linewidth=0.2, linestyle='-', label = 'noisy ecg')
    ax2.set(xlabel ='time(s, {} to {})'.format(t0,tf), ylabel='val set')
    ax2.legend(loc = 2)

    plt.show()

    plot = str(input("Plot again(y/n)?: "))

print("Plotting Training Loss")
fig, (ax1, ax2) = plt.subplots(2, sharey=True)
ax1.plot(train_loss, color='k', linewidth=0.4, linestyle='-', label = 'train_set loss');
ax1.legend(loc = 2);
ax1.set(title = "({} | {} | LR:{})".format(trained_data.model, trained_data.motion, trained_lr),
       ylabel = 'Train Loss');

ax2.plot(val_loss, color='b', linewidth=0.4, linestyle='-', label = 'val_set loss')
ax2.legend(loc = 2);
ax2.set(xlabel = "Epochs", ylabel = "Val Loss")

plt.show()


print("Session Terminated")
