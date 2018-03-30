# Evaluating Sample Stacked denoising autoencoder (2 layers)
# Written by Woochan H.

import os
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


# Checks file for appropirate model and prints availabe trained parameters
if str(input("Load Stacked Autoencoder model?(y/n): ")) == 'y':
    dir = "{}/Trained_Params/{}".format(os.getcwd(), 'Stacked Autoencoder')
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

# Define Model Structure. Should be same as one used for Training
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

# Loads model and model parameters
model_params = torch.load(params_dir)
train_loss = np.load('{}/{}/trainloss.npy'.format(dir,save_name))
test_loss = np.load('{}/{}/testloss.npy'.format(dir,save_name))
mymodel = StackedAutoEncoder()
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
if str(input("Was it trained on this machine(y/n)?: ")) != 'y':
    trained_data.default_filepath()
    trained_data.set_ecg_filepath()
    trained_data.set_emg_filepath()
    trained_data.set_acc_filepath()

clean_ecg = trained_data.pull_all_ecg()
noisy_ecg = trained_data.pull_all_emg()
acc_dat = trained_data.pull_all_acc()

# Acc data modified to fit that of noisy emg
acc_dat = np.array(list(acc_dat[:, 0:6000].transpose())*int(108*trained_data.opened_emg/trained_data.opened_acc)).transpose()

# Clean generated from gaussian dist, N(0, 0.05)
clean_acc = np.random.randn(np.shape(acc_dat)[0], np.shape(acc_dat)[1]) *0.05

# Add ACC data onto clean/noisy ecg data
input_dat = np.vstack((noisy_ecg, acc_dat))
label_dat = np.vstack((clean_ecg, clean_acc))

# Reformat to shape that can be imported to neural net
input_dat = trained_data.reformat(input_dat, feature_len = trained_data.feature_len, data_form = trained_data.format)
label_dat = trained_data.reformat(label_dat, feature_len = trained_data.feature_len, data_form = trained_data.format)

train_set, test_set = trained_data.data_splitter(input_dat, label_dat, shuffle = True, ratio = 4)
print("Step 1: Data Import Done")

if str(input("Continue(y/n)?: ")) == 'n':
    quit()


# Generate tensors for training / testing
i_x, i_y = Variable(torch.from_numpy(train_set[:,0,:]).float()), train_set[:,1,:]
t_x, t_y = Variable(torch.from_numpy(test_set[:,0,:]).float()), test_set[:,1,:]

# Evaluate model on train data
_, _, _, pred_i_y = mymodel(i_x)
# Evaluate model on test data
_, _, _, pred_t_y = mymodel(t_x)
print("Step 2: Model Evaluation Finished")


print("Step 3: Plotting Results")
# Change tensor output to numpy and undo reformatting
pred_i_y = trained_data.undo_reformat(pred_i_y.data.numpy())
pred_t_y = trained_data.undo_reformat(pred_t_y.data.numpy())
i_x = trained_data.undo_reformat(i_x.data.numpy())
t_x = trained_data.undo_reformat(t_x.data.numpy())
i_y = trained_data.undo_reformat(i_y)
t_y = trained_data.undo_reformat(t_y)

print("Available data lenght: {}".format(np.shape(t_y)))
t0 = int(input("Plotting | Start time?: "))
tf = t0 + int(input("Plotting | Duration?: "))

i_x, i_y, pred_i_y = i_x[0,t0:tf], i_y[0,t0:tf], pred_i_y[0,t0:tf]
t_x, t_y, pred_t_y = t_x[0,t0:tf], t_y[0,t0:tf], pred_t_y[0,t0:tf]

train_loss = np.average(train_loss[-50:])
test_loss = np.average(test_loss[-50:])


# Plot Results
fig, (ax1, ax2) = plt.subplots(2, sharey=True)
ax1.plot(pred_i_y, color='b', linewidth=0.4, linestyle='-', label = 'denoised ecg')
ax1.plot(i_y, color='k', linewidth=0.4, linestyle='-', label = 'clean ecg')
ax1.plot(i_x, color='r', linewidth=0.2, linestyle='-', label = 'noisy ecg')
ax1.set(title='Model Output | after epochs: {} | train_loss: {:.4f}'.format(trained_epochs, train_loss),
        ylabel='train set')
ax1.legend(loc = 2)

ax2.plot(pred_t_y, color='b', linewidth=0.4, linestyle='-', label = 'denoised ecg')
ax2.plot(t_y, color='k', linewidth=0.4, linestyle='-', label = 'clean ecg')
ax2.plot(t_x, color='r', linewidth=0.2, linestyle='-', label = 'noisy ecg')
ax2.set(xlabel ='time(s, {} to {})'.format(t0,tf), ylabel='test set')
ax2.legend(loc = 2)

plt.show()

plt.figure(figsize = (10,4));
plt.plot(train_loss, color='k', linewidth=0.4, linestyle='-', label = 'train_set loss');
plt.plot(test_loss, color='b', linewidth=0.4, linestyle='-', label = 'test_set loss')
plt.legend(loc = 2);
plt.title("Training Loss({} | {} | LR:{})".format(trained_data.model, trained_data.motion, trained_lr));
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()



print("Session Terminated. Parameters not saved")
