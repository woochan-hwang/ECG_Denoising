# Convolutional denoising autoencoder for ECGs
# Written by Woochan H.

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.utils.data as loader
from chicken_selects import *

import matplotlib
if str(input("x11-backend(y/n)?: ")) == 'y':
    matplotlib.use('GTKAgg')
    print("GTKAgg backend in use")
import matplotlib.pyplot as plt

if str(input("Continue Training from pretrained model(y/n)?: ")) == 'y':

    # Checks file for appropirate model and prints availabe trained parameters
    if str(input("Load Conv Autoencoder model(y/n)?: ")) == 'y':
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

if str(input("Continue(y/n)?: ")) == 'n':
    quit()


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

# Loads model and model parameters
model_params = torch.load(params_dir)
CAE = ConvAutoEncoder()
CAE.load_state_dict(model_params['state_dict'])
train_loss, val_loss = [], []

# Additional information about trained instance
trained_data = model_params['data_setting']
print("Step 1: Model Setup Done")

# Load data in the same setting apart from using harder EMG samples
# Call data into numpy array format. Check soure code for additional input specifications
print("Following folders available:{}".format(os.listdir(os.getcwd())))
emgfile = str(input("Which EMG data will you use?: "))
trained_data.default_filepath()
trained_data.set_ecg_filepath()
trained_data.set_emg_filepath(filepath = emgfile)
trained_data.set_acc_filepath()

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
noisy_ecg = trained_data.pull_all_emg()

# Acc data modified to fit that of noisy emg. Adjust for increasing EMG ratio
acc = np.array(list(acc_dat[:, 0:24000].transpose()))
acc_dat = np.array((list(acc) + list(acc*1.4) +list(acc*1.7)+ list(acc*2))*int(k*trained_data.opened_emg/trained_data.opened_acc)).transpose()

# Clean generated from gaussian dist, N(0, 0.05)
clean_acc = np.random.randn(np.shape(acc_dat)[0], np.shape(acc_dat)[1]) *0.05

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

if str(input("Continue(y/n)?: ")) == 'n':
    quit()


# Hyper Parameters
EPOCH = int(input("Epochs?: "))
LR = float(input("Learning rate?: "))
BATCH_SIZE = int(input("Batch size?: "))

if torch.cuda.is_available() == True:
    CAE = CAE.cuda()
optimizer = torch.optim.Adam(CAE.parameters(), lr=LR, weight_decay=1e-5)
loss_func = nn.MSELoss()


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
    t_x, t_y = Variable(val_set[:,0:1,:,:]), Variable(val_set[:,1:2,:,:])
    print("t_x", t_x.size())

    # Moves data and model to gpu if available
    if torch.cuda.is_available() == True:
        t_x = t_x.cuda()
        t_y = t_y.cuda()

    print("Step 2: Model Training Start")

    for epoch in range(EPOCH):
        for step, train_data in enumerate(train_loader):

            if torch.cuda.is_available() == True:
                b_x = Variable(train_data[:,0:1,:,:]).cuda()
                b_y = Variable(train_data[:,1:2,:,:]).cuda()
            else:
                b_x = Variable(train_data[:,0:1,:,:])
                b_y = Variable(train_data[:,1:2,:,:])

            de = CAE(b_x)
            loss = loss_func(de, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluates current model state on val data set
        pred_y = CAE(t_x)
        loss_val_set = loss_func(pred_y, t_y)
        print('Epoch: {} | train loss: {:.6f} | val loss: {:.6f}'.format(epoch+1, loss.data[0], loss_val_set.data[0]))
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
    if str(input("Save Parameters?(y/n): ")) == 'y':
        save_name = str(input("Save parameters as?: "))
        save_model(save_name, 'Adam', 'MSELoss', LR)
    else:
        print("Parameters not saved")

    # Plot Loss
    threshold_train = 500*min(train_loss)
    threshold_val = 500*min(val_loss)
    for i in range(len(train_loss)):
        if train_loss[i] > threshold_train:
            train_loss[i] = threshold_train
        if val_loss[i] > threshold_val:
            val_loss[i] = threshold_val
    plt.figure(figsize = (10,4));
    plt.plot(train_loss, color='k', linewidth=0.4, linestyle='-', label = 'train_set loss');
    plt.plot(val_loss, color='b', linewidth=0.4, linestyle='-', label = 'val_set loss')
    plt.legend(loc = 2);
    plt.title("FineTune: Training Loss({} | {} | LR:{})".format(trained_data.model, trained_data.motion, LR));
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.show()
