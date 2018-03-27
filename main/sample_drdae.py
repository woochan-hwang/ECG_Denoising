# Sample Stacked denoising autoencoder (2 layers)
# Note use of chicken_selects module and how to process data imports
# Written by Woochan H. (Last modified: 26/03/18)

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from chicken_selects import *

# Object Data('model type', 'motion', noiselevel, cuda = False)
data = Data('Recurrent DAE', 'flexion_extension', 1)

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
print("Step 0: Data Import Done")

#input_dat = data.undo_reformat(input_dat)
#label_dat = data.undo_reformat(label_dat)

_ = input("Continue(y/n)?: ")

# Change numpy array to tensor variable
Tensor_Input = Variable(torch.from_numpy(input_dat).float())
Tensor_Label = Variable(torch.from_numpy(label_dat).float())

# Hyper Parameters
EPOCH = int(input("Epochs?: "))
LR = float(input("Learning rate?: "))
BATCH_SIZE = int(input("Batch size?: "))

# Autoencoder Model Structure
class DRDAE(nn.Module):
    def __init__(self):
        super(DRDAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(200*4,200),
            nn.ReLU(),
            nn.Linear(200,70),
        )

        self.i2h = nn.Linear(140,70)
        self.i2o = nn.Linear(140,70)

        self.decoder = nn.Sequential(
            nn.Linear(70,200),
            nn.ReLU(),
            nn.Linear(200,200*4)
        )

    def forward(self, x, hidden):
        encoded = self.encoder(x)
#        print('encoded size:', encoded.size())
#        print('hidden size: ', hidden.size())
        combined = torch.cat((encoded, hidden), 0)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.decoder(output)
#        print('output size: ', output.size())
        return output, hidden

    def initHidden(self):
        h = Variable(torch.from_numpy(np.random.randn(70,)*0.1).float())
        if data.cuda == True:
            h = h.cuda()
        return h

print("Step 1: Model Setup Done")

drdae = DRDAE()
optimizer = torch.optim.Adam(drdae.parameters(), lr=LR)
loss_func = nn.MSELoss()
train_loss = []
hidden = drdae.initHidden()

def save_params(save_name):
    dir = '{}/Trained_Params/{}/{}'.format(data.filepath, data.model, save_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(drdae.encoder, dir + '/encoder_{}.pt'.format(EPOCH))
    torch.save(drdae.decoder, dir + '/decoder_{}.pt'.format(EPOCH))
    torch.save(drdae.i2h, dir + '/i2h_{}.pt'.format(EPOCH))
    torch.save(drdae.i2o, dir + '/i2o_{}.pt'.format(EPOCH))
    np.save(dir + '/trainloss_{}.npy'.format(EPOCH),train_loss)
    print("Step 3: Model Saved")


# Train the model
try:
    print("Step 2: Model Training Start")

    if data.cuda == True:
        drdae = drdae.cuda()
        Tensor_Input = Tensor_Input.cuda()
        Tensor_Label = Tensor_Label.cuda()

    for epoch in range(EPOCH):

        hidden = drdae.initHidden()

        for i in range(1, Tensor_Input.size()[0]):

            output, hidden = drdae.forward(Tensor_Input[i], hidden)

            loss = loss_func(output, Tensor_Label[i])
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()

        print('Epoch: ', epoch + 1, '| train loss: %.6f' % loss.data[0])
        train_loss.append(loss.data[0])

    print("Step 2: Model Training Finished")

    # Plot Loss
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
