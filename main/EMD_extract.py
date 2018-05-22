# Written by Woochan H.
'''
This method incorporates the concept of EMD(Empirical Mode Decomposition).
Implementation is based on vanilla EMD with Cauchy Convergence.

This extracts the EMD and stores it as numpy.
IMF orders higher than 5th will all be summed back to the 5th order, including residue.
This is to prevent having different number of IMFs available for denoising using
neural networks in the next stage.
'''
import numpy as np
from chicken_selects import *
from PyEMD import EMD

#noiselevel = int(input("EMG noise level?: "))
noiselevel = 3
# Object Data('model type', 'motion', noiselevel, cuda = False)
data = Data('Convolutional Autoencoder', 'mixed', noiselevel = noiselevel)

# Object EMD from PyEMD package. Default Cauchy convergence. 
EMD = EMD()

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
print(np.shape(train_set))

# Run EMDs on dataset up to 5th order IMFs.
def EMDsplit(dataset):
    samples = np.shape(dataset)[0]
    newdata = np.zeros((samples,10,4,300))
    short_samples = []
    for i in range(samples):
        x = dataset[i,0,0,:]
        y = dataset[i,1,0,:]
        x_IMFs = EMD(x)
        y_IMFs = EMD(y)
        x_len, y_len = np.shape(x_IMFs)[0], np.shape(y_IMFs)[0]
        # For higher order IMFs and residue, sum it back to IMF5.
        if x_len > 5:
            for imf in range(5, x_len):
                x_IMFs[4,:] += x_IMFs[imf,:]
        if y_len > 5:
            for imf in range(5, y_len):
                y_IMFs[4,:] += y_IMFs[imf,:]
        for j in range(5):
            newdata[i,j,0,:] = x_IMFs[j]
            newdata[i,4+j,0,:] = y_IMFs[j]
        print("Step {}/{} done".format(i,samples))
        if x_len <5 or y_len <5:
            print("This step had a short EMD")
            short_samples.append(i)
    return newdata, short_samples

def save_EMD(EMD, short_list):
    dir = '{}/{}'.format(data.filepath, 'EMDs')
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(dir + '/EMDs.npy',EMD)
    np.save(dir + '/short_list.npy',short_list)
    print("EMD Saved")

train_EMD, short_list = EMDsplit(train_set)
save_EMD(train_EMD, short_list)

print(os.listdir(os.getcwd()))
print(os.getcwd())
