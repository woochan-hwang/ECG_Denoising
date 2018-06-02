# Written by Woochan H.
# Exports preprocessed data in npy file. 

import numpy as np
from chicken_selects import *

#noiselevel = int(input("EMG noise level?: "))
noiselevel = float(input("Noise level?: "))
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
print(np.shape(noisy_ecg))

dir = '{}/Ze/NL_{}'.format(data.filepath, data.noiselevel)
if not os.path.exists(dir):
    os.makedirs(dir)

np.save(dir + '/noisyecg',noisy_ecg)
np.save(dir + '/cleanecg',clean_ecg)

print("Saved Noisy ECG of NL {}".format(noiselevel))
