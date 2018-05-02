# Evaluating Sample Stacked denoising autoencoder (2 layers)
# Written by Woochan H.

import os
import numpy as np
from chicken_selects import *
import scipy.fftpack

import matplotlib
if str(input("x11-backend?(y/n): ")) == 'y':
    matplotlib.use('GTKAgg')
    print("GTKAgg backend in use")
import matplotlib.pyplot as plt

# Object Data('model type', 'motion', noiselevel, cuda = False)
data = Data('Stacked Autoencoder', 'flexion_extension', 1)

# Specify directory if you have changed folder name / dir
data.set_ecg_filepath()
data.set_emg_filepath(filepath = 'H2_emgdata')
data.set_acc_filepath()

print("Import data")
# Call data into numpy array format. Check soure code for additional input specifications
clean_ecg = data.pull_ecg('101')
noisy_ecg = data.pull_emg('Noisy101')
acc_dat = data.pull_acc('overall_acc')
ambulatory = data.pull_ecg('118e24')

clean_ecg = np.reshape(clean_ecg, (432000,))
ambulatory = np.reshape(ambulatory, (432000,))
# Match size
ambulatory = ambulatory[0:6000]
acc_dat = acc_dat[0,0:6000]
noisy_ecg = noisy_ecg[0:6000]
clean_ecg = clean_ecg[0:6000]
print(np.shape(clean_ecg))
# Clean generated from gaussian dist, N(0, 0.05)
clean_acc = np.random.randn(np.shape(acc_dat)[0]) *0.05

print("Data import done")

# FFT on data
ecg_fft = scipy.fftpack.rfft(clean_ecg)
emg_fft = scipy.fftpack.rfft(noisy_ecg)
acc_fft = scipy.fftpack.rfft(acc_dat)
cacc_fft = scipy.fftpack.rfft(clean_acc)
amb_fft = scipy.fftpack.rfft(ambulatory)

print("FFT done")
print(np.shape(ecg_fft))
print(np.shape(emg_fft))
print(np.shape(acc_fft))
print(ecg_fft)

# Autocorrelation functions




plt.figure(figsize = (10,4));
plt.plot(ecg_fft[5:1000], color='k', linewidth=1.0, linestyle='-', label = 'clean ecg')
plt.plot(emg_fft[5:1000], color='r', linewidth=0.4, linestyle='-', label = 'noisy ecg')
plt.plot(amb_fft[5:1000], color='b', linewidth=0.4, linestyle='-', label = 'amb ecg')
plt.title('FFT output')
plt.legend(loc = 4)

plt.show()

'''
# Plot Results
fig, (ax1, ax2) = plt.subplots(2, sharey=False)
ax1.plot(ecg_fft, color='k', linewidth=0.4, linestyle='-', label = 'clean ecg')
ax1.plot(emg_fft, color='r', linewidth=0.2, linestyle='-', label = 'noisy ecg')
ax1.plot(amb_fft, color='b', linewidth=0.4, linestyle='-', label = 'amb ecg')
ax1.set(title='FFT Output', ylabel='ECG')
ax1.legend(loc = 4)

ax2.plot(cacc_fft, color='k', linewidth=0.4, linestyle='-', label = 'clean acc')
ax2.plot(acc_fft, color='r', linewidth=0.2, linestyle='-', label = 'noisy acc')
ax2.set(xlabel ='frequency (real fft)', ylabel='ACC')
ax2.legend(loc = 4)
print("Plot setup")

plt.show()
'''
