# Data import sheet

import wfdb
import numpy as np
import matplotlib.pyplot as plt

# Import raw ECG data.
t0 = 0
tf = int(30 * 360 * 60)

ecg1_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/101", sampfrom = t0, sampto = tf, channels = [0])
ecg2_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/102", sampfrom = t0, sampto = tf, channels = [0])
ecg3_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/103", sampfrom = t0, sampto = tf, channels = [0])
ecg4_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/104", sampfrom = t0, sampto = tf, channels = [0])
ecg5_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/105", sampfrom = t0, sampto = tf, channels = [0])
ecg6_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/106", sampfrom = t0, sampto = tf, channels = [0])
ecg7_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/107", sampfrom = t0, sampto = tf, channels = [0])
ecg8_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/108", sampfrom = t0, sampto = tf, channels = [0])
ecg9_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/109", sampfrom = t0, sampto = tf, channels = [0])
ecg10_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/111", sampfrom = t0, sampto = tf, channels = [0])
ecg11_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/112", sampfrom = t0, sampto = tf, channels = [0])
ecg12_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/113", sampfrom = t0, sampto = tf, channels = [0])
ecg13_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/114", sampfrom = t0, sampto = tf, channels = [0])
ecg14_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/115", sampfrom = t0, sampto = tf, channels = [0])
ecg15_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/116", sampfrom = t0, sampto = tf, channels = [0])
ecg16_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/117", sampfrom = t0, sampto = tf, channels = [0])
ecg17_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/118", sampfrom = t0, sampto = tf, channels = [0])
ecg18_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/119", sampfrom = t0, sampto = tf, channels = [0])

test_data = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/100", sampfrom = t0, sampto = tf, channels = [0])

# EMG import 2 min long
# emg_signal = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/emg_healthy", sampfrom = t0, sampto = 2*60*360, channels = [0])

# Reshape analouge data = cut into length N by reshaping into ndarray of (total/N, N)
def Reshape(signal_input, n):
    signal = signal_input.p_signals
    if len(signal) % n == 0:
        outarray = np.reshape(signal, (len(signal)/n,n))
    else:
        print("Not valid N number")
    return outarray

ecg1 = Reshape(ecg1_data, 1);
ecg2 = Reshape(ecg2_data, 1);
ecg3 = Reshape(ecg3_data, 1);
ecg4 = Reshape(ecg4_data, 1);
ecg5 = Reshape(ecg5_data, 1);
ecg6 = Reshape(ecg6_data, 1);
ecg7 = Reshape(ecg7_data, 1);
ecg8 = Reshape(ecg8_data, 1);
ecg9 = Reshape(ecg9_data, 1);
ecg10 = Reshape(ecg10_data, 1);
ecg11 = Reshape(ecg11_data, 1);
ecg12 = Reshape(ecg12_data, 1);
ecg13 = Reshape(ecg13_data, 1);
ecg14 = Reshape(ecg14_data, 1);
ecg15 = Reshape(ecg15_data, 1);
ecg16 = Reshape(ecg16_data, 1);
ecg17 = Reshape(ecg17_data, 1);
ecg18 = Reshape(ecg18_data, 1);


train_data_clean = np.concatenate((ecg1, ecg2, ecg3, ecg4, ecg5, ecg6, ecg7, ecg8, ecg9, ecg10, ecg11, ecg12, ecg13, ecg14, ecg15, ecg16, ecg17, ecg18), axis = 0);
test_data_clean = Reshape(test_data, 1);

print('Train data raw length: ', np.shape(train_data_clean))
print('Test data raw length', np.shape(test_data_clean))

# Noise Generation
# Random guassian noise, default mean = 0, std = 0.01
def Gaussian_noise(signal, u = 0, std = 0.01, amp = 1):
    noise = np.multiply(amp, np.random.normal(u, std, len(signal)))
    return noise

# Baseline Wander, default resp rate = 12/min, amp = 0.05 (standard is 15% of peak to peak of ECG)
def Baseline_wander(signal, respf = 12, amp = 0.05):
    N = float(len(signal))
    resppm = (float(respf)/60)
    ix = np.arange(N)
    noise  = amp*np.sin(2*np.pi/(resppm*(N/2))*ix)
    return noise


# Adding Gaussian noise
train_data_Gaussian = np.reshape(train_data_clean, (11664000, )) + Gaussian_noise(train_data_clean, amp = 5);
test_data_Gaussian = np.reshape(test_data_clean, (648000, )) + Gaussian_noise(test_data_clean, amp = 3);

print(np.shape(train_data_Gaussian))

# Adding Gaussian noise + Baseline wander
train_data_GnB = train_data_Gaussian + Baseline_wander(train_data_clean);
test_data_GnB = test_data_Gaussian + Baseline_wander(test_data_clean);

# Final form for pretraining on just Gaussian noise
AE_Input_train = np.reshape(train_data_GnB,(43200, 270))
AE_Input_test = np.reshape(test_data_GnB, (2400,270))
AE_Label = np.reshape(train_data_clean, (43200, 270))

print(np.shape(AE_Input_train), type(AE_Input_train))
