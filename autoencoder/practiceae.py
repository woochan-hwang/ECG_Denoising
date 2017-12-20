# Data import sheet.
# Noise added, Accelerometer data

import wfdb
import numpy as np
import matplotlib.pyplot as plt

t0 = 0
tf = int(30 * 360 * 60)

# Import ECG signal and annotations, 30 min long
# Load Training Set
ecg1_signal = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/101", sampfrom = t0, sampto = tf, channels = [0])
ecg1_ann = wfdb.rdann("/Users/WoochanH/python/ecgproject/sampledata/101", "atr", sampfrom = t0, sampto = tf)
ecg2_signal = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/102", sampfrom = t0, sampto = tf, channels = [0])
ecg2_ann = wfdb.rdann("/Users/WoochanH/python/ecgproject/sampledata/102", "atr", sampfrom = t0, sampto = tf)
print(np.shape(ecg1_signal.p_signals))

# Load Test Set
test_ecg_signal = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/103", sampfrom = t0, sampto = tf, channels = [0])
test_ecg_ann = wfdb.rdann("/Users/WoochanH/python/ecgproject/sampledata/103", "atr", sampfrom = t0, sampto = tf)
print(np.shape(test_ecg_signal.p_signals))

# Import EMG signal and annotations , 2 min long
emg_signal = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/emg_healthy", sampfrom = t0, sampto = 2*60*360, channels = [0])
print(np.shape(emg_signal.p_signals))

# Reshape analouge data = cut into length N by reshaping into ndarray of (total/N, N)
def Reshape(signal_input, n):
    signal = signal_input.p_signals
    if len(signal) % n == 0:
        outarray = np.reshape(signal, (len(signal)/n,n))
    else:
        print("Not valid N number")
    return outarray

ecg1 = Reshape(ecg1_signal, 1);
ecg2 = Reshape(ecg2_signal, 1);
ecg_test = Reshape(test_ecg_signal, 1);
emg = Reshape(emg_signal, 1);

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

# EMG Noise
def EMG_noise(base_signal, emg_signal, t1, t2, amp = 1):
    movement = []
    for i in range(len(base_signal)):
        if i < t1:
            movement.append(0)
        elif i >= t1 and i < t2:
            movement.append(emg_signal[i-t1])
        else:
            movement.append(0)
    noise = np.multiply(amp,movement)
    return noise

# ACC noise
def ACC_noise(signal, t1, t2, amp = 5):
    movement = []
    noise = np.random.normal(0, 0.01, len(signal))
    for i in range(len(signal)):
        if i < t1:
            movement.append(noise[i])
        elif i >= t1 and i < t2:
            movement.append(np.multiply((1 + amp),noise[i]))
        else:
            movement.append(noise[i])
    noise = np.multiply(amp,movement)
    return noise


# Noise Ensamble with Accelerometer data
def Combine_noise(ecg_input, emg_input, emg_t1, emg_t2, emg_amp = 1):
    Output = np.reshape(ecg_input, (648000,)) + Gaussian_noise(ecg_input) + Baseline_wander(ecg_input) + EMG_noise(ecg_input, emg_input, emg_t1, emg_t2, emg_amp)
    return Output

print(np.shape(ecg2))
print(np.shape(np.concatenate((ecg1 + ecg2), axis = 0)))

Mydata_dirty = Combine_noise(ecg1, emg, 50000, 93200, 3);
Mydata_clean = Combine_noise(ecg1, emg, 50000, 93200, 0);    # Noisy but without EMG
Test_dirty = Combine_noise(ecg_test, emg, 50000, 93200, 5);
#Just Gaussian_noise
Pretrain1 = np.reshape(ecg1, (648000, )) + Gaussian_noise(ecg1, amp = 3)
test_gaussian = np.reshape(ecg2, (648000, )) + Gaussian_noise(ecg2, amp = 2)
#Gaussian + Baseline
Pretrain2 = Combine_noise(ecg1, emg, 50000, 93200, 0);



Data_Input = np.reshape(Mydata_dirty, (1000, 648))
Data_Label = np.reshape(ecg1, (1000, 648))
Test_data = np.reshape(Test_dirty, (1000,648))

Data_Pretrain1 = np.reshape(Pretrain1, (1000, 648))
Label_Pretrain1 = np.reshape(ecg1, (1000, 648))
Test_Pretrain1 = np.reshape(test_gaussian, (1000, 648))
Test_Pretrain1_label = np.reshape(ecg2, (1000,648))



'''
plt.figure(figsize=(10,4))
plt.axes([0.1,0.1,0.8,0.8])
plt.title("ECG + EMG noise + Gaussian + Baseline Wander")
plt.plot(Data_Input[100],color='red', linewidth=0.2, linestyle='-')
plt.show()
'''

'''
def Combine_noise2(ecg_input, emg_input, emg_ts, emg_tf, emg_amp):
    Noiseadded = ecg_input + Gaussian_noise(ecg_input) + Baseline_wander(ecg_input) + EMG_noise(emg_input, emg_ts, emg_tf, emg_amp)
    Base_acc = ACC_noise(ecg_input, emg_ts, emg_tf, amp = 10*emg_amp)
    Output = np.stack((Noiseadded, Base_acc), axis = -1)
    return Output

Mydata = Combine_noise(ecg_raw, emg_raw, 500, 1800, 1);
Cleandata = Combine_noise(ecg_raw, emg_raw, 500, 1800, 0)
print(np.shape(Mydata))
'''
