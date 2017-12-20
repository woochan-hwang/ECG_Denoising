# Representative Noise Generation for ECG
# Woochan Hwang _ iBSc Group Project

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
import scipy


# Input time settings
#t0 = int(input("Start time?(min): ") * 360 * 60)
#tf = int(input("End time?(min): ") * 360 * 60)
t0 = 0
tf = int(0.1 * 360 * 60)

# Import ECG signal and annotations
ecg_signal = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/101", sampfrom = t0, sampto = tf, channels = [0])
ecg_ann = wfdb.rdann("/Users/WoochanH/python/ecgproject/sampledata/101", "atr", sampfrom = t0, sampto = tf)
print(dir(ecg_signal));
print(np.shape(ecg_signal.p_signals))

# Import EMG signal and annotations
emg_signal = wfdb.rdsamp("/Users/WoochanH/python/ecgproject/sampledata/emg_healthy", sampfrom = t0, sampto = tf, channels = [0])
print(dir(emg_signal));
print(np.shape(emg_signal.p_signals))

# Reshape analouge data = newraw
def Reshape(signal_input, outfunction):
    signal = signal_input.p_signals
    for i in range(0,len(signal)):
        outfunction.append(float(signal[i][0]))
    return outfunction

ecg_raw = []
emg_raw = []
Reshape(ecg_signal, ecg_raw);
Reshape(emg_signal, emg_raw);
print(np.shape(ecg_raw))
print(np.shape(emg_raw))


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
def EMG_noise(signal, t1, t2, amp = 1):
    movement = []
    for i in range(len(signal)):
        if i < t1:
            movement.append(0)
        elif i >= t1 and i < t2:
            movement.append(signal[i])
        else:
            movement.append(0)
    noise = np.multiply(amp,movement)
    return noise


# Noise Ensamble
def Combine_noise(ecg_input, emg_input, emg_ts, emg_tf, emg_amp):
    Output = ecg_input + Gaussian_noise(ecg_input) + Baseline_wander(ecg_input) + EMG_noise(emg_input, emg_ts, emg_tf, emg_amp)
    return Output

Mydata = Combine_noise(ecg_raw, emg_raw, 500, 2000, 5);
Mydata2 = Combine_noise(ecg_raw, emg_raw, 500, 2000, 0);

# Figure Plotting
plt.figure(figsize=(10,4))
plt.axes([0.1,0.1,0.8,0.8])
plt.title("ECG + EMG noise + Gaussian + Baseline Wander")
plt.plot(Mydata,color='red', linewidth=0.2, linestyle='-')
plt.show()

plt.figure(figsize=(10,4))
plt.axes([0.1,0.1,0.8,0.8])
plt.title("ECG + Gaussian + Baseline Wander")
plt.plot(Mydata2,color='red', linewidth=0.2, linestyle='-')
plt.show()


# FFT on Mydata
FFT_with_EMG = np.fft.fft(Mydata)
FFT_without_EMG = np.fft.fft(Mydata2)
FFT_just_EMG = np.fft.fft(emg_raw)

# Plotting FFT form
plt.figure(figsize=(10,4))
plt.subplot(1, 1, 1)
plt.axes([0.1,0.1,0.8,0.8])
plt.plot(FFT_with_EMG, color='red', linewidth=0.2, linestyle='-')
plt.title("FFT on data with EMG noise")

plt.figure(figsize=(10,4))
plt.subplot(2, 1, 2)
plt.axes([0.1,0.1,0.8,0.8])
plt.plot(FFT_just_EMG, color='red', linewidth=0.2, linestyle='-')
plt.title("FFT on just EMG")

plt.show()



# Filtering
# Scipy Butterworth Low Pass filters
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order = 6
fs = 110.0       # sample rate, Hz
cutoff = 20.0  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(Mydata, cutoff, fs, order)

plt.subplot(2, 1, 2)
plt.plot(Mydata2, 'b-', linewidth =0.1, label='clean data')
plt.plot(y, 'r-', linewidth=0.1, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()
