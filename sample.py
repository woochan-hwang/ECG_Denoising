import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

acc = pd.read_csv('EMG accelerometer data/2017-08-02_t014_myo01_acceleration.dat', sep=",").values
gyro = pd.read_csv('EMG accelerometer data/2017-08-02_t014_myo01_gyro.dat', sep=",").values
myo = pd.read_csv('EMG accelerometer data/2017-08-02_t014_myo01_myo.dat', sep=",").values
rot = pd.read_csv('EMG accelerometer data/2017-08-02_t014_myo01_rotation.dat', sep=",").values

def parsing(array):
    new_name = np.zeros([len(array), len(array[2][0].split(" "))])
    for time in range(2, len(array)):
        strver = array[time]
        new_name[time,:] = strver[0].split(" ")
    return new_name

p_acc = parsing(acc)
p_gyro = parsing(gyro)
p_myo = parsing(myo)
p_rot = parsing(rot)

def vectorize(array):
    new_name = np.zeros([len(array), 2])
    for time in range(len(array)):
        new_name[time,0] = (array[time,0]-array[2,0])*100
        new_name[time,1] = np.sqrt((array[time,1])**2  +(array[time,2])**2 + (array[time,3])**2)
    return new_name

v_acc = vectorize(p_acc)
print(np.shape(v_acc))
v_acc


plt.figure(figsize=(10,4))
plt.plot(v_acc[4:,], linestyle = '-', color = 'k', linewidth = '0.5')
plt.show()
