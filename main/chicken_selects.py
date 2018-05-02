'''
Data retreival and preprocessing for specific training environments

Written by Woochan H.

This module allows you to conveniently open and process data objects for specific environment settings.
It will automatically set CUDA, file directory settings accordingly with minimal effort.

The EnvSetter class sets the details of the training environment.
The WFDBData and EMGData class opens each dataset. They contain methods that allow you to open
all the data contained in the folder and concat it to one variable, or each file individually if you wish.
The WFDBData also has methods that allow you to add gaussian noise / baseline_wander.
Processor contains methods of reorganizing data in a way appropriate for neural networks.

The Data class inherits all the methods of the other classes, you should create a class
instance for the Data class and use the methods from the parent classes.

I tried my best to annotate each method and make them flexible. However, if you keep to the same file structure
as the /main file I uploaded, you should be able to use the default settings without having to change much.
Ideally, this file should be kept within the /main file as well.

Please refer to sample_sae.py and sample_drdae.py for example use.
'''

import os
import wfdb
import torch
import numpy as np

# Describing training environment
class EnvSetter(object):

    # Default cuda is False. Default filepath is current dir
    def __init__(self, cuda = False):
        self.cuda = cuda
        self.filepath = os.getcwd()

    # Set filepath to path of specific file
    def set_filepath(self, name):
        for root, dirs, files in os.walk(os.getcwd()):
            if name in files:
                self.filepath = os.path.join(root, name)

    # Only use this if you wish to change the host path for some reason. Default is set to my setting.
    def change_filepath(self, new_path):
        self.filepath = new_path
        print("Parent filepath changed. Consider Resetting ECG,EMG,ACC paths")

    # Use to set back to current dir
    def default_filepath(self):
        self.filepath = os.getcwd()
        print("Parent filepath changed. Consider Resetting ECG,EMG,ACC paths")

    def get_filepath(self):
        return self.filepath

    # Set CUDA setting. Default is False
    def cuda_on(self):
        self.cuda = True
        print("CUDA enabled")

    def cuda_off(self):
        self.cuda = False
        print("CUDA disabled")


# Object for getting data from MIT-BIH Arrythmia Database
class WFDBData(EnvSetter):

    def __init__(self):
        super(WFDBData, self).__init__()
        self.ecgpath = self.filepath

    def set_ecg_filepath(self, filepath = 'ecgdata'):
        self.ecgpath = str(self.get_filepath()) + '/' + str(filepath)
        print('ECGs will be imported from {}'.format(self.ecgpath))

    def get_ecg_filepath(self):
        return self.ecgpath

    # Pull all data using wfdb format. Use pull_signal to get raw data.
    def pull_wfdb(self, filename, t0 = 0, tf = int(30 * 360 * 60)):
        return wfdb.rdsamp("{}/{}".format(self.get_ecg_filepath(), filename), sampfrom = t0, sampto = tf, channels = [0])

    # Search and extract all data in the file_path using wfdb format. Use pull_all_signal to extract signal.
    def pull_all_wfdb(self, t0 = 0, tf = int(30 * 360 * 60)):
        items = os.listdir(self.get_ecg_filepath())
        items.sort()
        newlist = []
        namelist = []
        for name in items:
            if name.endswith(".dat"):
                namelist.append(name)
                name = name[:-4]
                dat = wfdb.rdsamp("{}/{}".format(self.get_ecg_filepath(), name), sampfrom = t0, sampto = tf, channels = [0])
                newlist.append(dat)
        print('These are the files[ECG] opened from the dir: {}'.format(namelist))
        return newlist

    # Save raw ecg signal from the wfdb format. 30 min each. 360Hz. 11bit. 10mV.
    def pull_ecg(self, filename, t0 = 0, tf = int(20 * 360 * 60)):
        temp_data = self.pull_wfdb(filename, t0, tf)
        if type(temp_data) == tuple:
            signal = temp_data[0]
        else:
            signal = temp_data.p_signals
        out_data = np.reshape(signal,(1, tf-t0))
        return out_data

    # Output raw ecg signal from all .dat files in file_path.
    def pull_all_ecg(self, t0 = 0, tf = int(30 * 360 * 60)):
        output = []
        for temp_data in self.pull_all_wfdb(t0, tf):
            if type(temp_data) == tuple:
                signal = temp_data[0]
            else:
                signal = temp_data.p_signals
            output.append(signal)
        output = np.array(output)
        output = np.reshape(output, (1, np.shape(output)[0]*np.shape(output)[1]))
#        print('Created signal(numpy array) with shape: {}'.format(np.shape(output)))
        return output

    # Set parameters for guassian noise
    def set_gaussian(self, signal, amp = 1, u = 0, std = 0.01):
        self.gaussian = np.multiply(amp, np.random.normal(u, std, len(signal)))

    def get_gaussian(self):
        return self.gaussian

    # Set parameters for baseline wander
    def set_baseline_wander(self, signal, respf = 12, amp = 0.05):
        N = float(len(signal))
        resppm = (float(respf)/60)
        ix = np.arange(N)
        self.baseline_wander = amp*np.sin(2*np.pi/(resppm*(N/2))*ix)

    def get_baseline_wander(self):
        return self.baseline_wander

    # Noisy signal with Gaussian and BW. Must be called after setting both
    def noisy_signal(self, raw_signal):
        return raw_signal + self.get_gaussian() + self.get_baseline_wander()


# Object for opening EMG and ACC data_csv format created from Joshua's Matlab script
class EMGData(EnvSetter):

    def __init__(self):
        super(EMGData, self).__init__()
        self.emgpath = self.filepath
        self.accpath = self.filepath
        self.opened_emg = 0
        self.opened_acc = 0

    # Specify filepath for EMG within motherpath
    def set_emg_filepath(self, filepath = 'emgdata_final'):
        self.emgpath = str(self.get_filepath()) + '/' + str(filepath)
        print('EMGs will be imported from {}'.format(self.emgpath))

    def get_emg_filepath(self):
        return self.emgpath

    # Specify filepath for ACC withint motherpath
    def set_acc_filepath(self, filepath = 'accdata_final'):
        self.accpath = str(self.get_filepath()) + '/' + str(filepath)
        print('ACCs will be imported from {}'.format(self.accpath))

    def get_acc_filepath(self):
        return self.accpath

    # Pull individual signals
    def pull_emg(self, filename, t0, tf):
        return np.genfromtxt("{}/{}.csv".format(self.get_emg_filepath(), filename), delimiter = ',')[t0:tf]

    def pull_acc(self, filename, t0, tf):
        return np.genfromtxt("{}/{}.csv".format(self.get_acc_filepath(), filename), delimiter = ',')[:,t0:tf]

    def pull_all_emg(self, t0 = 0, tf = 10000):
        file_path = self.get_emg_filepath()
        motionlist = ['motion1', 'motion2', 'motion3', 'motion4']
        newlist = []
        c = 0
        # Open each motion file and all data files within each. Concat all to newlist
        for motion in motionlist:
            motion_path = file_path + '/' + motion
            items = os.listdir(motion_path)
            items.sort()
            for name in items:
                if name.endswith(".csv"):
                    name = name[:-4]
                    data = self.pull_emg(filename = motion + '/' + name, t0 = 0, tf = tf)[:,1]
                    data[0] = 0
                    if len(data) != tf:
                        print("Not enough data: ", len(data))
                    newlist.append(data)
                    c += 1
            print("EMG: Loaded {}".format(motion))
        self.opened_emg = c
        signal = np.reshape(np.array(newlist), (1, -1))
        return signal

    def pull_all_acc(self, t0 = 0, tf = 10000):
        file_path = self.get_acc_filepath()
        motionlist = ['motion1', 'motion2', 'motion3', 'motion4']
        newlist = []
        c = 0
        # Open each motion file and all data files within each. Concat all to newlist
        for motion in motionlist:
            motion_path = file_path + '/' + motion
            items = os.listdir(motion_path)
            items.sort()
            for name in items:
                if name.endswith(".csv"):
                    name = name[:-4]
                    data = self.pull_acc(filename = motion + '/' + name, t0 = 0, tf = tf)
                    if len(data) != tf:
                        print("Not enough data: ", len(data))
                    newlist.append(data)
                    c += 1
            print("ACC: Loaded {}".format(motion))
        self.opened_acc = c
        signal = np.reshape(np.array(newlist), (-1, 3)).transpose()
        return signal


# Common data processing for neural network training
class Processor(EnvSetter):

    def __init__(self):
        super(Processor, self).__init__()

    # Change data format to specified form / length
    # Format 1: <ECG..., X..., Y..., Z...>; each * feature_len
    # Total lenght being 4 * feature_len
    # Format 2: <Sample num, 4, feature_len>
    def reformat(self, data, data_form = 0, feature_len = 0):
        self.format = data_form
        self.feature_len = feature_len

        if data_form == 0 or feature_len == 0:
            print("Please specify format type and length")
        else:
            if data_form == 1:
                output = self.format1(data, feature_len)
            elif data_form == 2:
                output = self.format2(data, feature_len)
            else:
                print("Undefined format type")
        return output

    # Revert output to numpy of original format. Accepts numpy not tensor object
    # To be used to plot denoised results
    def undo_reformat(self, data):
        if self.format == 1:
            output = self.undo_format1(data, self.feature_len)
        elif self.format == 2:
            output = self.undo_format2(data, self.feature_len)
        else:
            print('Unalbe to undo format')
        return output

    # These functions should not be called directly
    def format1(self, input_arr, feature_len):
        l = int(np.shape(input_arr)[1])
        k = int(feature_len)
        sample_num = int(l / feature_len)
        output = np.zeros((sample_num, 4*k))
        for i in range(0,sample_num):
            output[i, 0:k] = input_arr[0, k*i:k*(i+1)]
            output[i, k:2*k] = input_arr[1, k*i:k*(i+1)]
            output[i, 2*k:3*k] = input_arr[2, k*i:k*(i+1)]
            output[i, 3*k:4*k] = input_arr[3, k*i:k*(i+1)]
        return np.array(output)

    def undo_format1(self, npver, feature_len):
        k = int(feature_len)
        sample_num = np.shape(npver)[0]
        sig_len = sample_num*k
        output = np.zeros((4,sig_len))
        for i in range(0,sample_num):
            output[0, k*i:k*(i+1)] = npver[i, 0:k]
            output[1, k*i:k*(i+1)] = npver[i, k:2*k]
            output[2, k*i:k*(i+1)] = npver[i, 2*k:3*k]
            output[3, k*i:k*(i+1)] = npver[i, 3*k:4*k]
        return np.array(output)

    def format2(self, input_arr, feature_len):
        l = int(np.shape(input_arr)[1])
        k = int(feature_len)
        sample_num = int(l / feature_len)
        output = np.zeros((sample_num, 4, k))
        for i in range(0,int(sample_num)):
            output[i] = input_arr[:, k*(i):k*(i+1)]
        return np.array(output)

    def undo_format2(self, npver, feature_len):
        k = int(feature_len)
        sample_num = np.shape(npver)[0]
        sig_len = sample_num*k
        output = np.zeros((4, sample_num*k))
        for i in range(1,sample_num):
            output[:, k*i:k*(i+1)] = npver[i]
        return np.array(output)

    # Formating into tensors usable in PyTorch. This is currently NOT wrapped in a Variable
    def to_tensor(self, data):
        return torch.from_numpy(data).float()

    def to_numpy(self, tensor):
        return tensor.data.numpy()


# Used to create Input/Label data for NN models
class Data(WFDBData, EMGData, Processor):

    def __init__(self, model, motion, noiselevel, cuda = False):
        self.model = model
        self.motion = motion
        self.noiselevel = noiselevel
        WFDBData.__init__(self)
        EMGData.__init__(self)
        EnvSetter.__init__(self, cuda)

    def data_splitter(self, input_data, label_data, shuffle = True, ratio = 4):
        if np.shape(input_data) != np.shape(label_data):
            print("Data dimensions doesn't match")
        elif np.shape(input_data)[0] % (ratio + 1) != 0:
            print("Ratio({}:1) does not fit with data dimension({})".format(ratio, np.shape(input_data)))
        else:
            sample_num = np.shape(input_data)[0]
            tuple, train_set, test_set = [], [], []

        for sample in range(sample_num):
            tuple.append((input_data[sample], label_data[sample]))

        if shuffle == True:
            np.random.seed(0)
            rdn_idx = np.random.choice([1,0], size = (sample_num, ), p = [1-1./(ratio+1), 1./(ratio+1)])
            for i in range(sample_num):
                if rdn_idx[i] == 1:
                    train_set.append(tuple[i])
                else:
                    test_set.append(tuple[i])
        else:
            [train_set, test_set] = np.vsplit(tuple, [sample_num*(1-1./(ratio+1))])

        return np.array(train_set), np.array(test_set)
