# ECG_Denoising
### iBSc project for denoising wearable ECG signals

This project is focused on developing a machine learning approach to denoising EMG noise from ECGs collected from wearable devices. 
The hypothesis is that data from an accelerometer can provide useful information about the EMG noise which can be used in denoising. 

The clean ECG data has been collected from the MIT-BIH Arrythmia Database[1]. The noisy signal has been produced by adding EMG data 
collected using a myo-band[2] and the myo-data-capture package[3] on top of the clean ECG. The acceleration data has been collected 
with the EMG data. The current sample uploaded is from a male subject performing flexion_extension motion of the elbow. The EMG was 
collected from the anterior aspect of the bicepts at the midpoint between the acromioclavicular joint and the antecubital fossa. 
As the motion arising from the flexion of the biceps are in the forearm, the accelerometer[4] was placed on the forearm, 10cm from the antecubital fossa. 
The data was resampled to match the sampling frequency of the myo-band[2].  

The module "chicken_selects" contains tools for opening and preprocessing the data. If the file structure is maintained, it should be able to automatically locate the relevant files easily. Otherwise, there are methods that allow to adjust this as appropriate. The source code has been annotated fairly well to allow easy use. 

Also refer to the two sample codes(sample_sae.py, sample_drdae.py) for initial attempts to train a Neural Network model using this data. 
To allow result plotting on local display while training on remote machine, enable x11 backend when prompted. 

**Recent Changes:** Mini-batch training has been implemented for sample_sae.py. Use chicken_selects.data_splitter() to separate test data from train data.

Implementation of tensorboard/pytorch_ver are to follow shortly.



### References

[1] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. 
    PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. 
    Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/content/101/23/e215.full]; 2000 (June 13).

[2] https://www.myo.com/

[3] https://market.myo.com/app/55009793e4b02e27fd3abe79/myo-data-capture%29

[4] LIS3DH 3-axis MEMS accelerometer
