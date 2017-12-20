# Main pipeline for CNN training

'''
Data Input = MIT-BIH ECG data, EMG data

Noise Generation (noisegen.py)
    1. Base ECG of length T
    2. Baseline wander of length T
    3. Gaussian noise of length T
    4. Added EMG noise from t1 to t2 with amplitude A

Accelorometer Data Generation
    1. Base: Null list[(x0,y0,z0),...] of length T. dtype == float
    2. Add gaussian noise to Base.
    3. From t1 to t2, add movement corresponding to EMG amplitude A.
        ** How does x,y,z relate to each other?
        ** Will be too fake if I create this based on a simple functiona
            > better to have real data once Accelorometer arrives

Preprocessing data
    Apply basic Moving average filter (SciPy)
    Then A or B. (A first, try B later)

        A. Discrete Fourier Transformation
            1. SciPy package
            2. Decide hyperparameters: Period = approximate of one heartbeat.
            3. Numpy stack, third axis = time

        B. Discrete Wavelet Transformation
            1. PyWavelets package
            2. Decide hyperparameters: depth, wavelet type
            3. Output data in the form of ML choosing the threshold of DWT ???

CNN structure(PyTorch)

    Hyperparameters: 2-3 layer depth, L1(element wise comparison) loss or MSE loss function,
                    Adam optimizer, ReLu

        Regression between Fourier_ecg_with_noise and Accelorometer data
                        |
        Filter trained with convolutional neural network
                        |
        Inverse discrete Fourier transformation of filtered data
                        |
        Loss fucntion against pre-noisegen data


    **Also try with a simple SVM(support vector machine)?


'''
