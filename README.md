# ECG_Denoising

Machine learning based denoising of ECG signals from wearable devices 

Datasheet provides preprocessing of data downloaded from MIT-BIH

 Adding Gaussian Noise
 Adding Baseline Wander
 Adding EMG noise (Inactivated atm)
 
Autoencoders are currently being trained

 Basic autoencoder can be found in trainaeecg.py
 Stacked denoising autoencoders under trainsaeecg.py
 Parameters are stored under trainedparams file which is needed for testing the models
 
Recurrentl Neural Networks are to be committed soon

 Will begin with simple LSTM 
  
