# CIFAR-10 Image Classification

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', and 'truck'. The data was loaded through the Pytorch library. The objective of this project is to train the computer to correctly classify un-labelled images.   

A convolutional neural network was trained, using Pytorch and Optuna for hyperparameter tuning. The best model achieved an accuracy of 71.2% on test data.


<u>Guide to file structure:</u>
- the model development folder contains .py files which were used to build the CNN in Pycharm. The same code has been reproduced in the Convolutional Neural Network.ipynb file, for easier viewing on GitHub.
- cnn_SGD_nesterov.pth is the trained model
- mean.pt and std_dev.pt (outputs from Data Exploration.ipynb) store the mean and standard deviation of each channel (R, G, B) of the images. This is used to facilitate image normalisation.