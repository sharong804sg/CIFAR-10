import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

my_mean = torch.load('mean.pt')
my_std = torch.load('std_dev.pt')

def get_CIFAR(train):
    """
    Function to download and transform CIFAR dataset
    :param train: Boolean value. If True, return training dataset. If False return test dataset.
    :param mean: sequence of means for each channel, to be used for normalisation
    :param std_dev: sequence of std deviations for each channel, to be used for normalisation
    :return: Dataset
    """
    my_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(my_mean, my_std)])

    my_cifar = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=my_transform)

    return my_cifar


def get_train_val_dataloader(training_dataset, my_batchsize, my_seed = None):
    """
    Function to split training data into training and validation subsets and format as dataloaders
    Model performance on validation set will be used for hyperparameter tuning.

    :param training_dataset: full set of training data, in pytorch Dataset format
    :param my_batchsize: batch size for pytorch DataLoader
    :param my_seed: optional seed to be used for train test split random state

    :return: tuple of pytorch DataLoaders - train_dataloader, val_dataloader
    """

    # separate into training & validation datasets
    total_len = len(training_dataset.data)
    train, val = torch.utils.data.random_split(dataset=training_dataset, lengths=[int(0.8*total_len), int(0.2*total_len)])

    #format as pytorch dataloader
    train_dataloader = DataLoader(train, batch_size=my_batchsize, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=my_batchsize)

    return train_dataloader, val_dataloader

def count_correct(predictions, y):
    """
    Counts number of correct predictions in a batch
    :param predictions: 1D tensor with predictions
    :param y: 1D tensor with true classes
    :return: number of correct predictions (pred==y)
    """
    predictions = predictions.numpy()
    y = y.numpy()

    n_correct = (predictions == y).sum()

    return n_correct