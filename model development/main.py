# Environment: pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna
from optuna.trial import TrialState

import functions

my_seed = 42

# GET TRAINING DATA
trainset = functions.get_CIFAR(train = True)


# TRAIN MODEL
def set_parameters(trial, my_optimizer):
    """
    Set parameters for neural network, optimisation algorithm etc.
    :param trial: Optuna trial object
    :param my_optimizer: optimizer to use - SGD / SGD_classical / SGD_nesterov / Adam

    :return: dictionary of parameters:
            - n_conv_layers: number of convolution layers in neural network
            - out_ch_conv{i}: number of output channels in convolution layer i
            - kernel_conv{i}_even: kernel width in convolution layer i - even option
            - kernel_conv{i}_odd:                                      - odd option

            - n_linear_layers: number of linear layers in neural network
            - n_units_lin{i}: number of units in linear layer i
            - dropout_lin{i}: dropout probability for linear layer i

            - lr: learning rate
            - batch_size: batch size
            - n_epochs = number of epochs (i.e. number of passes through training data during optimisation)
    """
    trial.suggest_int("n_conv_layers", 2, 2)

    for i in range(trial.params['n_conv_layers']):
        trial.suggest_int(f'out_ch_conv{i}', 1, 50)
        trial.suggest_categorical(f'kernel_conv{i}_even', [2, 4, 6])
        trial.suggest_categorical(f'kernel_conv{i}_odd', [3, 5, 7])

    trial.suggest_int("n_linear_layers", 1, 3)

    for i in range(trial.params['n_linear_layers']):
        trial.suggest_int(f'n_units_lin{i}', 1, 200)
        trial.suggest_float(f"dropout_lin{i}", 0.1, 0.9)

    trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    trial.suggest_int("batch_size", 10, 10)
    trial.suggest_int("n_epochs", 5, 5)

    trial.suggest_categorical("optimizer", [my_optimizer])
    if (my_optimizer=='SGD_classical') | (my_optimizer=='SGD_nesterov'):
        trial.suggest_float("momentum", 0.6, 0.999)
    elif my_optimizer=='Adam':
        trial.suggest_float("beta1", 0.6, 0.999)
        trial.suggest_float("beta2", 0.8, 0.999)

    my_params = trial.params

    return my_params


def define_model(my_params):
    """Defines convolutional neural network based on set parameters
    :param my_params: dictionary of parameters (see set_parameters() for full list)
    """

    layers = []

    # Define Convolution Layers
    in_ch = 3  # number of input channels = no. of channels in feature matrix = 3 (RGB)
    img_width = 32 # number of px along length & width of feature matrix
    for i in range(my_params['n_conv_layers']):
        # convolution layer
        out_ch = my_params[f'out_ch_conv{i}']  # number of output channels for this layer
        # for even image width use odd kernel width so that resulting img width is divisible by 2 during pooling
        if (img_width % 2) == 0:
            kernel_size = my_params[f'kernel_conv{i}_odd']
        else:
            kernel_size = my_params[f'kernel_conv{i}_even']
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size))

        layers.append(nn.ReLU())  # activation function
        layers.append(nn.MaxPool2d(2,2))  # pooling layer

        in_ch = out_ch  # no. of input channels for next layer = no. of output channels from this layer
        img_width = int((img_width-(kernel_size-1))/2)

    layers.append(nn.Flatten(start_dim=1))  # flatten all dimensions except batch

    # Define Linear Layers
    in_features = in_ch * img_width * img_width
    for i in range(my_params['n_linear_layers']):
        # linear layer
        out_features = my_params[f'n_units_lin{i}']
        layers.append(nn.Linear(in_features, out_features))

        layers.append(nn.ReLU())  # activation function

        #drop-out regularisation
        p = my_params[f"dropout_lin{i}"]
        layers.append(nn.Dropout(p))

        in_features = out_features  # no. of inputs for next layer = no. of outputs of this layer

    layers.append(nn.Linear(in_features, 10))  # output layer

    return nn.Sequential(*layers)


def objective(trial, my_optimizer):
    """
    Objective for Optuna to optimise
    :param trial: Optuna trial object
    :param optimizer_name: optimizer to use
                            - SGD: SGD without momentum
                            - SGD_classical: SGD with classical momentum
                            - SGD_nesterov: SGD with nesterov momentum
                            - Adam
    :return: accuracy - fraction of correctly labelled validation points. This is what Optuna seeks to maximise
    """

    #set parameters
    my_params = set_parameters(trial, my_optimizer)

    # Instantiate model
    model = define_model(my_params)

    # Instantiate optimizer
    lr = my_params['lr']
    if my_optimizer == 'SGD':
        optimizer = getattr(optim, "SGD")(model.parameters(), lr=lr)
    elif my_optimizer == 'SGD_classical':
        momentum = my_params['momentum']
        optimizer = getattr(optim, "SGD")(model.parameters(), lr=lr, momentum=momentum)
    elif my_optimizer == 'SGD_nesterov':
        momentum = my_params['momentum']
        optimizer = getattr(optim, "SGD")(model.parameters(), lr=lr, momentum=momentum,
                                                   nesterov=True)
    elif my_optimizer == 'Adam':
        beta1 = my_params['beta1']
        beta2 = my_params['beta2']
        optimizer = getattr(optim, "Adam")(model.parameters(), lr=lr, betas=(beta1, beta2))
    else:
        raise ValueError("optimizer_name must be 'SGD' / 'SGD_classical' / 'SGD_nesterov' / 'Adam'")

    # get data
    train_dataloader, val_dataloader = functions.get_train_val_dataloader(training_dataset=trainset,
                                                                          my_batchsize=my_params['batch_size'])

    # train model
    for epoch in range(my_params['n_epochs']):

        #train
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            # X and y are tensors. X.size() = (batch_size,n_features), y.size()=(batch_size,)
            # set datatype for compatibility with nn.
            X = X.float()
            y = y.long()

            # calculate model output and resulting loss
            model_output = model(X)  # tensor. size=(batch_size x n_classes)
            loss_fn = nn.CrossEntropyLoss() # instantiate loss function
            loss = loss_fn(model_output, y)

            # Backpropagation to update model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validate. We do this at each epoch to facilitate pruning:
        # i.e. early termination of trials which are clearly not going to be optimum
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(val_dataloader):
                X = X.float()
                y = y.long()

                # calculate model output and total number of correct predictions for this batch
                model_output = model(X)
                pred = torch.argmax(model_output, dim=1)  # prediction = class with highest output value
                correct += functions.count_correct(pred, y)

        accuracy = correct / len(val_dataloader.dataset)

        # report accuracy to allow Optuna to decide whether to prune this trial
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy  # return final validation accuracy after all epochs (unless pruned)


# instantiate optuna study
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
# Optimise hyperparameters will try {n_trials} param combinations or till {timeout} seconds is hit
study.optimize(lambda trial: objective(trial, "Adam"), n_trials=1)

#display study results
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
best_trial = study.best_trial

print("  Validation Accuracy: ", best_trial.value)

print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# TRAIN FINAL MODEL USING TUNED HYPER-PARAMETERS
def train_final_model(my_params):
    """
    Train final model using tuned hyperparameters from best Optuna trial
    :param my_params: dictionary of parameters from Optuna trial object that had best validation accuracy

    :return: model
    """

    # Instantiate model
    model = define_model(my_params)

    # Instantiate optimizer
    my_optimizer = my_params['optimizer']
    lr = my_params['lr']
    if my_optimizer == 'SGD':
        optimizer = getattr(optim, "SGD")(model.parameters(), lr=lr)
    elif my_optimizer == 'SGD_classical':
        momentum = my_params['momentum']
        optimizer = getattr(optim, "SGD")(model.parameters(), lr=lr, momentum=momentum)
    elif my_optimizer == 'SGD_nesterov':
        momentum = my_params['momentum']
        optimizer = getattr(optim, "SGD")(model.parameters(), lr=lr, momentum=momentum,
                                          nesterov=True)
    elif my_optimizer == 'Adam':
        beta1 = my_params['beta1']
        beta2 = my_params['beta2']
        optimizer = getattr(optim, "Adam")(model.parameters(), lr=lr, betas=(beta1, beta2))
    else:
        raise ValueError("optimizer_name must be 'SGD' / 'SGD_classical' / 'SGD_nesterov' / 'Adam'")

    # get data
    train_dataloader = DataLoader(dataset=trainset, batch_size=my_params['batch_size'])

    # train model
    for epoch in range(my_params['n_epochs']):
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            # set datatype for compatibility with nn.
            X = X.float()
            y = y.long()

            # calculate model output and resulting loss
            model_output = model(X)  # tensor. size=(batch_size x n_classes)
            loss_fn = nn.CrossEntropyLoss()  # instantiate loss function
            loss = loss_fn(model_output, y)

            # Backpropagation to update model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

best_params = best_trial.params
final_model = train_final_model(best_params)


# EVALUATE FINAL TRAINING ACCURACY
def predict_and_evaluate(model, my_dataset):
    """
    Function to run trained and tuned model on provided dataframe to obtain predictions and evaluate
    accuracy

    :param model: trained model
    :param my_dataset: dataset including features and target/label

    :return: accuracy
    """
    my_dataloader = DataLoader(my_dataset, batch_size=10, shuffle=False)

    model.eval()
    correct = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(my_dataloader):
            X = X.float()
            y = y.long()

            # calculate model output and total number of correct predictions for this batch
            model_output = model(X)
            pred = torch.argmax(model_output, dim=1)  # prediction = class with highest output value
            correct += functions.count_correct(pred, y)

    accuracy = correct / len(my_dataloader.dataset)

    return accuracy


train_acc = predict_and_evaluate(final_model, trainset)
print(f"  Final Training Accuracy: {train_acc}")

# EVALUATE ACCURACY ON TEST DATA
testset = functions.get_CIFAR(train = False)
test_acc = predict_and_evaluate(final_model, testset)
print(f"  Test Accuracy: {test_acc}")

# SAVE FINAL MODEL
torch.save(final_model, 'cnn_' + best_params['optimizer'] + '.pth')