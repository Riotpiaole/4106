import os
import sys

import argparse
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from data_utils import DataSets, datasets

from models.msrn_torch import MSRN
from models.densenet import DenseNet

from test import (
    test_msrn,
    test_dense,
    adverstial_training,
)

import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def main(
        GPU=True,
        epochs=20,
        lr=0.0001,
        momentum=0.9,
        weight_decay=1e-4,
        batch_size=32,
        start_epochs=0,
        model_name="msrn",
        optimizer="adam",
        naive=False):

    assert model_name.lower() in [
        "msrn", "densenet", "densenet64"], "Model not available only support msrn and densenet"

    discriminator = None

    model_name = model_name.lower()
    dense = True if model_name == "densenet" else False

    # loading all of the data
    training_data_loader = DataLoader(
        dataset=DataSets(dense=dense),
        batch_size=batch_size)

    testing_data_loader = DataLoader(
        dataset=DataSets(dense=dense, dataset='test'),
        batch_size=5)

    # validation_data_loader = DataLoader(
    #     dataset=DataSets(dataset='val'),
    #     batch_size=5)

    print("==================================================")
    # checking for model type
    print("Model: " + model_name + " with loss: ", end="")

    if model_name == "msrn":
        model = MSRN()
        criterion = nn.L1Loss(reduction='elementwise_mean')
        print(" L1 loss")
    elif model_name == "densenet64":
        model = DenseNet()
        discriminator = MSRN()
        criterion = nn.functional.nll_loss
    elif model_name == "densenet":
        model = DenseNet()
        # this can be tested with cross entropy
        # criterion = nn.CrossEntropyLoss
        criterion = nn.functional.nll_loss
        print(" negative Log loss")
    else:
        raise ValueError(
            "Invalid model_name not support {}".format(model_name))

    # check for GPU support
    print("Using GPU:  " + str(GPU))
    if GPU:
        model = nn.DataParallel(model).cuda()
        if model_name == "msrn":
            criterion = criterion.cuda()
    else:
        model = model.cpu()

    print("Optimizer: " + optimizer + " with lr: " + str(lr))
    # setting up optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay)
    elif optimizer == "sgd":
        # TODO add momentum flag
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay)
    elif optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay)
    else:
        raise ValueError("Not supported Loss function")
    print("==================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'msrn':
        MSRN().to(device).summary()
    elif model_name == 'densenet64':
        DenseNet(upscale=2).to(device).summary()
    else:
        DenseNet().to(device).summary()
    print("==================================================")
    log_folder = 'Logs/' + model_name

    # Loggering the training loss
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    model.name = model_name
    append_write = 'w' if os.path.exists(log_folder) else 'a'

    train_log = open(os.path.join(log_folder, 'train.csv'), append_write)
    test_log = open(os.path.join(log_folder, 'test.csv'), append_write)
    val_log = open(os.path.join(log_folder, 'val.csv'), append_write)

    if start_epochs > 0:
        model = load_model(model, model_name, start_epochs + 1)

    # Training in numbers of epochs
    for epoch in range(start_epochs, epochs):
        train(
            training_data_loader,
            optimizer,
            model,
            criterion,
            epoch,
            train_log,
            GPU,
            discriminator if discriminator else None,
            naive)  # if running on 64`
        # densenet
        print('testing the model')
        if model_name == "densenet":
            test_dense(testing_data_loader,
                       optimizer,
                       model,
                       criterion,

                       epoch,
                       test_log,
                       GPU)
        else:
            test_msrn(testing_data_loader,
                      optimizer,
                      model,
                      criterion,
                      epoch,
                      test_log,
                      GPU,
                      discriminator=discriminator if discriminator else None)
        save_checkpoint(model, epoch, model_name)

    train_log.close()
    test_log.close()
    val_log.close()


def save_checkpoint(model, epoch, model_dir):
    model_folder = "Weights/" + model_dir + "/"
    model_out_path = model_folder + "{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {} ".format(model_out_path))


def load_model(model, model_dir, epochs=1158):
    model_path = "Weights/" + model_dir + "/" + str(epochs - 1) + ".pth"
    assert os.path.isfile(
        model_path), "= no model found at '{}'".format(model_path)
    print(
        "= loading pretrianed model '{}' epochs {} ".format(
            model_dir, epochs))
    weights = torch.load(
        model_path,
        map_location=torch.device('cpu'))
    model.load_state_dict(weights['model'].state_dict())
    return model


def train(
        training_data_loader,
        optimizer,
        model,
        criterion,
        epoch,
        train_log,
        GPU=True,
        discriminator=None,
        naive=False):
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

    model.train()
    nProcessed = 0

    # Loading discrminator for superresolutioning the image
    if discriminator:
        # loading model_weights
        try:
            discriminator = nn.DataParallel(discriminator).cuda()
            discriminator = load_model(discriminator, 'msrn')
        except FileNotFoundError:
            raise FileNotFoundError(
                "No pretrainied Model Found. Please trained MSRN first")

    for iteration, batch in enumerate(training_data_loader, 1):

        if model.name == "densenet64":
            input, label = adverstial_training(batch, discriminator)
            input, label = Variable(input),\
                Variable(label, requires_grad=False)
        else:
            input, label = Variable(batch[0]),\
                Variable(batch[len(batch) - 1], requires_grad=False)

        size = len(batch[0])

        # Training the Network
        if GPU:
            input = input.cuda()
            label = label.cuda()

        output = model(input)
        loss = criterion(output, label)
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        nProcessed += len(input)
        progress = epoch + iteration / len(training_data_loader) - 1

        if iteration % 10 == 0:
            print(
                '=>Train Epoch {}: {:.4f} [{}/{} ({:.0f}%)]'.format(
                    epoch,
                    progress,
                    nProcessed,
                    len(training_data_loader.dataset),
                    100. *
                    iteration /
                    len(training_data_loader)), end="")
            train_log.write('{:.2f}'.format(progress))
            if model.name == "densenet" or model.name == "densenet64":
                dense_loggering(loss, output, label, size, train_log)
            else:
                msrn_loggering(loss, output, label, size, train_log)
            train_log.flush()


def dense_loggering(loss, output, target, size, train_log):
    pred = output.data.max(1)[1]  # get index of max log probability
    incorrect = pred.ne(target.data).cpu().sum()
    err = 100. * incorrect / size
    print(' Loss {:.6f} Error {:.6f}'.format(
        loss.item(), err), end="\r")
    train_log.write('{},{}\n'.format(loss.item(), err))


def msrn_loggering(loss, output, target, size, train_log):
    print(" Loss: {:.6f}".format(loss.item()), end="\r")
    train_log.write(',{:.6f}\n'.format(loss.item()))


if __name__ == "__main__":
    main(
        GPU=True,
        batch_size=64,
        start_epochs=0,
        model_name='densenet64',
        epochs=20)
    del datasets
