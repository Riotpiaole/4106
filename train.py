import os
import sys

import argparse
import torch

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from data_utils import DataSets
from models.msrn_torch import MSRN
from models.densenet import DenseNet

global net


def main(
        GPU=True,
        epochs=20,
        lr=0.0001,
        momentum=0.9,
        weight_decay=1e-4,
        model_name="msrn",
        optimizer="adam"):

    assert model_name.lower() in [
        "msrn", "densenet"], "Model not available only support msrn and densenet"

    net = model_name.lower()

    model_name = model_name.lower()

    # loading all of the data
    training_data_loader = DataLoader(
        dataset=DataSets(),
        batch_size=32)

    testing_data_loader = DataLoader(
        dataset=DataSets(dataset='test'),
        batch_size=5)

    validation_data_loader = DataLoader(
        dataset=DataSets(dataset='val'),
        batch_size=5)
    print("==================================================")
    # checking for model type
    print("Model: " + model_name + " with loss: ", end="")

    if model_name == "msrn":
        model = MSRN()
        criterion = nn.L1Loss(reduction='elementwise_mean')
        print(" L1 loss")
    else:
        model = DenseNet()
        # this can be tested with cross entropy
        # criterion = nn.CrossEntropyLoss
        criterion = nn.functional.nll_loss
        print(" negative Log loss")

    # check for GPU support
    print("Using GPU:  " + str(GPU))
    if GPU:
        model = nn.DataParallel(model).cuda()
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
    model.summary()
    print("==================================================")
    log_folder = 'Logs'

    # Loggering the training loss
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)

    train_log = open(os.path.join(log_folder, 'train.csv'), 'w')
    test_log = open(os.path.join(log_folder, 'test.csv'), 'w')
    val_log = open(os.path.join(log_folder, 'val.csv'), 'w')

    # Training in numbers of epochs
    for epoch in range(0, epochs):
        try:  # try to catch interruption during training
            train(
                training_data_loader,
                testing_data_loader,
                validation_data_loader,
                optimizer,
                model,
                criterion,
                epoch,
                train_log,
                test_log,
                val_log,
                GPU)
        except KeyboardInterrupt:
            pass
            # save_checkpoint(model, epoch, model_name)

        # save_checkpoint(model, epoch, model_name)

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


def train(
        training_data_loader,
        test_data_loader,
        validation_data_loader,
        optimizer,
        model,
        criterion,
        epoch,
        train_log,
        test_log,
        val_log,
        GPU=True):
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

    model.train()
    nProcessed = 0
    for iteration, batch in enumerate(training_data_loader, 1):

        input, label = Variable(
            batch[0]), Variable(
            batch[1], requires_grad=False)

        size = len(batch[0])

        # Training the Network
        if GPU:
            input = input.cuda()
            label = label.cuda()

        output = model(input)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        nProcessed += len(input)
        progress = epoch + iteration / len(training_data_loader) - 1

        if iteration % 10 == 0:
            print(
                '=>Train Epoch {}: {:.2f} [{}/{} ({:.0f}%)]'.format(
                    epoch,
                    progress,
                    nProcessed,
                    len(training_data_loader.dataset),
                    100. *
                    iteration /
                    len(training_data_loader)),
                end="")
            if net == "dense":
                dense_loggering(loss, output, label, size)
            else:
                msrn_loggering(loss, output, label, size)


def dense_loggering(loss, output, target, size):
    pred = output.data.max(1)[1]  # get index of max log probability
    incorrect = pred.ne(target.data).cpu().sum()
    err = 100. * incorrect / size
    print('\tLoss {:.6f}\tError {:.6f}'.format(
        loss.data[0], err))


def msrn_loggering(loss, output, target, size):
    print("\tLoss: {:.6f}".format(loss.data[0]))


if __name__ == "__main__":
    main(GPU=False, model_name='densenet')

    del datasets
