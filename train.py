import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import DataSets
from torch.autograd import Variable
from models.msrn_torch import MSRN
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


def main(
        GPU=True,
        criterion=nn.L1Loss(
            size_average=True),
        epochs=20,
        lr=0.0001):
    training_data_loader = DataLoader(
        dataset=DataSets(),
        batch_size=16)
    model = MSRN()
    if GPU:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr)

    for epoch in range(0, epochs):
        train(training_data_loader, optimizer, model,
              criterion,
              epoch, GPU)
        save_checkpoint(model, epoch)


def save_checkpoint(model, epoch):
    model_folder = "Weights/"
    model_out_path = model_folder + "{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {} ".format(model_out_path))


def train(training_data_loader, optimizer, model, criterion, epoch, GPU=True):
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input, label = Variable(
            batch[0]), Variable(
            batch[1], requires_grad=False)

        if GPU:
            input = input.cuda()
            label = label.cuda()

        sr = model(input)
        loss = criterion(sr, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print("Loss {} {} ".format(
                loss.data[0], iteration + (epoch - 1) * len(training_data_loader)))
            print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch,
                                                               iteration, len(training_data_loader), loss.data[0]))


if __name__ == "__main__":
    main(GPU=False)
