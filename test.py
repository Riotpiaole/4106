import torch

import torch.nn as nn
from torch.autograd import Variable

from utils import ycrcb2rgb


def adverstial_training(batch, discrminator):
    x, cr, cb, y, _ = batch
    x = x.cuda()
    y_disc = discrminator.forward(x)
    X_disc = torch.from_numpy(ycrcb2rgb(y_disc, cr, cb)).float()
    return X_disc, y


def test_dense(testing_data_loader,
               optimizer,
               model,
               criterion,
               epoch,
               test_log,
               GPU=True):
    model.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testing_data_loader:
        if GPU:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += nn.functional.nll_loss(output, target).item()
        pred = output.data.max(1)[1]
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss /= len(testing_data_loader)
    nTotal = len(testing_data_loader.dataset)
    err = 100. * incorrect / nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    test_log.write('{},{:.4f},{:.0f}\n'.format(epoch, test_loss, err))
    test_log.flush()


def test_msrn(testing_data_loader,
              optimizer,
              model,
              criterion,
              epoch,
              test_log,
              GPU=True,
              discriminator=None):

    model.eval()
    incorrect = 0
    test_loss = 0
    for batch in testing_data_loader:
        target_loc = len(batch) - 2 if discriminator else len(batch) - 1
        data = batch[0]
        if discriminator:
            data, _ = adverstial_training(batch, discriminator)

        target = batch[target_loc]
        data, target = Variable(data), Variable(target)
        if GPU:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        if model.name == "densenet64":
            test_loss += nn.functional.nll_loss(output, target).item()
            pred = output.data.max(1)[1]
            incorrect += pred.ne(target.data).cpu().sum()
            test_loss /= len(testing_data_loader)
            nTotal = len(testing_data_loader.dataset)
            err = 100. * incorrect / nTotal
            print(
                '\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
                    test_loss, incorrect, nTotal, err))

            test_log.write('{},{:.4f},{:.0f}\n'.format(epoch, test_loss, err))

        else:
            test_loss += nn.L1Loss(
                reduction='elementwise_mean')(
                output, target)

            test_loss /= len(testing_data_loader)
            print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

            test_log.write('{},{:.4f}\n'.format(epoch, test_loss))
