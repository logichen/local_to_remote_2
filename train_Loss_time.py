# -*-coding:utf-8-*-
import argparse
import logging
import yaml
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict
from models import *

from utils import Logger, count_parameters, data_augmentation, \
    load_checkpoint, get_data_loader, mixup_data, mixup_criterion, \
    save_checkpoint, adjust_learning_rate, get_current_lr
from torch.optim.optimizer import Optimizer, required



from torch.nn import _reduction as _Reduction
import warnings


parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
parser.add_argument('--work-path', required=True, type=str)
parser.add_argument('--resume', action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args()
logger = Logger(log_file_name=args.work_path + '/log.txt',
                log_level=logging.DEBUG, logger_name="CIFAR").get_log()



def _get_softmax_dim(name, ndim, stacklevel):
    # type: (str, int, int) -> int
    warnings.warn("Implicit dimension choice for {} has been deprecated. "
                  "Change the call to include dim=X as an argument.".format(name), stacklevel=stacklevel)
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    # type: (Tensor,     Optional[int], int, Optional[int]) -> Tensor
    if dim is None:
        dim = _get_softmax_dim('log_softmax', input.dim(), _stacklevel)

    if dtype is None:

        ret = input.log_softmax(dim)

    else:

        ret = input.log_softmax(dim, dtype=dtype)

    return ret


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,

             reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    dim = input.dim()

    if dim < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'

                         .format(input.size(0), target.size(0)))

    if dim == 2:

        ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)

    elif dim == 4:

        ret = torch._C._nn.nll_loss2d(input, target, weight, _Reduction.get_enum(reduction), ignore_index)

    else:

        # dim == 3 or dim > 4

        n = input.size(0)

        c = input.size(1)

        out_size = (n,) + input.size()[2:]

        if target.size()[1:] != input.size()[2:]:
            raise ValueError('Expected target size {}, got {}'.format(

                out_size, target.size()))

        input = input.contiguous().view(n, c, 1, -1)

        target = target.contiguous().view(n, 1, -1)

        reduction_enum = _Reduction.get_enum(reduction)

        if reduction != 'none':

            ret = torch._C._nn.nll_loss2d(

                input, target, weight, reduction_enum, ignore_index)

        else:

            out = torch._C._nn.nll_loss2d(

                input, target, weight, reduction_enum, ignore_index)

            ret = out.view(out_size)

    return ret


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,

                  reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)


class _Loss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):

        super(_Loss, self).__init__()

        if size_average is not None or reduce is not None:

            self.reduction = _Reduction.legacy_get_string(size_average, reduce)

        else:

            self.reduction = reduction


class _WeightedLoss(_Loss):

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)

        self.register_buffer('weight', weight)


class CrossEntropyLoss_act(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,

                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss_act, self).__init__(weight, size_average, reduce, reduction)

        self.ignore_index = ignore_index

        self.err = 0

    def forward(self, input, target):
        # F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

        err = cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        # err = (torch.exp(err)-0.95)
        # err = 3*torch.log(torch.exp(err)-1)
        # err = (2.5*torch.atan(3*err))**2 + err

        # err = (torch.atan(20*err))**4 + err
        err = 20 * torch.atan(err) + err
        # err = 5*err
        # return cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

        return err


#new SGD

class SGD_flat(Optimizer):

    # def __init__(self, params, lr=required, momentum=0, flat=0, dampening=0,
    #              weight_decay=0, nesterov=False):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # defaults = dict(lr=lr, momentum=momentum, flat=flat, dampening=dampening,
        #                 weight_decay=weight_decay, nesterov=nesterov)
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_flat, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_flat, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, flat, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # k2 = 10
        # k1 = -1
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            # flat = group['flat']
            for p in group['params']:
                if p.grad is None:
                    continue
                # d_p = p.grad.data
                # print(d_p)
                # d_p = torch.tan(1.3*d_p)
                param_state = self.state[p]
                d_p = p.grad.data

                if 'dfdp_previous' not in param_state:
                    dfdp_pre = param_state['dfdp_previous'] = torch.clone(d_p).detach()
                else:
                    dfdp_pre = param_state['dfdp_previous']

                if flat == 0:

                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        # param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)

                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf
                    # print('mean',torch.mean(torch.sqrt(torch.pow(dfdp_pre - d_p, 2))))
                    # print('max',torch.max(torch.sqrt(torch.pow(dfdp_pre - d_p, 2))))
                    delta_p = group['lr'] * d_p
                    delta_p = torch.where(torch.sqrt(torch.pow(dfdp_pre - d_p, 2)) > 0.001, 0*d_p, delta_p)


                    # delta_p = torch.where(torch.sqrt(torch.pow(dfdp_pre - d_p, 2)) > 0.01, 0.0001 * torch.atan_(group['lr']*d_p) / 1.55, delta_p)
                    # if torch.max(torch.sqrt(torch.pow(dfdp_pre - d_p, 2))) > 0.09:
                    #     delta_p = 0.01 * torch.atan_(170*group['lr']*d_p) / 1.55
                    # else:
                    #     delta_p = 0.02 * torch.atan_(90*group['lr']*d_p) / 1.55
                    # p.data.add_(-group['lr'], d_p) #be careful with - or +
                    # p.data = p.data - group['lr'].mul(d_p)

                elif flat == 1:
                    # dfdL = k2 * torch.exp(k1 - k2 * p.data)
                    # d_p = d_p.mul(dfdL)

                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        # param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.add_(1 - dampening, d_p)

                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf
                    # p.data.add_(-group['lr'], d_p) #be careful with - or +

                    delta_p = group['lr'] * d_p

                    delta_p = torch.where(torch.sqrt(torch.pow(dfdp_pre - d_p, 2)) > 0.001, 0*d_p, 0.01 * torch.atan_(delta_p) / 1.55)

                    # delta_p = torch.where(torch.sqrt(torch.pow(dfdp_pre - d_p, 2)) > 0.01, 0.01 * torch.atan_(group['lr']*d_p) / 1.55, delta_p)
                    # if torch.max(torch.sqrt(torch.pow(dfdp_pre - d_p, 2))) > 0.1:
                    #     delta_p = 0.01 * torch.atan_(group['lr']*d_p) / 1.55
                    # else:
                    #     delta_p = 0.015 * torch.atan_(group['lr']*d_p) / 1.55
                p.data = p.data - delta_p

                param_state['dfdp_previous'] = torch.clone(d_p).detach()

        return loss



class SGD_loss(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_loss, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_loss, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, train_loss, epoch, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # print(d_p)
                # d_p = torch.tan(1.3*d_p)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)

                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                direct_loss = d_p
                direct_loss = (2 / np.pi * np.arctan(train_loss))*direct_loss
                # if epoch != 0:
                    # print('d_p:\n', d_p.abs().max().max(), d_p.abs().min().min(), d_p)
                    # print('p', p)`
                    # print('train_loss**2',train_loss**2)
                    # train_loss = 0.5*(7/np.pi*np.arctan(train_loss))**2
                    # direct_loss = train_loss*d_p

                    # train_loss = np.log(5*train_loss+1)
                    # print('train_loss:\n', train_loss)
                    # direct_loss = 0.4*d_p.sign()*train_loss/(10000*d_p.abs()+3)
                    # direct_loss = d_p*torch.log(torch.tensor(train_loss).cuda())
                    # direct_loss = 0.002*d_p*train_loss

                    # direct_loss = p.abs().mul(torch.sign(d_p)*train_loss)
                    # direct_loss = 1.0/1.507*torch.atan(direct_loss)

                    # direct_loss = train_loss**2*((-50*d_p).abs()).exp()*0.000002*d_p.sign()


                    # direct_loss = train_loss**2/(d_p.abs()+1)*0.0002*d_p.sign()
                    # print('direct_loss2:\n', direct_loss.abs().max().max(),direct_loss.abs().min().min())
                    # direct_loss = 0.00001*torch.atan(torch.tensor(100000.0)*train_loss)*torch.sign(d_p)
                    # if epoch == 2:
                        # print('d_p:\n', d_p)
                        # print('direct_loss:\n', direct_loss)

                # else:
                #     direct_loss = d_p
                if epoch !=0 & epoch % 10 == 0:
                    p.data.add_(-0.01 * group['lr'], direct_loss)
                elif epoch !=0 & epoch % 5 == 0:
                    p.data.add_(-0.1 * group['lr'], direct_loss)
                elif epoch !=0 & epoch % 2 == 0:
                    p.data.add_(-5 * group['lr'], direct_loss)
                else:
                    p.data.add_(-group['lr'], direct_loss)
                # p.data.add_(-group['lr'], direct_loss)
        return loss


def train(train_loader, net, criterion, optimizer, epoch, device):
    global writer

    start = time.time()
    net.train()

    train_loss = 15
    correct = 0
    total = 0
    logger.info(" === Epoch: [{}/{}] === ".format(epoch, config.epochs))

    # logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, config.epochs))

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        if config.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, config.mixup_alpha, device)

            outputs = net(inputs)
            loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        # zero the gradient buffers
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weight
        # optimizer.step()
        # print('epoch', epoch)
        optimizer.step(train_loss, epoch)

        # count the loss and acc
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if config.mixup:
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()

        if (batch_index + 1) % 100 == 0:
            logger.info("   == step: [{:3}/{}], train loss: {:.7f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                batch_index + 1, len(train_loader),
                train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    logger.info("   == step: [{:3}/{}], train loss: {:.7f} | train acc: {:6.3f}% | lr: {:.6f}".format(
        batch_index + 1, len(train_loader),
        train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    end = time.time()
    logger.info("   == cost time: {:.4f}s".format(end - start))
    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total

    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)

    return train_loss, train_acc


def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec, writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info(" === Validate ===".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info("   == test loss: {:.7f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('test_loss', test_loss, global_step=epoch)
    writer.add_scalar('test_acc', test_acc, global_step=epoch)
    # Save checkpoint.
    acc = 100. * correct / total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, args.work_path + '/' + config.ckpt_name)
    if is_best:
        best_prec = acc


def main():
    global args, config, last_epoch, best_prec, writer
    writer = SummaryWriter(log_dir=args.work_path + '/event')

    # read config from yaml file
    with open(args.work_path + '/config.yaml') as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)

    # define netowrk
    net = get_model(config)
    logger.info(net)
    logger.info(" == total parameters: " + str(count_parameters(net)))

    # CPU or GPU
    device = 'cuda' if config.use_gpu else 'cpu'
    # data parallel for multiple-GPU
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.to(device)

    # define loss and optimizer
    criterion = CrossEntropyLoss_act()
    # criterion = nn.CrossEntropyLoss()

    optimizer = SGD_loss(
        net.parameters(),
        config.lr_scheduler.base_lr,
        momentum=config.optimize.momentum,
        weight_decay=config.optimize.weight_decay,
        nesterov=config.optimize.nesterov)
    # optimizer = torch.optim.SGD(
    #     net.parameters(),
    #     config.lr_scheduler.base_lr,
    #     momentum=config.optimize.momentum,
    #     weight_decay=config.optimize.weight_decay,
    #     nesterov=config.optimize.nesterov)

    # resume from a checkpoint
    last_epoch = -1
    best_prec = 0
    if args.work_path:
        ckpt_file_name = args.work_path + '/' + config.ckpt_name + '.pth.tar'
        if args.resume:
            best_prec, last_epoch = load_checkpoint(
                ckpt_file_name, net, optimizer=optimizer)

    # load training data, do data augmentation and get data loader
    transform_train = transforms.Compose(
        data_augmentation(config))

    transform_test = transforms.Compose(
        data_augmentation(config, is_train=False))

    train_loader, test_loader = get_data_loader(
        transform_train, transform_test, config)

    logger.info("            =======  Training  =======\n")
    for epoch in range(last_epoch + 1, config.epochs):
        lr = adjust_learning_rate(optimizer, epoch, config)
        writer.add_scalar('learning_rate', lr, epoch)
        train(train_loader, net, criterion, optimizer, epoch, device)
        if epoch == 0 or (
                epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device)
    writer.close()
    logger.info(
        "======== Training Finished.   best_test_acc: {:.3f}% ========".format(best_prec))


if __name__ == "__main__":
    main()
