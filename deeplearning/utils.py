import os
import time

import torch
import torch.nn as nn

from deeplearning.nets.initialize import nn_registry
from deeplearning.trainer import UnetRegressorTrainer
from deeplearning.trainer.measure import AverageMeter

def init_all(config, dataset, logger):
    network = nn_registry[config.model](dataset.channel, 
                    num_classes=dataset.num_classes, 
                    im_size=dataset.im_size, 
                    pretrained=config.pretrained, 
                    config=config)
    network = network.to(config.device)
    network.train()

    # criterion = nn.CrossEntropyLoss().to(config.device)
    criterion = nn.MSELoss().to(config.device)

    # Optimizer
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=config.nesterov)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(network.parameters(), config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.__dict__[config.optimizer](network.parameters(), config.lr, momentum=config.momentum,
                                                            weight_decay=config.weight_decay, nesterov=config.nesterov)

    # LR scheduler
    if config.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs,
                                                                eta_min=config.min_lr)
    elif config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size,
                                                    gamma=config.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.__dict__[config.scheduler](optimizer)

    # load checkpoint if the path exists
    if os.path.exists(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path)
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        start_epoch = checkpoint['epoch']
        logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(config.checkpoint_path, checkpoint['epoch']))
    else:
        start_epoch = 0

    return network, criterion, optimizer, scheduler, start_epoch


def init_optimizer( config, network, N):
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), 0.1*config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=config.nesterov)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.__dict__[config.optimizer](network.parameters(), config.lr, momentum=config.momentum,
                                                            weight_decay=config.weight_decay, nesterov=config.nesterov)

    if config.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N * config.epochs,
                                                                eta_min=config.min_lr)
    elif config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=N * config.step_size,
                                                    gamma=config.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.__dict__[config.scheduler](optimizer)

    return optimizer, scheduler

def save_checkpoint(state, path):
    torch.save(state, path)


def train(train_loader, network, optimizer, scheduler, epoch, config, logger, record):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # switch to train mode
    network.train()
    trainer = UnetRegressorTrainer(config, network)

    end = time.time()
    for i, (contents, _) in enumerate(train_loader):
        # if i != len(train_loader)-1: continue
        if config.debug:
            if i != len(train_loader)-1: continue

        optimizer.zero_grad()

        contents = contents.to(config.device)
        loss = trainer(contents)
        losses.update(loss.data.item(), contents.shape[0])

        # Compute gradient and do SGD step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            network.parameters(), config.grad_clip)

        optimizer.step()
        scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_every == 0 or i == len(train_loader)-1:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses))
            
            record["train_loss"].append(losses.avg)





