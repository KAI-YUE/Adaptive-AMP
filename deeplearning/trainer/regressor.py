import os
import cv2
import time
import datetime

import numpy as np
from scipy.stats import invgamma

import torch
import torch.nn as nn

from deeplearning.trainer.measure import AverageMeter

class UnetRegressorTrainer(nn.Module):
    def __init__(self, config, model):
        super().__init__()

        self.model = model
        self.upperbound = torch.tensor(config.upperbound, dtype=torch.int, device=config.device)
        self.config = config

        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        Algorithm of training adaptive regressor/denoiser
        """

        # prior distribution for sigma^2
        # 1. inverse gamma distribution InvGamma(shape, scale)
        shape, scale = self.config.shape, self.config.scale
        num_samples = x.shape[0]

        # Generating sample data
        s2_numpy = invgamma.rvs(shape, scale=scale, size=num_samples)
        s2 = torch.from_numpy(s2_numpy).to(x)
        s_ = torch.zeros_like(s2).to(torch.int)

        noisy_data = torch.clone(x)
        noise = torch.zeros_like(x)
        for i in range(num_samples):
            s = torch.sqrt(s2[i])
            noise[i] = torch.randn_like(x[i]) * s / self.config.denominator
            noisy_data[i] += noise[i]
            index_ = torch.round(s)
            index_ = index_.to(torch.int) if index_ < self.upperbound else self.upperbound - 1
            s_[i] = index_

        # print(s_)

        # denoised = self.model(noisy_data, s_)
        # loss = self.criterion(denoised, x)
        
        pred_noise = self.model(noisy_data, s_)
        loss = self.criterion(pred_noise, noise)

        return loss
    

def test_regressor(test_loader, network, epoch, config, logger, record):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    criterion = nn.MSELoss()

    # prior distribution for sigma^2
    # 1. inverse gamma distribution InvGamma(shape, scale)
    shape, scale = config.shape, config.scale

    end = time.time()
    for i, (x, _) in enumerate(test_loader):
        x = x.to(config.device)
        num_samples = x.shape[0]

        # Generating sample data
        s2_numpy = invgamma.rvs(shape, scale=scale, size=num_samples)
        s2 = torch.from_numpy(s2_numpy).to(x)
        s_ = torch.zeros_like(s2).to(torch.int)

        noisy_data = torch.clone(x)
        noise = torch.zeros_like(x)
        for j in range(num_samples):
            s = torch.sqrt(s2[j]) 
            s = torch.round(s).to(torch.int).to(x.device) if s < config.upperbound else config.upperbound - 1
            noise[j] = torch.randn_like(x[j]) * s / config.denominator
            noisy_data[j] += noise[j]
            s_[j] = s

        # Compute output
        with torch.no_grad():
            output = network(noisy_data, s_)
            # loss = criterion(output, x).mean()
            loss = criterion(output, noise)

        # Measure accuracy and record loss
        losses.update(loss.data.item(), x.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.visualize_every == 0:
            # visualize?
            sample_save(config, x, noisy_data, noisy_data - output, epoch)

    logger.info('Test error: * Err :{:.3f}'.format(losses.avg))
    record["test_loss"].append(losses.avg)

    network.no_grad = False
    network.train()

    return losses.avg


def sample_save(config, image, image_corrupt, nn_output, epoch, verbose=True):
    """
    Save a batch to image with format [groud_truth, mask, output]
    """
    if (verbose):
        print("saving a batch of sample for validation in epoch{}".format(epoch))
    
    current_time = datetime.datetime.now()
    current_time_str = datetime.datetime.strftime(current_time ,'%H_%M_%S')
    
    if not os.path.exists(os.path.join(config.output_dir, str(epoch))):
        os.mkdir(os.path.join(config.output_dir, str(epoch)))

    if not os.path.exists(os.path.join(config.output_dir, str(epoch), current_time_str)):
        os.mkdir(os.path.join(config.output_dir, str(epoch), current_time_str))
    
    for i in range(image.shape[0]):
        img = post_process(config, image[i,...])
        img_corrupt = post_process(config, image_corrupt[i,...])
        output = post_process(config, nn_output[i,...])
        img_combine = np.hstack((img, img_corrupt, output))   
        cv2.imwrite(os.path.join(config.output_dir, str(epoch), current_time_str, 'sample_{:d}'.format(i) + '.jpg'), img_combine[:,:,::-1])


def post_process(config, img, inverseTrans=False):
    ch, h, w = img.shape
    
    if (inverseTrans):
        mean, std = config.data_mean, config.data_std
        if ch == 1:
            img = img*config.data_std + config.data_mean
        else:
            mean, std = torch.tensor(mean).to(img), torch.tensor(std).to(img)
            img = img*std[:,None,None] + mean[:,None,None]

    img *= 255
    torch.clamp(img, 0, 255)
    
    img = img.to(torch.uint8)
    if (ch == 1):
        img = img.expand(3, h, w)
        
    img = img.cpu().numpy()
    img = img.transpose([1, 2, 0])

    return img