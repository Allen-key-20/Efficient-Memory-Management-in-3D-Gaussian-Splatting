import torch
import torch.nn.functional as F
import os
import csv
from utils.image_utils import psnr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
import torchvision
from PIL import Image
import time
import math
import GPUtil



class DecayScheduler(object):
    def __init__(self, total_steps, decay_name='cosine', start=0.25, end=1.0, params=None):
        self.decay_name = decay_name
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.params = params

    def __call__(self, step):

        if step > self.total_steps:
            return self.end
        return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(step / self.total_steps * math.pi))

        # if self.decay_name == 'fix':
        #     return self.start
        # elif self.decay_name == 'linear':
        #     if step>self.total_steps:
        #         return self.end
        #     return self.start + (self.end - self.start) * step / self.total_steps
        # elif self.decay_name == 'exp':
        #     if step>self.total_steps:
        #         return self.end
        #     return max(self.end, self.start*(np.exp(-np.log(1/self.params['temperature'])*step/self.total_steps/self.params['decay_period'])))
        #     # return self.start * (self.end / self.start) ** (step / self.total_steps)
        # elif self.decay_name == 'inv_sqrt':
        #     return self.start * (self.total_steps / (self.total_steps + step)) ** 0.5
        # elif self.decay_name == 'cosine':
        #     if step>self.total_steps:
        #         return self.end
        #     return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(step / self.total_steps * math.pi))
        # else:
        #     raise ValueError('Unknown decay name: {}'.format(self.decay_name))

def down_smaple_img(original_image, scale=1.0):
    assert scale <= 1.0, "Scale must be <= 1.0"
    if scale == 1.0:
        return original_image
    else:
        return torch.nn.functional.interpolate(original_image.unsqueeze(0), scale_factor=scale, mode='bilinear',
                                               align_corners=False).squeeze(0)

def down_smaple_img_2(gt_image, factor, iteration):
    if iteration < 500:
        factor = factor /2
    gt = gt_image.unsqueeze(0)
    down = F.interpolate(gt, scale_factor=factor, mode='bilinear')
    gt = F.interpolate(down, size=gt.shape[2:], mode='bilinear')
    return gt.squeeze(0)

def recording(image, gt_image, args, iteration, gaussians, init_mem_use, it=100):
    path = os.path.abspath(args.model_path)
    path = os.path.join(path, 'csv')

    if iteration % it == 0:

        gpu = GPUtil.getGPUs()[0]
        used_memory = gpu.memoryUsed - init_mem_use

        with open(rf'{path}/memory_used.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([used_memory])

    with open(rf'{path}/psnr.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        psnr_value = psnr(image, gt_image).mean().flatten()
        writer.writerow(psnr_value.tolist())

    point_num = gaussians.get_xyz.shape[0]
    with open(rf'{path}/points_num.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([point_num])

    return psnr_value.item()


def compare(model, outputdir, it=100):
    ourgs = rf"{outputdir}/{model}/csv/memory_used.csv"
    data_n = pd.read_csv(ourgs, header=None)
    data_n = np.array(data_n)
    max_n_0 = np.max(data_n[0:150])
    max_n_1 = np.max(data_n)

    with open(rf'{outputdir}/{model}/0_metric.txt', 'a') as file:
        file.write(f"\n{max_n_0}")
        file.write(f"\n{max_n_1}")
        file.flush()

