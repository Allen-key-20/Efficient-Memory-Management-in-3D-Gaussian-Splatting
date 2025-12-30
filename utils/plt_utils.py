import torch
import torch.nn.functional as F
import os
import csv
from utils.image_utils import psnr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
#import matplotlib.pyplot as plt
import subprocess
import re
import math

# ---------- 字体 ----------
times_font = fm.FontProperties(
    fname=fm.findfont(fm.FontProperties(family='Times New Roman'))
)
plt.rcParams['font.family'] = times_font.get_name()


class DecayScheduler(object):
    def __init__(self, total_steps, decay_name='cosine', start=0.25, end=1.0, params=None):
        self.decay_name = decay_name
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.params = params

    def __call__(self, step):
        if self.decay_name == 'fix':
            return self.start
        elif self.decay_name == 'linear':
            if step>self.total_steps:
                return self.end
            return self.start + (self.end - self.start) * step / self.total_steps
        elif self.decay_name == 'exp':
            if step>self.total_steps:
                return self.end
            return max(self.end, self.start*(np.exp(-np.log(1/self.params['temperature'])*step/self.total_steps/self.params['decay_period'])))
            # return self.start * (self.end / self.start) ** (step / self.total_steps)
        elif self.decay_name == 'inv_sqrt':
            return self.start * (self.total_steps / (self.total_steps + step)) ** 0.5
        elif self.decay_name == 'cosine':
            if step>self.total_steps:
                return self.end
            return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(step / self.total_steps * math.pi))
        else:
            raise ValueError('Unknown decay name: {}'.format(self.decay_name))

def down_smaple_img(original_image, scale=1.0):
    assert scale <= 1.0, "Scale must be <= 1.0"
    if scale == 1.0:
        return original_image
    else:
        return torch.nn.functional.interpolate(original_image.unsqueeze(0), scale_factor=scale, mode='bilinear',
                                               align_corners=False).squeeze(0)

def down_smaple_img2(gt_image, factor, iteration):
    if iteration < 500:
        factor = factor /2
    gt = gt_image.unsqueeze(0)
    down = F.interpolate(gt, scale_factor=factor, mode='bilinear')
    gt = F.interpolate(down, size=gt.shape[2:], mode='bilinear')
    return gt.squeeze(0)

def recording(image, gt_image, args, iteration, gaussians, init_mem_use, it=100):
    path = os.path.abspath(args.model_path)

    if iteration % it == 0:
        result = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        memory_usage_pattern = r"(\d+)MiB /"
        memory_usage = re.findall(memory_usage_pattern, result)
        used_memory = int(memory_usage[0]) - init_mem_use
        #print(used_memory)
        with open(rf'{path}/memory_used.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([used_memory])

    with open(rf'{path}/psnr.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(psnr(image, gt_image).mean().flatten().tolist())

    point_num = gaussians.get_xyz.shape[0]
    with open(rf'{path}/points_num.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([point_num])


def psnr_plt(model,outputdir,it):
    try:
        ogs = rf"{outputdir}/vanilla/{model}/psnr.csv"
        data_o = pd.read_csv(ogs, header=None)
        data_o = np.array(data_o)
        averages_o = data_o.reshape(-1, it).mean(axis=1)
        plt.plot(averages_o, label='vanilla GS')
    except:
        print("原版高斯psnr不存在")

    ngs = rf"{outputdir}/{model}/psnr.csv"
    data_n = pd.read_csv(ngs, header=None)
    data_n = np.array(data_n)
    averages_n = data_n.reshape(-1, it).mean(axis=1)
    plt.plot(averages_n, label='ours')

    plt.title(f'dataset : {model}')
    plt.xlabel(f'every {it} iterations')
    plt.ylabel('PSNR')
    plt.legend(loc='lower right')
    savepath = rf"{outputdir}/{model}/psnr_compared.jpg"
    plt.savefig(savepath)
    plt.close()

def point_num_plt(model,outputdir,it):
    try:
        ogs = rf"{outputdir}/vanilla/{model}/points_num.csv"
        data_o = pd.read_csv(ogs, header=None)
        data_o = np.array(data_o)
        averages_o = data_o.reshape(-1, it).mean(axis=1)
        plt.plot(averages_o, label='3DGS')
    except:
        print("原版高斯points_num不存在")

    ngs = rf"{outputdir}/{model}/points_num.csv"
    data_n = pd.read_csv(ngs, header=None)
    data_n = np.array(data_n)
    averages_n = data_n.reshape(-1, it).mean(axis=1)
    plt.plot(averages_n, label='Ours')

    plt.title(f'scene : {model}')
    #plt.xlabel(f'every {it} iterations')
    #plt.ylabel('points_num')
    plt.legend(loc='lower right')
    #savepath = rf"{outputdir}/{model}/point_num_compared.jpg"
    savepath = rf"{outputdir}/num_plt/{model}_point_num_compared.jpg"
    plt.savefig(savepath)
    plt.close()

def metric_compare(model,outputdir,it):
    try:
        ogs = rf"{outputdir}/vanilla/{model}/metric.txt"
        with open(ogs, 'r') as file_0:
            lines_0 = file_0.read()
    except:
        lines_0 = ""
        print("原版高斯metric不存在")

    ngs = rf"{outputdir}/{model}/metric.txt"
    with open(ngs, 'r+') as file:
        lines = file.read()
        file.seek(0)
        file.write(lines_0 + lines)

def memory_plt(model, outputdir, it):
    try:
        ogs = rf"{outputdir}/vanilla/{model}/memory_used.csv"
        data_o = pd.read_csv(ogs, header=None)
        data_o = np.array(data_o)
        max_o = np.max(data_o)
        plt.plot(data_o, label='3DGS')
    except:
        max_o = 0
        print("原版高斯memory_used不存在")

    ngs = rf"{outputdir}/{model}/memory_used.csv"
    data_n = pd.read_csv(ngs, header=None)
    data_n = np.array(data_n)
    max_n = np.max(data_n)
    plt.plot(data_n, label='ours')

    plt.title(f'scene : {model}')
    #plt.xlabel(f'every {it} iterations')
    plt.ylabel('GPU memory used / MB')
    plt.legend(loc='lower right')
    #savepath = rf"{outputdir}/{model}/memory_used_compared.jpg"
    savepath = rf"{outputdir}/mem_plt/{model}_memory_used_compared.jpg"
    plt.savefig(savepath)
    plt.close()

    with open(rf'{outputdir}/{model}/metric.txt', 'a') as file:
        #file.write(f"{max_o}\n")
        file.write(f"{max_n}\n")
        file.flush()

def plt_compare(model, outputdir, it=100):
    #psnr_plt(model, outputdir, it)
    point_num_plt(model, outputdir, it)
    #metric_compare(model, outputdir, it)
    #memory_plt(model, outputdir, it)

