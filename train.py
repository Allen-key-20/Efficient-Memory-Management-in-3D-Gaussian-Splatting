#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from our_utils import recording, down_smaple_img, DecayScheduler
import GPUtil
from fused_ssim import fused_ssim as fast_ssim
import torchvision
import time

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    ema_PSNR_for_log = 0.0
    render_path, gts_path, tb_writer = prepare_output_and_logger(dataset, opt.iterations)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    dc_grad = None
    t = None
    render_time = 0
    back_time = 0
    resize_scale = DecayScheduler(opt.total_steps)

    viewpoint_stack = scene.getTrainCameras().copy()
    with open(rf"./{dataset.model_path}/train_len.txt", "w", encoding="UTF-8") as rec:
        rec.write(f"{len(viewpoint_stack)}")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    for iteration in range(first_iter+1, opt.iterations + 1):
        xyz_lr = gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image = viewpoint_cam.original_image.cuda()
        if opt.DaN and iteration < opt.total_steps:
            gt_image = down_smaple_img(gt_image, resize_scale(iteration))

        r1 = time.time()
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, iteration, gt_image.shape, opt)
        r2 = time.time()
        if iteration > 15000:
            render_time += r2 - r1

        image, viewspace_point_tensor, visibility_filter = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"]

        # Loss
        psnr_value = recording(image, gt_image, dataset, iteration, gaussians, init_mem_use)
        loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image) + opt.lambda_dssim * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))

        b1 = time.time()
        loss.backward()
        b2 = time.time()
        if iteration > 15000:
            back_time += b2 - b1

        if iteration < opt.densify_until_iter:
            dc_grad = gaussians._features_dc.grad.detach().squeeze()
            t = render_pkg["t"]

        with torch.no_grad():
            ema_PSNR_for_log = 0.4 * psnr_value + 0.6 * ema_PSNR_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{ema_PSNR_for_log:.{3}f}", "GS_N": f"{gaussians.size}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # if (iteration in saving_iterations):
            #     #print("ITER {} Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, dc_grad, t)

                if iteration > opt.level_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.level_up_sh(opt.sh_threshold, opt.opa_threshold)
                    gaussians.level_down_sh(opt.t_threshold)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, opt.SP)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if opt.DaN and iteration < opt.noise_until:
                gaussians.add_noise(opt.noise, xyz_lr)

            # if (iteration in checkpoint_iterations):
            #     print("ITER {} Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        torch.cuda.empty_cache()

    try:
        with open(rf"{args.model_path}/0_time.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"渲染总时间：{render_time}\n")
            rec.write(f"back总时间：{back_time}\n")
    except:
        pass

    #print(f"训练完毕，3阶有{gaussians._features_rest.shape[0]}个，总共{gaussians._xyz.shape[0]}个")
    with open(rf"{args.model_path}/0_metric.txt", "a", encoding="UTF-8") as rec:
        rec.write(f"{gaussians._xyz.shape[0]}\n")
        rec.write(f"{gaussians._features_rest.shape[0]}\n")

    t_test_list = []
    viewpoint_stack_test = list(scene.getTestCameras().copy())
    for idx, viewpoint_cam in enumerate(tqdm(viewpoint_stack_test, desc="Rendering progress")):
        torch.cuda.synchronize()
        t_start = time.time()
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        torch.cuda.synchronize()
        t_end = time.time()
        t_test_list.append(t_end - t_start)
        image = render_pkg["render"]
        torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        gt_image = viewpoint_cam.original_image.cuda()
        torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    test_fps_all = 1.0 / torch.tensor(t_test_list[:]).mean()
    test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()

    with open(rf"./{dataset.model_path}/0_metric.txt", "a", encoding="UTF-8") as rec:
        rec.write(f"{test_fps_all}\n")
        rec.write(f"{test_fps}\n")

def prepare_output_and_logger(args, iterations):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    # print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    render_path = os.path.join(args.model_path, 'test', "ours_{}".format(iterations), "renders")
    gts_path = os.path.join(args.model_path, 'test', "ours_{}".format(iterations), "gt")
    csv_path = os.path.join(args.model_path, 'csv')
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    file_path = os.path.join(csv_path, 'psnr.csv')  # 自动处理 Win/Linux 分隔符
    if os.path.isfile(file_path):  # 存在且是文件才删
        os.remove(file_path)
    file_path = os.path.join(csv_path, 'points_num.csv')  # 自动处理 Win/Linux 分隔符
    if os.path.isfile(file_path):  # 存在且是文件才删
        os.remove(file_path)
    file_path = os.path.join(csv_path, 'memory_used.csv')  # 自动处理 Win/Linux 分隔符
    if os.path.isfile(file_path):  # 存在且是文件才删
        os.remove(file_path)

    return render_path, gts_path, None

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("ITER {} Evaluate {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    gpu = GPUtil.getGPUs()[0]
    init_mem_use = gpu.memoryUsed
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from
    )

    # All done
    #print("Training complete.")
