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
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim
from gaussian_renderer import render, render_motion
import sys
from scene import Scene, GaussianModel, MotionNetwork
from utils.general_utils import safe_state
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.camera_utils import loadCamOnTheFly
import copy

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    testing_iterations = [i for i in range(0, opt.iterations + 1, 2000)]
    checkpoint_iterations =  saving_iterations = [i for i in range(0, opt.iterations + 1, 10000)] + [opt.iterations]

    # vars
    warm_step = 3000
    opt.densify_until_iter = opt.iterations - 1000
    bg_iter = opt.iterations # opt.densify_until_iter
    lpips_start_iter = opt.densify_until_iter - 2000
    motion_stop_iter = bg_iter
    mouth_select_iter = bg_iter - 10000
    mouth_step = 1 / mouth_select_iter
    hair_mask_interval = 7
    select_interval = 15

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    motion_net = MotionNetwork(args=dataset).cuda()
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: (0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / bg_iter))

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, motion_params, motion_optimizer_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)

    bg_color = [0, 1, 0]   # [1, 1, 1] # if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # find a big mouth
        mouth_global_lb = viewpoint_cam.talking_dict['mouth_bound'][0]
        mouth_global_ub = viewpoint_cam.talking_dict['mouth_bound'][1]
        mouth_global_lb += (mouth_global_ub - mouth_global_lb) * 0.2
        mouth_window = (mouth_global_ub - mouth_global_lb) * 0.2

        mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)
        mouth_ub = mouth_lb + mouth_window
        mouth_lb = mouth_lb - mouth_window


        au_global_lb = 0
        au_global_ub = 1
        au_window = 0.3

        au_lb = au_global_lb + mouth_step * iteration * (au_global_ub - au_global_lb)
        au_ub = au_lb + au_window
        au_lb = au_lb - au_window * 0.5


        if iteration < warm_step:
            if iteration % select_interval == 0:
                while viewpoint_cam.talking_dict['mouth_bound'][2] < mouth_lb or viewpoint_cam.talking_dict['mouth_bound'][2] > mouth_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        if warm_step < iteration < mouth_select_iter:

            if iteration % select_interval == 0:
                while viewpoint_cam.talking_dict['blink'] < au_lb or viewpoint_cam.talking_dict['blink'] > au_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if viewpoint_cam.original_image == None:
            viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        head_mask =  face_mask + hair_mask

        if iteration > lpips_start_iter:
            max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            mouth_mask = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()

        
        hair_mask_iter = (warm_step < iteration < lpips_start_iter - 1000) and iteration % hair_mask_interval != 0

        if iteration < warm_step:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        else:
            render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True)

        image_white, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image  = viewpoint_cam.original_image.cuda() / 255.0
        gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask

        if iteration > motion_stop_iter:
            for param in motion_net.parameters():
                param.requires_grad = False
        if iteration > bg_iter:
            gaussians._xyz.requires_grad = False
            gaussians._opacity.requires_grad = False
            # gaussians._features_dc.requires_grad = False
            # gaussians._features_rest.requires_grad = False
            gaussians._scaling.requires_grad = False
            gaussians._rotation.requires_grad = False
        
        # Loss
        if iteration < bg_iter:
            if hair_mask_iter:
                image_white[:, hair_mask] = background[:, None]
                gt_image_white[:, hair_mask] = background[:, None]
            
            # image_white[:, mouth_mask] = 1
            gt_image_white[:, mouth_mask] = background[:, None]

            Ll1 = l1_loss(image_white, gt_image_white)
            loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image_white, gt_image_white))
            
            # mouth_alpha_loss = 1e-2 * (alpha[:,mouth_mask]).mean()
            # if not torch.isnan(mouth_alpha_loss):
                # loss += mouth_alpha_loss
            # print(alpha[:,mouth_mask], mouth_mask.sum())

            if iteration > warm_step:
                loss += 1e-5 * (render_pkg['motion']['d_xyz'].abs()).mean()
                loss += 1e-5 * (render_pkg['motion']['d_rot'].abs()).mean()
                loss += 1e-5 * (render_pkg['motion']['d_opa'].abs()).mean()
                loss += 1e-5 * (render_pkg['motion']['d_scale'].abs()).mean()
                
                loss += 1e-3 * (((1-alpha) * head_mask).mean() + (alpha * ~head_mask).mean())


                [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
                loss += 1e-4 * (render_pkg["attn"][1, xmin:xmax, ymin:ymax]).mean()
                if not hair_mask_iter:
                    loss += 1e-4 * (render_pkg["attn"][1][hair_mask]).mean()
                    loss += 1e-4 * (render_pkg["attn"][0][hair_mask]).mean()

                # loss += l2_loss(image_white[:, xmin:xmax, ymin:ymax], image_white[:, xmin:xmax, ymin:ymax])

            image_t = image_white.clone()
            gt_image_t = gt_image_white.clone()

        else:
            # with real bg
            image = image_white - background[:, None, None] * (1.0 - alpha) + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha)

            Ll1 = l1_loss(image, gt_image)
            loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            image_t = image.clone()
            gt_image_t = gt_image.clone()

        if iteration > lpips_start_iter:   
            # mask mouth
            [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
            loss += 0.01 * lpips_criterion(image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1, gt_image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1).mean()

            image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]
            gt_image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]
            
            patch_size = random.randint(32, 48) * 2
            loss += 0.2 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()
            # loss += 0.5 * lpips_criterion(image_t[None, ...] * 2 - 1, gt_image_t[None, ...] * 2 - 1).mean()


        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "Mouth": f"{mouth_lb:.{1}f}-{mouth_ub:.{1}f}"}) # , "AU25": f"{au_lb:.{1}f}-{au_ub:.{1}f}"
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, motion_net, render if iteration < warm_step else render_motion, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(str(iteration)+'_face')

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckpt = (gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                torch.save(ckpt, scene.model_path + "/chkpnt_face_" + str(iteration) + ".pth")
                torch.save(ckpt, scene.model_path + "/chkpnt_face_latest" + ".pth")


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)

            
            # bg prune
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                from utils.sh_utils import eval_sh

                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                bg_color_mask = (colors_precomp[..., 0] < 30/255) * (colors_precomp[..., 1] > 225/255) * (colors_precomp[..., 2] < 30/255)
                gaussians.prune_points(bg_color_mask.squeeze())


            # Optimizer step
            if iteration < opt.iterations:
                motion_optimizer.step()
                gaussians.optimizer.step()

                motion_optimizer.zero_grad()
                gaussians.optimizer.zero_grad(set_to_none = True)

                scheduler.step()



def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, motion_net, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(5, 100, 5)]}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if viewpoint.original_image == None:
                        viewpoint = loadCamOnTheFly(copy.deepcopy(viewpoint))
                        
                    if renderFunc is render:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    else:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, return_attn=True, frame_idx=0, *renderArgs)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    alpha = render_pkg["alpha"]
                    image = image - renderArgs[1][:, None, None] * (1.0 - alpha) + viewpoint.background.cuda() / 255.0 * (1.0 - alpha)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0)
                    
                    mouth_mask = torch.as_tensor(viewpoint.talking_dict["mouth_mask"]).cuda()
                    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    mouth_mask_post = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), (render_pkg["depth"] / render_pkg["depth"].max())[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask_post".format(viewpoint.image_name), (~mouth_mask_post * gt_image)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask".format(viewpoint.image_name), (~mouth_mask[None] * gt_image)[None], global_step=iteration)

                        if renderFunc is not render:
                            tb_writer.add_images(config['name'] + "_view_{}/attn_a".format(viewpoint.image_name), (render_pkg["attn"][0] / render_pkg["attn"][0].max())[None, None], global_step=iteration)  
                            tb_writer.add_images(config['name'] + "_view_{}/attn_e".format(viewpoint.image_name), (render_pkg["attn"][1] / render_pkg["attn"][1].max())[None, None], global_step=iteration)  

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
