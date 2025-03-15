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

import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_motion, render_motion_mouth
import torchvision
from utils.general_utils import safe_state
from utils.camera_utils import loadCamOnTheFly
import copy
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, MotionNetwork, MouthMotionNetwork

import torch.nn.functional as F

def dilate_fn(bin_img, ksize=13):
    pad = (ksize - 1) // 2
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=pad)
    return out

def render_set(model_path, name, iteration, views, gaussians, motion_net, gaussians_mouth, motion_net_mouth, pipeline, background, fast, dilate):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    all_preds = []
    all_gts = []

    all_preds_face = []
    all_preds_mouth = []


    for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True)):
        if view.original_image == None:
            view = loadCamOnTheFly(copy.deepcopy(view))
        with torch.no_grad():
            render_pkg = render_motion(view, gaussians, motion_net, pipeline, background, frame_idx=0)
            render_pkg_mouth = render_motion_mouth(view, gaussians_mouth, motion_net_mouth, pipeline, background, frame_idx=0)
        # gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if dilate:
            alpha_mouth = dilate_fn(render_pkg_mouth["alpha"][None])[0]
        else:
            alpha_mouth = render_pkg_mouth["alpha"]
            
        mouth_image = render_pkg_mouth["render"] + view.background.cuda() / 255.0 * (1.0 - alpha_mouth)

        # alpha = gaussian_blur(render_pkg["alpha"], [3, 3], 2)
        alpha = render_pkg["alpha"]
        image = render_pkg["render"] + mouth_image * (1.0 - alpha)

        pred = (image[0:3, ...].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()* 255).astype(np.uint8)
        all_preds.append(pred)
        
        if not fast:
            all_preds_face.append((render_pkg["render"].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()* 255).astype(np.uint8))
            all_preds_mouth.append((render_pkg_mouth["render"].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()* 255).astype(np.uint8))

            all_gts.append(view.original_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    
    imageio.mimwrite(os.path.join(render_path, 'out.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
    if not fast:
        imageio.mimwrite(os.path.join(gts_path, 'out.mp4'), all_gts, fps=25, quality=8, macro_block_size=1)

        imageio.mimwrite(os.path.join(render_path, 'out_face.mp4'), all_preds_face, fps=25, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(render_path, 'out_mouth.mp4'), all_preds_mouth, fps=25, quality=8, macro_block_size=1)



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, use_train : bool, fast, dilate):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians_mouth = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        motion_net = MotionNetwork(args=dataset).cuda()
        motion_net_mouth = MouthMotionNetwork(args=dataset).cuda()

        (model_params, motion_params, model_mouth_params, motion_mouth_params) = torch.load(os.path.join(dataset.model_path, "chkpnt_fuse_latest.pth"))
        motion_net.load_state_dict(motion_params, strict=False)
        gaussians.restore(model_params, None)

        motion_net_mouth.load_state_dict(motion_mouth_params, strict=False)
        gaussians_mouth.restore(model_mouth_params, None)

        
        # motion_net.fix(gaussians.get_xyz.cuda())
        # motion_net_mouth.fix(gaussians_mouth.get_xyz.cuda())

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        render_set(dataset.model_path, "test" if not use_train else "train", scene.loaded_iter, scene.getTestCameras() if not use_train else scene.getTrainCameras(), gaussians, motion_net, gaussians_mouth, motion_net_mouth, pipeline, background, fast, dilate)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--use_train", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dilate", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.use_train, args.fast, args.dilate)
