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

from scene.cameras import Camera
import torch
import numpy as np
import os
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    
    if cam_info.image is not None:
        image_rgb = PILtoTorch(cam_info.image).type("torch.ByteTensor")
        gt_image = image_rgb[:3, ...]
    else:
        gt_image = None
        
    if cam_info.background is not None:
        background = PILtoTorch(cam_info.background)[:3, ...].type("torch.ByteTensor")
    else:
        background = None

    loaded_mask = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask, background=background, talking_dict=cam_info.talking_dict,
                  image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def loadCamOnTheFly(camera):
    image_path = camera.image_path
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    bg_img = PILtoTorch(np.array(Image.open(os.path.join("/".join(image_path.split("/")[:-2]), 'bc.jpg')).convert("RGB"))).to(camera.data_device)
    torso_img_path = image_path.replace("gt_imgs", "torso_imgs").replace("jpg", "png")
    torso_img = PILtoTorch(np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0).to(camera.data_device)
    bg = torso_img[:3] * torso_img[3:] / 255 + bg_img * (1.0 - torso_img[3:] / 255)

    teeth_mask_path = image_path.replace("gt_imgs", "teeth_mask").replace("jpg", "npy")
    teeth_mask = torch.as_tensor(np.load(teeth_mask_path)).to(camera.data_device)

    mask_path = image_path.replace("gt_imgs", "parsing").replace("jpg", "png")
    mask = PILtoTorch(np.array(Image.open(mask_path).convert("RGB")) * 1.0).to(camera.data_device)
    camera.talking_dict['face_mask'] = (mask[2] > 254) * (mask[0] == 0) * (mask[1] == 0) ^ teeth_mask
    camera.talking_dict['hair_mask'] = (mask[0] < 1) * (mask[1] < 1) * (mask[2] < 1)
    camera.talking_dict['mouth_mask'] = (mask[0] == 100) * (mask[1] == 100) * (mask[2] == 100) + teeth_mask
    
    camera.original_image = PILtoTorch(image).type("torch.ByteTensor").clamp(0, 255).to(camera.data_device)
    camera.background = bg.type("torch.ByteTensor").clamp(0, 255).to(camera.data_device)
    camera.image_width = camera.original_image.shape[2]
    camera.image_height = camera.original_image.shape[1]
    
    return camera
