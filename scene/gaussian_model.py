#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussian_hierarchy._C import load_hierarchy, write_hierarchy
from scene.OurAdam import Adam

import faiss
import time

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize




    def __init__(self, sh_degree : int, gt_point_cloud_path : str = None, constraint_treshold : float = 0.05):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_depth = torch.empty(0) # PACOMMENT: Added this to store the depth gradient accumulation
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.nodes = None
        self.boxes = None

        self.pretrained_exposures = None

        self.skybox_points = 0
        self.skybox_locked = True

        self.gt_point_cloud = None # PACOMMENT: Added this to store the ground truth point cloud
        self.gt_x_min = None
        self.gt_x_max = None
        self.gt_y_min = None
        self.gt_y_max = None
        if gt_point_cloud_path and os.path.exists(gt_point_cloud_path):
            self.load_gt_point_cloud(gt_point_cloud_path)

        self.constraint_treshold = constraint_treshold

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_depth, # PACOMMENT: Added this to store the depth gradient accumulation
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        xyz_gradient_accum_depth, # PACOMMENT: Added this to store the depth gradient accumulation
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_depth = xyz_gradient_accum_depth # PACOMMENT: Added this to store the depth gradient accumulation
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        return self._exposure[self.exposure_mapping[image_name]]
        # return self._exposure

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def create_from_pcd(
            self, 
            pcd : BasicPointCloud, 
            cam_infos : int,
            spatial_lr_scale : float,
            skybox_points: int,
            scaffold_file: str,
            bounds_file: str,
            skybox_locked: bool):
        
        self.spatial_lr_scale = spatial_lr_scale

        xyz = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        
        minimum,_ = torch.min(xyz, axis=0)
        maximum,_ = torch.max(xyz, axis=0)
        mean = 0.5 * (minimum + maximum)

        self.skybox_locked = skybox_locked
        if scaffold_file != "" and skybox_points > 0:
            print(f"Overriding skybox_points: loading skybox from scaffold_file: {scaffold_file}")
            skybox_points = 0
        if skybox_points > 0:
            self.skybox_points = skybox_points
            radius = torch.linalg.norm(maximum - mean)

            theta = (2.0 * torch.pi * torch.rand(skybox_points, device="cuda")).float()
            phi = (torch.arccos(1.0 - 1.4 * torch.rand(skybox_points, device="cuda"))).float()
            skybox_xyz = torch.zeros((skybox_points, 3))
            skybox_xyz[:, 0] = radius * 10 * torch.cos(theta)*torch.sin(phi)
            skybox_xyz[:, 1] = radius * 10 * torch.sin(theta)*torch.sin(phi)
            skybox_xyz[:, 2] = radius * 10 * torch.cos(phi)
            skybox_xyz += mean.cpu()
            xyz = torch.concat((skybox_xyz.cuda(), xyz))
            fused_color = torch.concat((torch.ones((skybox_points, 3)).cuda(), fused_color))
            fused_color[:skybox_points,0] *= 0.7
            fused_color[:skybox_points,1] *= 0.8
            fused_color[:skybox_points,2] *= 0.95

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = RGB2SH(fused_color)
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        if scaffold_file == "" and skybox_points > 0:
            dist2[:skybox_points] *= 10
            dist2[skybox_points:] = torch.clamp_max(dist2[skybox_points:], 10) 
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        if scaffold_file == "" and skybox_points > 0:
            opacities = self.inverse_opacity_activation(0.02 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
            opacities[:skybox_points] = 0.7
        else: 
            opacities = self.inverse_opacity_activation(0.01 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))

        features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        features_rest = features[:,:,1:].transpose(1, 2).contiguous()

        self.scaffold_points = None
        if scaffold_file != "": 
            scaffold_xyz, features_dc_scaffold, features_extra_scaffold, opacities_scaffold, scales_scaffold, rots_scaffold = self.load_ply_file(scaffold_file + "/point_cloud.ply", 1)
            scaffold_xyz = torch.from_numpy(scaffold_xyz).float()
            features_dc_scaffold = torch.from_numpy(features_dc_scaffold).permute(0, 2, 1).float()
            features_extra_scaffold = torch.from_numpy(features_extra_scaffold).permute(0, 2, 1).float()
            opacities_scaffold = torch.from_numpy(opacities_scaffold).float()
            scales_scaffold = torch.from_numpy(scales_scaffold).float()
            rots_scaffold = torch.from_numpy(rots_scaffold).float()

            with open(scaffold_file + "/pc_info.txt") as f:
                skybox_points = int(f.readline())

            self.skybox_points = skybox_points
            with open(os.path.join(bounds_file, "center.txt")) as centerfile:
                with open(os.path.join(bounds_file, "extent.txt")) as extentfile:
                    centerline = centerfile.readline()
                    extentline = extentfile.readline()

                    c = centerline.split(' ')
                    e = extentline.split(' ')
                    center = torch.Tensor([float(c[0]), float(c[1]), float(c[2])]).cuda()
                    extent = torch.Tensor([float(e[0]), float(e[1]), float(e[2])]).cuda()

            distances1 = torch.abs(scaffold_xyz.cuda() - center)
            selec = torch.logical_and(
                torch.max(distances1[:,0], distances1[:,1]) > 0.5 * extent[0],
                torch.max(distances1[:,0], distances1[:,1]) < 1.5 * extent[0])
            selec[:skybox_points] = True

            self.scaffold_points = selec.nonzero().size(0)

            xyz = torch.concat((scaffold_xyz.cuda()[selec], xyz))
            features_dc = torch.concat((features_dc_scaffold.cuda()[selec,0:1,:], features_dc))

            filler = torch.zeros((features_extra_scaffold.cuda()[selec,:,:].size(0), 15, 3))
            filler[:,0:3,:] = features_extra_scaffold.cuda()[selec,:,:]
            features_rest = torch.concat((filler.cuda(), features_rest))
            scales = torch.concat((scales_scaffold.cuda()[selec], scales))
            rots = torch.concat((rots_scaffold.cuda()[selec], rots))
            opacities = torch.concat((opacities_scaffold.cuda()[selec], opacities))

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}

        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        print("Number of points at initialisation : ", self._xyz.shape[0])

    def training_setup(self, training_args, our_adam=True):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_depth = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # PACOMMENT: Added this to store the depth gradient accumulation
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        if our_adam:
            self.optimizer = Adam(l, lr=0.0, eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if self.pretrained_exposures is None:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final, lr_delay_steps=training_args.exposure_lr_delay_steps, lr_delay_mult=training_args.exposure_lr_delay_mult, max_steps=training_args.iterations)

       
    def load_ply_file(self, path, degree):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return xyz, features_dc, features_extra, opacities, scales, rots


    def create_from_hier(self, path, spatial_lr_scale : float, scaffold_file : str):
        self.spatial_lr_scale = spatial_lr_scale

        xyz, shs_all, alpha, scales, rots, nodes, boxes = load_hierarchy(path)

        base = os.path.dirname(path)

        try:
            with open(os.path.join(base, "anchors.bin"), mode='rb') as f:
                bytes = f.read()
                int_val = int.from_bytes(bytes[:4], "little", signed="False")
                dt = np.dtype(np.int32)
                vals = np.frombuffer(bytes[4:], dtype=dt) 
                self.anchors = torch.from_numpy(vals).long().cuda()
        except:
            print("WARNING: NO ANCHORS FOUND")
            self.anchors = torch.Tensor([]).long()

        #retrieve exposure
        exposure_file = os.path.join(base, "exposure.json")
        if os.path.exists(exposure_file):
            with open(exposure_file, "r") as f:
                exposures = json.load(f)

            self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
        else:
            print(f"No exposure to be loaded at {exposure_file}")
            self.pretrained_exposures = None

        #retrieve skybox
        self.skybox_points = 0         
        if scaffold_file != "":
            scaffold_xyz, features_dc_scaffold, features_extra_scaffold, opacities_scaffold, scales_scaffold, rots_scaffold = self.load_ply_file(scaffold_file + "/point_cloud.ply", 1)
            scaffold_xyz = torch.from_numpy(scaffold_xyz).float()
            features_dc_scaffold = torch.from_numpy(features_dc_scaffold).permute(0, 2, 1).float()
            features_extra_scaffold = torch.from_numpy(features_extra_scaffold).permute(0, 2, 1).float()
            opacities_scaffold = torch.from_numpy(opacities_scaffold).float()
            scales_scaffold = torch.from_numpy(scales_scaffold).float()
            rots_scaffold = torch.from_numpy(rots_scaffold).float()

            with open(scaffold_file + "/pc_info.txt") as f:
                    skybox_points = int(f.readline())

            self.skybox_points = skybox_points

        if self.skybox_points > 0:
            if scaffold_file != "":
                skybox_xyz, features_dc_sky, features_rest_sky, opacities_sky, scales_sky, rots_sky = scaffold_xyz[:skybox_points], features_dc_scaffold[:skybox_points], features_extra_scaffold[:skybox_points], opacities_scaffold[:skybox_points], scales_scaffold[:skybox_points], rots_scaffold[:skybox_points]

            opacities_sky = torch.sigmoid(opacities_sky)
            xyz = torch.cat((xyz, skybox_xyz))
            alpha = torch.cat((alpha, opacities_sky))
            scales = torch.cat((scales, scales_sky))
            rots = torch.cat((rots, rots_sky))
            filler = torch.zeros(features_dc_sky.size(0), 16, 3)
            filler[:, :1, :] = features_dc_sky
            filler[:, 1:4, :] = features_rest_sky
            shs_all = torch.cat((shs_all, filler))

        self._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(shs_all.cuda()[:,:1,:].requires_grad_(True))
        self._features_rest = nn.Parameter(shs_all.cuda()[:,1:16,:].requires_grad_(True))
        self._opacity = nn.Parameter(alpha.cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = torch.abs

        self.hierarchy_path = path

        self.nodes = nodes.cuda()
        self.boxes = boxes.cuda()

    def create_from_pt(self, path, spatial_lr_scale : float ):
        self.spatial_lr_scale = spatial_lr_scale

        xyz = torch.load(path + "/done_xyz.pt")
        shs_dc = torch.load(path + "/done_dc.pt")
        shs_rest = torch.load(path + "/done_rest.pt")
        alpha = torch.load(path + "/done_opacity.pt")
        scales = torch.load(path + "/done_scaling.pt")
        rots = torch.load(path + "/done_rotation.pt")

        self._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(shs_dc.cuda().requires_grad_(True))
        self._features_rest = nn.Parameter(shs_rest.cuda().requires_grad_(True))
        self._opacity = nn.Parameter(alpha.cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_hier(self):
        write_hierarchy(self.hierarchy_path + "_opt",
                        self._xyz,
                        torch.cat((self._features_dc, self._features_rest), 1),
                        self.opacity_activation(self._opacity),
                        self._scaling,
                        self._rotation,
                        self.nodes,
                        self.boxes)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''

        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_pt(self, path):
        mkdir_p(path)

        torch.save(self._xyz.detach().cpu(), os.path.join(path, "done_xyz.pt"))
        torch.save(self._features_dc.cpu(), os.path.join(path, "done_dc.pt"))
        torch.save(self._features_rest.cpu(), os.path.join(path, "done_rest.pt"))
        torch.save(self._opacity.cpu(), os.path.join(path, "done_opacity.pt"))
        torch.save(self._scaling, os.path.join(path, "done_scaling.pt"))
        torch.save(self._rotation, os.path.join(path, "done_rotation.pt"))

        import struct
        def load_pt(path):
            xyz = torch.load(os.path.join(path, "done_xyz.pt")).detach().cpu()
            features_dc = torch.load(os.path.join(path, "done_dc.pt")).detach().cpu()
            features_rest = torch.load( os.path.join(path, "done_rest.pt")).detach().cpu()
            opacity = torch.load(os.path.join(path, "done_opacity.pt")).detach().cpu()
            scaling = torch.load(os.path.join(path, "done_scaling.pt")).detach().cpu()
            rotation = torch.load(os.path.join(path, "done_rotation.pt")).detach().cpu()

            return xyz, features_dc, features_rest, opacity, scaling, rotation


        xyz, features_dc, features_rest, opacity, scaling, rotation = load_pt(path)

        my_int = xyz.size(0)
        with open(os.path.join(path, "point_cloud.bin"), 'wb') as f:
            f.write(struct.pack('i', my_int))
            f.write(xyz.numpy().tobytes())
            print(features_dc[0])
            print(features_rest[0])
            f.write(torch.cat((features_dc, features_rest), dim=1).numpy().tobytes())
            f.write(opacity.numpy().tobytes())
            f.write(scaling.numpy().tobytes())
            f.write(rotation.numpy().tobytes())


    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = torch.cat((self._opacity[:self.skybox_points], inverse_sigmoid(torch.min(self.get_opacity[self.skybox_points:], torch.ones_like(self.get_opacity[self.skybox_points:])*0.01))), 0)
        #opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        xyz, features_dc, features_extra, opacities, scales, rots = self.load_ply_file(path, self.max_sh_degree)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
            
            # -------- ADD THIS FOR MASK UPDATE --------
            if hasattr(self, '_is_new_in_current_step_mask') and self._is_new_in_current_step_mask is not None:
                 if group["name"] == "xyz": # Or any other key attribute
                    if self._is_new_in_current_step_mask.shape[0] != mask.shape[0] and self._is_new_in_current_step_mask.shape[0] != torch.sum(mask):
                        # This check might be tricky due to the mask being for selection, not original size.
                        # The crucial part is that self._is_new_in_current_step_mask should be indexed by 'mask'
                        # if 'mask' applies to its current state.
                        # Let's assume self._is_new_in_current_step_mask is already the size of group["params"][0] *before* pruning here.
                        pass # If there's a size mismatch prior to this, it's an issue from previous step
                    
                    self._is_new_in_current_step_mask = self._is_new_in_current_step_mask[mask]
            # -------- END ADDITION --------

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_depth = self.xyz_gradient_accum_depth[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            # -------- ADD THIS FOR MASK UPDATE --------
            # Store original param before modification for mask concatenation logic
            original_param_for_mask_update = group["params"][0]
            # -------- END ADDITION --------

            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
            
            # -------- ADD THIS FOR MASK UPDATE --------
            if hasattr(self, '_is_new_in_current_step_mask') and self._is_new_in_current_step_mask is not None:
                if group["name"] == "xyz": # Or any other key attribute that dictates the number of Gaussians
                    # Ensure the existing mask matches the state *before* concatenation for this group
                    if self._is_new_in_current_step_mask.shape[0] == original_param_for_mask_update.shape[0]:
                        new_gaussians_marker = torch.ones(extension_tensor.shape[0], dtype=torch.bool, device=self._xyz.device)
                        self._is_new_in_current_step_mask = torch.cat((self._is_new_in_current_step_mask, new_gaussians_marker))
                    # else:
                        # This case implies a mismatch, which shouldn't happen if initialized correctly.
                        # print("Warning: _is_new_in_current_step_mask size mismatch during cat_tensors_to_optimizer")
            # -------- END ADDITION --------

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_depth = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros((new_xyz.shape[0]), device="cuda")))
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition (not is_depth_only)
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad * self.max_radii2D * torch.pow(self.get_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        # No densification of the scaffold
        if self.scaffold_points is not None:
            selected_pts_mask[:self.scaffold_points] = False

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # Store the count of new points before adding them
        num_new_points = N * selected_pts_mask.sum()

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

        # Create a mask for newly added points
        self.newly_split_points_mask = torch.zeros(self.get_xyz.shape[0], dtype=bool, device="cuda")
        if num_new_points > 0:
            self.newly_split_points_mask[-num_new_points:] = True

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition (not is_depth_only)
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) * self.max_radii2D * torch.pow(self.get_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        # No densification of the scaffold
        if self.scaffold_points is not None:
            selected_pts_mask[:self.scaffold_points] = False
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, gt_point_cloud_constraints):
        grads = self.xyz_gradient_accum 
        grads[grads.isnan()] = 0.0

        # -------- ADD THIS: Initialize the tracking mask --------
        if self.get_xyz is not None and self.get_xyz.shape[0] > 0 :
            self._is_new_in_current_step_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
        else: # Handle case of no points initially
            self._is_new_in_current_step_mask = torch.empty(0, dtype=torch.bool, device="cuda" if torch.cuda.is_available() else "cpu") # Adjust device as needed
        # -------- END ADDITION --------

        # Mark that we don't have any newly split points yet
        self.newly_split_points_mask = torch.zeros(self.get_xyz.shape[0], dtype=bool, device="cuda")

        # grads_depth = self.xyz_gradient_accum_depth
        # grads_depth[grads_depth.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if self.scaffold_points is not None:
            prune_mask[:self.scaffold_points] = False
        
        if gt_point_cloud_constraints:
            prune_mask = self.compare_points_to_gt(prune_mask, self.newly_split_points_mask)
        
        # -------- ADD THIS PRINT --------
        # Ensure mask sizes match before logical_and. This should be true if updates were correct.
        if self._is_new_in_current_step_mask.shape[0] == prune_mask.shape[0] and self._is_new_in_current_step_mask.shape[0] > 0:
            newly_created_and_now_pruned_count = (prune_mask & self._is_new_in_current_step_mask).sum().item()
            if newly_created_and_now_pruned_count > 0: # Optional
                print(f"Number of newly created Gaussians (clone/split) that will be pruned by main criteria: {newly_created_and_now_pruned_count}")
        # elif self._is_new_in_current_step_mask.shape[0] != prune_mask.shape[0] and self.get_xyz.shape[0] > 0:
            # print(f"Warning: Mismatch between _is_new_in_current_step_mask ({self._is_new_in_current_step_mask.shape[0]}) and prune_mask ({prune_mask.shape[0]})")
        # -------- END ADDITION --------

        self.prune_points(prune_mask)

        # -------- ADD THIS: Clean up the temporary mask --------
        del self._is_new_in_current_step_mask 
        # -------- END ADDITION --------

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, is_depth_only):
        norm_value = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        max_value = torch.max(norm_value, self.xyz_gradient_accum[update_filter])
        
        # If is_depth_only is True and the max is the norm of viewspace_point_tensor
        # (i.e., norm_value > self.xyz_gradient_accum[update_filter]), multiply by 1.5
        if is_depth_only:
            # Create a mask where norm_value is greater than or equal to the accumulated value
            mask = norm_value >= self.xyz_gradient_accum[update_filter]
            # Where the mask is True, multiply max_value by 1.5
            max_value = torch.where(mask, max_value * 1, max_value)
        
        self.xyz_gradient_accum[update_filter] = max_value
        self.denom[update_filter] += 1


    def load_gt_point_cloud(self, gt_point_cloud_path):
        """
        Loads the ground truth point cloud from a .ply file
        and creates a FAISS index directly on the GPU for fast nearest neighbor search.
        Falls back to CPU if GPU is not available.
        """
        print(f"Loading ground truth point cloud from {gt_point_cloud_path}")
        plydata = PlyData.read(gt_point_cloud_path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        xyz = xyz.astype(np.float32)

        self.gt_x_min = np.min(xyz[:, 0])
        self.gt_x_max = np.max(xyz[:, 0])
        self.gt_y_min = np.min(xyz[:, 1])
        self.gt_y_max = np.max(xyz[:, 1])
        
        print(f"Loaded {xyz.shape[0]} points from ground truth")
        print(f"X range: {self.gt_x_min} to {self.gt_x_max}")
        print(f"Y range: {self.gt_y_min} to {self.gt_y_max}")
        
        dimension = 3  # 3D points

        print("Ground Truth Sample:", xyz[:10])
        print("Comparing Sample:", self.get_xyz.detach().cpu().numpy()[:10])
        
        try:
            # Check if FAISS was built with GPU support
            assert hasattr(faiss, 'StandardGpuResources')
            
            # Create GPU resources
            res = faiss.StandardGpuResources()
            
            # Configure GPU index
            config = faiss.GpuIndexFlatConfig()
            config.useFloat16 = False  # Use full precision for better accuracy
            config.device = 0  # Use first GPU
            
            # Create the index directly on GPU
            self.gt_point_cloud = faiss.GpuIndexFlatL2(res, dimension, config)
            
            # Add vectors to the index
            self.gt_point_cloud.add(xyz)
            print("FAISS index created directly on GPU successfully")
        except Exception as e:
            # Fallback to CPU if GPU fails
            print(f"Could not use GPU for FAISS: {e}. Creating CPU index instead.")
            
            # Create CPU index
            cpu_index = faiss.IndexFlatL2(dimension)
            cpu_index.add(xyz)
            self.gt_point_cloud = cpu_index
        
        print(f"FAISS index created with {xyz.shape[0]} points")

    def compare_points_to_gt(self, existing_mask=None, protected_mask=None):
        """
        Generate a mask for points that are further than the threshold distance from the ground truth point cloud.
        Only compares points that are not already in the existing mask and are inside the ground truth bounds.
        
        Args:
            existing_mask: Optional existing mask to combine with the distance-based mask
                
        Returns:
            prune_mask: Boolean mask where True indicates points that should be pruned
                    (further than threshold from ground truth AND not in scaffold)
        """

        if self.gt_point_cloud is None:
            print("No ground truth point cloud available, skipping distance-based pruning")
            return existing_mask if existing_mask is not None else torch.zeros(self.get_xyz.shape[0], dtype=bool, device="cuda")
        
        start_time = time.time()
        
        distance_threshold = self.constraint_treshold  # Constraint threshold
        
        # Get current points
        xyz = self.get_xyz.detach()

        # Create a mask for points we need to check
        # (points not in existing mask AND not in protected mask AND within GT bounds)
        points_to_check_mask = torch.ones(xyz.shape[0], dtype=bool, device=xyz.device)

        # densified_to_check = torch.zeros(xyz.shape[0], dtype=bool, device=xyz.device)

        # densified_to_check = torch.logical_or(densified_to_check, protected_mask)
        
        # Don't check points that are already marked for pruning
        if existing_mask is not None:
            points_to_check_mask = torch.logical_and(points_to_check_mask, ~existing_mask)
        
        # Don't check protected points (like newly split points)
        if protected_mask is not None:
            points_to_check_mask = torch.logical_and(points_to_check_mask, ~protected_mask)
        
        # Also limit to points within GT bounds
        within_gt_bounds_mask = (
            (xyz[:, 0] >= self.gt_x_min) & (xyz[:, 0] <= self.gt_x_max) &
            (xyz[:, 1] >= self.gt_y_min) & (xyz[:, 1] <= self.gt_y_max)
        )
        
        # Combine the masks to get only points we need to check
        points_to_check_mask = torch.logical_and(points_to_check_mask, within_gt_bounds_mask)
        
        # densified_to_check = torch.logical_and(densified_to_check, within_gt_bounds_mask)
        
        # Get indices of points we need to check
        indices_to_check = torch.nonzero(points_to_check_mask).squeeze(1)

        # indices_densified_to_check = torch.nonzero(densified_to_check).squeeze(1)
        
        # If no points need checking, return the existing mask
        if len(indices_to_check) == 0:
            return existing_mask if existing_mask is not None else torch.zeros(xyz.shape[0], dtype=bool, device=xyz.device)
        
        # Extract only the points we need to check
        xyz_to_check = xyz[indices_to_check]

        # xyz_densified_to_check = xyz[indices_densified_to_check]
        
        # Convert tensor to numpy for FAISS
        xyz_np_to_check = xyz_to_check.cpu().numpy().astype(np.float32)

        # xyz_densified_np_to_check = xyz_densified_to_check.cpu().numpy().astype(np.float32)
        
        # Search for nearest neighbors (only for points we're checking)
        distances, _ = self.gt_point_cloud.search(xyz_np_to_check, k=1)

        # distances_densified, _ = self.gt_point_cloud.search(xyz_densified_np_to_check, k=1)
        
        distance_prune_mask_checked = distances[:, 0] > distance_threshold**2
        # Create mask for checked points that are TOO FAR from ground truth (to be pruned)
        # distance_prune_mask_checked = distances[:, 0] > distance_threshold**2

        # distance_prune_mask_densified = distances_densified[:, 0] > distance_threshold**2

        # # Print the number of points that are too far for the densified points
        # print(f"Number of points that are too far for the densified points: {distance_prune_mask_densified.sum()} out of {len(xyz_densified_to_check)} (or {len(distance_prune_mask_densified)}")
        
        # Convert mask to torch tensor
        distance_prune_mask_checked = torch.from_numpy(distance_prune_mask_checked).to(xyz.device)
        
        # Initialize full mask with all False
        distance_prune_mask_full = torch.zeros(xyz.shape[0], dtype=bool, device=xyz.device)
        
        # Set the pruning status only for the checked points
        distance_prune_mask_full[indices_to_check] = distance_prune_mask_checked
        
        # If there's an existing mask, combine it with our distance-based mask (logical OR)
        if existing_mask is not None:
            combined_prune_mask = torch.logical_or(existing_mask, distance_prune_mask_full)
        else:
            combined_prune_mask = distance_prune_mask_full

        # Check how much existing_mask and protected_mask overlap to see if newly densified points are being pruned
        if existing_mask is not None:
            overlap_mask = torch.logical_and(existing_mask, protected_mask)
            overlap_count = overlap_mask.sum().item()
        
        # Count how many additional points will be pruned due to distance constraint
        additional_pruned = torch.sum(distance_prune_mask_full).item()
        total_pruned = torch.sum(combined_prune_mask).item()
        total_points_checked = len(indices_to_check)
        
        return combined_prune_mask