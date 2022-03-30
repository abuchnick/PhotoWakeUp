from argparse import ArgumentError
import sys
import os

import os.path as osp

import time
from typing import Optional
import yaml
import torch
import numpy as np

import smplx

import import_smplifyx as smplifyx

from smplifyx.utils import JointMapper
from smplifyx.data_parser import create_dataset
from smplifyx.fit_single_frame import fit_single_frame

from smplifyx.camera import create_camera
from smplifyx.prior import create_prior

from human_body_prior.tools.model_loader import load_vposer

import pickle

torch.backends.cudnn.enabled = False

PROJECT_ROOT = realpath(join(__file__, ".."))

class SmplifyX:
    def __init__(self, conf_file: str = osp.join(PROJECT_ROOT, "smplx_conf.yaml"), conf_override: Optional[dict] = None):
        with open(conf_file, "r") as file:
            self.configuration = yaml.load(file, Loader=yaml.FullLoader)
        if conf_override is not None:
            self.configuration.update(conf_override)
        if self.configuration['use_cuda'] and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        float_dtype = self.configuration['float_dtype']
        if float_dtype == 'float64':
            self.dtype = torch.float64
        elif float_dtype == 'float32':
            self.dtype = torch.float32
        else:
            raise Exception(
                'Unknown float type {}, exiting!'.format(float_dtype))

    def __call__(self):
        cwd = os.getcwd()
        try:
            # change working directory to project root
            os.chdir(os.path.abspath(os.path.join(__file__, '..')))
            self.run()
        finally:
            os.chdir(cwd)

    def run(self):
        output_folder = self.configuration.pop('output_folder')
        if not osp.exists(output_folder):
            os.makedirs(output_folder)

        result_folder = self.configuration.pop('result_folder', 'results')
        result_folder = osp.join(output_folder, result_folder)
        if not osp.exists(result_folder):
            os.makedirs(result_folder)

        mesh_folder = self.configuration.pop('mesh_folder', 'meshes')
        mesh_folder = osp.join(output_folder, mesh_folder)
        if not osp.exists(mesh_folder):
            os.makedirs(mesh_folder)

        out_img_folder = osp.join(output_folder, 'images')
        if not osp.exists(out_img_folder):
            os.makedirs(out_img_folder)

        img_folder = self.configuration.pop('img_folder', 'images')
        dataset_obj = create_dataset(
            img_folder=img_folder, **self.configuration)

        start = time.time()

        input_gender = self.configuration.pop('gender', 'neutral')
        gender_lbl_type = self.configuration.pop('gender_lbl_type', 'none')
        max_persons = self.configuration.pop('max_persons', -1)

        joint_mapper = JointMapper(dataset_obj.get_model2data())

        model_params = dict(model_path=self.configuration.get('model_folder'),
                            joint_mapper=joint_mapper,
                            create_global_orient=True,
                            create_body_pose=not self.configuration.get(
                                'use_vposer'),
                            create_betas=True,
                            create_left_hand_pose=True,
                            create_right_hand_pose=True,
                            create_expression=True,
                            create_jaw_pose=True,
                            create_leye_pose=True,
                            create_reye_pose=True,
                            create_transl=False,
                            dtype=self.dtype,
                            **self.configuration)

        male_model = smplx.create(gender='male', **model_params)
        # SMPL-H has no gender-neutral model
        if self.configuration.get('model_type') != 'smplh':
            neutral_model = smplx.create(gender='neutral', **model_params)
        female_model = smplx.create(gender='female', **model_params)

        # Create the camera object
        focal_length = self.configuration.get('focal_length')
        camera = create_camera(focal_length_x=focal_length,
                               focal_length_y=focal_length,
                               dtype=self.dtype,
                               **self.configuration)

        if hasattr(camera, 'rotation'):
            camera.rotation.requires_grad = False

        use_hands = self.configuration.get('use_hands', True)
        use_face = self.configuration.get('use_face', True)

        body_pose_prior = create_prior(
            prior_type=self.configuration.get('body_prior_type'),
            dtype=self.dtype,
            **self.configuration)

        jaw_prior, expr_prior = None, None
        if use_face:
            jaw_prior = create_prior(
                prior_type=self.configuration.get('jaw_prior_type'),
                dtype=self.dtype,
                **self.configuration)
            expr_prior = create_prior(
                prior_type=self.configuration.get('expr_prior_type', 'l2'),
                dtype=self.dtype, **self.configuration)

        left_hand_prior, right_hand_prior = None, None
        if use_hands:
            lhand_args = self.configuration.copy()
            lhand_args['num_gaussians'] = self.configuration.get(
                'num_pca_comps')
            left_hand_prior = create_prior(
                prior_type=self.configuration.get('left_hand_prior_type'),
                dtype=self.dtype,
                use_left_hand=True,
                **lhand_args)

            rhand_args = self.configuration.copy()
            rhand_args['num_gaussians'] = self.configuration.get(
                'num_pca_comps')
            right_hand_prior = create_prior(
                prior_type=self.configuration.get('right_hand_prior_type'),
                dtype=self.dtype,
                use_right_hand=True,
                **rhand_args)

        shape_prior = create_prior(
            prior_type=self.configuration.get('shape_prior_type', 'l2'),
            dtype=self.dtype, **self.configuration)

        angle_prior = create_prior(prior_type='angle', dtype=self.dtype)

        camera = camera.to(device=self.device)
        female_model = female_model.to(device=self.device)
        male_model = male_model.to(device=self.device)
        if self.configuration.get('model_type') != 'smplh':
            neutral_model = neutral_model.to(device=self.device)
        body_pose_prior = body_pose_prior.to(device=self.device)
        angle_prior = angle_prior.to(device=self.device)
        shape_prior = shape_prior.to(device=self.device)
        if use_face:
            expr_prior = expr_prior.to(device=self.device)
            jaw_prior = jaw_prior.to(device=self.device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=self.device)
            right_hand_prior = right_hand_prior.to(device=self.device)

        # A weight for every joint of the model
        joint_weights = dataset_obj.get_joint_weights().to(device=self.device,
                                                           dtype=self.dtype)
        # Add a fake batch dimension for broadcasting
        joint_weights.unsqueeze_(dim=0)

        batch_result = {}
        for idx, data in enumerate(dataset_obj):
            img = data['img']
            fn = data['fn']
            keypoints = data['keypoints']
            print('Processing: {}'.format(data['img_path']))

            curr_result_folder = osp.join(result_folder, fn)
            if not osp.exists(curr_result_folder):
                os.makedirs(curr_result_folder)
            curr_mesh_folder = osp.join(mesh_folder, fn)
            if not osp.exists(curr_mesh_folder):
                os.makedirs(curr_mesh_folder)

            curr_result = batch_result[fn] = {}

            for person_id in range(keypoints.shape[0]):
                if person_id >= max_persons and max_persons > 0:
                    continue

                curr_result_fn = osp.join(curr_result_folder,
                                          '{:03d}.pkl'.format(person_id))
                curr_mesh_fn = osp.join(curr_mesh_folder,
                                        '{:03d}.obj'.format(person_id))

                curr_img_folder = osp.join(output_folder, 'images', fn,
                                           '{:03d}'.format(person_id))
                if not osp.exists(curr_img_folder):
                    os.makedirs(curr_img_folder)

                if gender_lbl_type != 'none':
                    if gender_lbl_type == 'pd' and 'gender_pd' in data:
                        gender = data['gender_pd'][person_id]
                    if gender_lbl_type == 'gt' and 'gender_gt' in data:
                        gender = data['gender_gt'][person_id]
                else:
                    gender = input_gender

                if gender == 'neutral':
                    body_model = neutral_model
                elif gender == 'female':
                    body_model = female_model
                elif gender == 'male':
                    body_model = male_model

                out_img_fn = osp.join(curr_img_folder, 'output.png')

                fit_single_frame(img, keypoints[[person_id]],
                                 body_model=body_model,
                                 camera=camera,
                                 joint_weights=joint_weights,
                                 dtype=self.dtype,
                                 output_folder=output_folder,
                                 result_folder=curr_result_folder,
                                 out_img_fn=out_img_fn,
                                 result_fn=curr_result_fn,
                                 mesh_fn=curr_mesh_fn,
                                 shape_prior=shape_prior,
                                 expr_prior=expr_prior,
                                 body_pose_prior=body_pose_prior,
                                 left_hand_prior=left_hand_prior,
                                 right_hand_prior=right_hand_prior,
                                 jaw_prior=jaw_prior,
                                 angle_prior=angle_prior,
                                 **self.configuration)
                with open(curr_result_fn, 'rb') as file:
                    params = pickle.load(file)
                curr_result[idx] = dict(
                    model=body_model,
                    params=params,
                    mesh=self.get_mesh_from_params(
                        model=body_model, params=params),
                    camera=dict(
                        focal_length=focal_length,
                        **{k: v.detach().cpu().numpy().squeeze() for k, v in camera.named_parameters()},
                        **{k: v.detach().cpu().numpy().squeeze() for k, v in camera.named_buffers()}
                    )
                )

        elapsed = time.time() - start
        time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                                 time.gmtime(elapsed))
        print('Processing the data took: {}'.format(time_msg))
        return batch_result

    def get_mesh_from_params(self, model=None, params=None, params_file=None):
        if params is None:
            with open(params_file, "rb") as file:
                params = pickle.load(file)

        if model is None:
            dataset_obj = create_dataset(**self.configuration)
            joint_mapper = JointMapper(dataset_obj.get_model2data())
            model_params = dict(model_path=self.configuration['model_folder'],
                                joint_mapper=joint_mapper,
                                create_global_orient=True,
                                global_orient=params['global_orient'],
                                create_body_pose=not self.configuration['use_vposer'],
                                create_betas=True,
                                create_left_hand_pose=True,
                                create_right_hand_pose=True,
                                create_expression=True,
                                create_jaw_pose=True,
                                create_leye_pose=True,
                                create_reye_pose=True,
                                create_transl=False,
                                dtype=self.dtype,
                                **self.configuration)
            model = smplx.create(**model_params)
        model = model.to(self.device)

        body_pose = torch.tensor(params['body_pose'])
        if self.configuration['use_vposer']:
            vposer, _ = load_vposer(
                self.configuration['vposer_ckpt'], vp_model='snapshot')
            vposer = vposer.to(device=self.device)
            body_pose = vposer.decode(
                body_pose,
                output_type='aa').view(1, -1)
        model_output = model(return_verts=True, body_pose=body_pose)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()
        joints = model_output.joints.detach().cpu().numpy().squeeze()
        # convert to OpenGL compatible axis
        vertices = vertices  @ np.diag([1, -1, -1])
        joints = joints  @ np.diag([1, -1, -1])
        faces = model.faces.detach().cpu().numpy().squeeze()
        skinning_map = model.lbs_weights.detach().cpu().numpy().squeeze()
        return dict(vertices=vertices, faces=faces, skinning_map=skinning_map, joints=joints)


if __name__ == "__main__":
    smplifyx_object = SmplifyX()
    result = smplifyx_object()
    with open('result.pkl', 'wb') as file:
        pickle.dump(result, file)
