import torch
import pickle as pkl
import cv2
import numpy as np
from Image_renderer import Renderer
import import_smplifyx as smplifyx
import smplx
import torch
import torch.nn.functional as F
from human_body_prior.tools.model_loader import load_vposer
from smplx.lbs import vertices2joints, batch_rigid_transform, batch_rodrigues

# COUNTER = 0

vposer, _ = load_vposer('./data/vposer_v1_0')

class Animation:
    def __init__(self, mesh_data, img, smplx_result):
        self.model = smplx_result['model']
        self.mesh = mesh_data
        self.img = img
        self.renderer = Renderer(
            vertices=mesh_data['transformed_vertices'],  # result['mesh']['vertices'],
            faces=mesh_data['faces'],  # result['mesh']['faces'],
            img=img,
            camera_translation=smplx_result['camera']['translation'],
            camera_rotation=smplx_result['camera']['rotation']
        )
        body_pose = vposer.decode(torch.tensor(smplx_result['params']['body_pose']), output_type='aa')
        self.full_pose = torch.cat([torch.zeros((1, 1, 3), dtype=body_pose.dtype),
            body_pose.reshape(-1, self.model.NUM_BODY_JOINTS, 3)], dim=1, dtype=body_pose.dtype).reshape(-1, 66)

    

    @staticmethod
    def reduce_weights_dimensions(weights: torch.Tensor, parents: torch.Tensor, stop_index: int=22) -> torch.Tensor:
        # weights: BxVxJ
        # parents: J
        # stop index: int
        J = weights.shape[-1]
        new_weights = torch.clone(weights[:, :, :stop_index])
        root_joints_sets = [set() for _ in range(stop_index)]
        for i in range(stop_index, J):
            joints_set = set()
            while i >= stop_index:
                joints_set.add(i)
                i = int(parents[i])
            root_joints_sets[i].update(joints_set)
        for i, j_set in enumerate(root_joints_sets):
            for j in j_set:
                new_weights[:, :, i] += weights[:, :, j]
        return new_weights


    def view_normals(self):
        # global COUNTER
        normals = self.renderer.render_normals()
        # cv2.imwrite(f'.\\temp\\test\\{COUNTER}.png', (255*(np.flip(normals[0], axis=2)+1.0)/2).astype(np.int8))
        # COUNTER += 1
        cv2.imshow('1', (np.flip(normals[0], axis=2)+1.0)/2)
        cv2.waitKey()
        cv2.destroyAllWindows()


    def view_texture(self, vertices):
        # global COUNTER
        img = self.renderer.render_texture(vertices=vertices)
        # cv2.imwrite(f'.\\temp\\test\\{COUNTER}.png', (255*(np.flip(normals[0], axis=2)+1.0)/2).astype(np.int8))
        # COUNTER += 1
        cv2.imshow('1', img)
        cv2.waitKey(5)
        # cv2.destroyAllWindows()


    def view_joints(self, joints):
        # global COUNTER
        render = self.renderer.render_joints(joints.squeeze().detach().numpy())
        # cv2.imwrite(f'.\\temp\\test\\{COUNTER}.png', render)
        # COUNTER += 1
        cv2.imshow('1', render)
        cv2.waitKey()
        cv2.destroyAllWindows()


##########################################################

full_pose = torch.cat(
    [torch.zeros((1, 1, 3), dtype=body_pose.dtype),  # model.global_orient.reshape(-1, 1, 3),
     body_pose.reshape(-1, model.NUM_BODY_JOINTS, 3)  # ,
     #  model.jaw_pose.reshape(-1, 1, 3) ,
     #  model.leye_pose.reshape(-1, 1, 3),
     #  model.reye_pose.reshape(-1, 1, 3),
     #  left_hand_pose.reshape(-1, model.NUM_HAND_JOINTS, 3),
     #  right_hand_pose.reshape(-1, model.NUM_HAND_JOINTS, 3)
     ], dim=1).reshape(-1, 66)

batch_size = full_pose.shape[0]
v = torch.tensor(np.expand_dims(mesh['transformed_vertices'], axis=0))
# renderer.set_vertices(v.squeeze().detach().numpy())
# view_normals(renderer)
view_texture(renderer, vertices=v.squeeze().detach().numpy())

inverse_pose = full_pose * (-1.0)
inverse_pose_rot_mats = batch_rodrigues(inverse_pose.view(-1, 3)).view(
    [batch_size, -1, 3, 3])

num_joints = 22  # model.J_regressor.shape[0]
posed_joints = torch.tensor(result['mesh']['joints_posed'][:22, :])
view_joints(renderer, posed_joints)
joints, rel_transforms = batch_rigid_transform(inverse_pose_rot_mats, posed_joints.reshape(1, -1, 3), model.parents[:22].reshape(-1), dtype=posed_joints.dtype)
view_joints(renderer, joints)
weights = torch.tensor(np.load('skinning_weights.npy'), dtype=posed_joints.dtype).unsqueeze(dim=0)
weights = weights / weights.sum(dim=-1, keepdim=True)
print(f"{torch.all(torch.abs(weights.sum(dim=-1) - 1.0) < 0.01)}")
weights = reduce_weights_dimensions(weights, model.parents, stop_index=num_joints)
print(f"{torch.all(torch.abs(weights.sum(dim=-1) - 1.0) < 0.01)}")
# (N x V x (J + 1)) x (N x (J + 1) x 16)
transformations = torch.matmul(weights, rel_transforms.view(batch_size, num_joints, 16)) \
    .view(batch_size, -1, 4, 4)

v_posed_homo = F.pad(v, (0, 1), value=1)
# v_posed_homo = F.pad(v, [0, 1], value=1.0)
v_homo = torch.matmul(transformations, torch.unsqueeze(v_posed_homo, dim=-1))

v = v_homo[:, :, :3, 0]

# renderer.set_vertices(v.squeeze().detach().numpy())
# view_normals(renderer)
view_texture(renderer, vertices=v.squeeze().detach().numpy())


skip_to_walk_full_pose = torch.tensor(np.load('skip_to_walk_full_pose.npy'), dtype=v.dtype)[::4].clone()
skip_to_walk_full_pose[:, 0:3] = 0.0
batch_size = skip_to_walk_full_pose.shape[0]
rot_mats = smplx.lbs.batch_rodrigues(skip_to_walk_full_pose.view(-1, 3)).view(
    [batch_size, -1, 3, 3])
posed_joints, rel_transforms = batch_rigid_transform(rot_mats, joints.expand(batch_size, -1, -1), model.parents[:22], dtype=joints.dtype)
transformations = torch.matmul(weights, rel_transforms.view(batch_size, num_joints, 16)) \
    .view(batch_size, -1, 4, 4)
v_posed_homo = F.pad(v, (0, 1), value=1)
# v_posed_homo = F.pad(v, [0, 1], value=1.0)
v_homo = torch.matmul(transformations, torch.unsqueeze(v_posed_homo, dim=-1))

v = v_homo[:, :, :3, 0]

# renderer.set_vertices(v.squeeze().detach().numpy())
# view_normals(renderer)
for i in range(batch_size):
    view_texture(renderer, vertices=v[i].squeeze().detach().numpy())
cv2.destroyAllWindows()
exit()