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


def view_normals(renderer):
    # global COUNTER
    normals = renderer.render_normals()
    # cv2.imwrite(f'.\\temp\\test\\{COUNTER}.png', (255*(np.flip(normals[0], axis=2)+1.0)/2).astype(np.int8))
    # COUNTER += 1
    cv2.imshow('1', (np.flip(normals[0], axis=2)+1.0)/2)
    cv2.waitKey()
    cv2.destroyAllWindows()


def view_texture(renderer, vertices):
    # global COUNTER
    img = renderer.render_texture(vertices=vertices)
    # cv2.imwrite(f'.\\temp\\test\\{COUNTER}.png', (255*(np.flip(normals[0], axis=2)+1.0)/2).astype(np.int8))
    # COUNTER += 1
    cv2.imshow('1', img)
    cv2.waitKey(5)
    # cv2.destroyAllWindows()


def view_joints(renderer, joints):
    # global COUNTER
    render = renderer.render_joints(joints.squeeze().detach().numpy())
    # cv2.imwrite(f'.\\temp\\test\\{COUNTER}.png', render)
    # COUNTER += 1
    cv2.imshow('1', render)
    cv2.waitKey()
    cv2.destroyAllWindows()


def reduce_weights_dimensions(weights, parents, stop_index=22):
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


with open('result.pkl', 'rb') as file:
    result = pkl.load(file)

model = result['model']
mesh = np.load('mesh_data.npz')

renderer = Renderer(
    vertices=mesh['transformed_vertices'],  # result['mesh']['vertices'],
    faces=mesh['faces'],  # result['mesh']['faces'],
    img=cv2.imread(r'./data/images_temp/goku.jpg'),
    camera_translation=result['camera']['translation'],
    camera_rotation=result['camera']['rotation']
)

# global_transform = smplx.lbs.batch_rodrigues(model.global_orient.view(1, 3)).squeeze().detach()

# view_normals(renderer)
# v = model.v_template
# renderer.set_vertices(v.detach().numpy().squeeze() + np.array([0, 0.6, 0]))
# view_normals(renderer)
# v = (v + smplx.lbs.blend_shapes(model.betas, model.shapedirs))
# renderer.set_vertices(v.detach().numpy().squeeze() + np.array([0, 0.6, 0]))
# view_normals(renderer)

vposer, _ = load_vposer('./data/vposer_v1_0')

body_pose = vposer.decode(torch.tensor(result['params']['body_pose']), output_type='aa')

left_hand_pose = model.left_hand_pose @ model.left_hand_components
right_hand_pose = model.right_hand_pose @ model.right_hand_components


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
inverse_pose_rot_mats = smplx.lbs.batch_rodrigues(inverse_pose.view(-1, 3)).view(
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

#############################################################

full_pose = torch.cat(
    [model.global_orient.reshape(-1, 1, 3),
     body_pose.reshape(-1, model.NUM_BODY_JOINTS, 3),
     #  model.jaw_pose.reshape(-1, 1, 3) ,
     #  model.leye_pose.reshape(-1, 1, 3),
     #  model.reye_pose.reshape(-1, 1, 3),
     #  left_hand_pose.reshape(-1, model.NUM_HAND_JOINTS, 3),
     #  right_hand_pose.reshape(-1, model.NUM_HAND_JOINTS, 3)
     ], dim=1).reshape(-1, 66)

# full_pose += model.pose_mean

batch_size = full_pose.shape[0]

rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view(
    [batch_size, -1, 3, 3])

# ident = torch.eye(3)
# pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
# # (N x P) x (P, V * 3) -> N x V x 3
# pose_offsets = torch.matmul(
#     pose_feature, model.posedirs).view(batch_size, -1, 3)

# v = (v + pose_offsets)
# renderer.set_vertices(v.detach() + np.array([0, 0.6, 0]))
# normals3 = renderer.render_normals()


joints = vertices2joints(model.J_regressor, v)[:, :22, :]
posed_joints, rel_transforms = batch_rigid_transform(rot_mats, joints, model.parents[:22], dtype=joints.dtype)

weights = model.lbs_weights.unsqueeze(dim=0)
print(f"{torch.all(torch.abs(weights.sum(dim=-1) - 1.0) < 0.01)}")
num_joints = 22  # model.J_regressor.shape[0]
weights = reduce_weights_dimensions(weights, model.parents, stop_index=num_joints)
print(f"{torch.all(torch.abs(weights.sum(dim=-1) - 1.0) < 0.01)}")
# (N x V x (J + 1)) x (N x (J + 1) x 16)
transformations = torch.matmul(weights, rel_transforms.view(batch_size, num_joints, 16)) \
    .view(batch_size, -1, 4, 4)

homogen_coord = torch.ones([batch_size, v.shape[1], 1],
                           dtype=v.dtype, device=v.device)
v_posed_homo = torch.cat([v, homogen_coord], dim=2)
# v_posed_homo = F.pad(v, [0, 1], value=1.0)
v_homo = torch.matmul(transformations, torch.unsqueeze(v_posed_homo, dim=-1))

v = v_homo[:, :, :3, 0]

renderer.set_vertices(v.squeeze().detach().numpy() @ np.diag([1, -1, -1]))
normals4 = renderer.render_normals()

model_output_posed = model(return_verts=True, body_pose=body_pose)
v = model_output_posed.vertices
renderer.set_vertices(v.squeeze().detach().numpy() @ np.diag([1, -1, -1]))
normals5 = renderer.render_normals()

cv2.imshow('1', (np.flip(normals4[0], axis=2)+1.0)/2)
cv2.waitKey()
# cv2.imshow('1', (np.flip(normals5[0], axis=2)+1.0)/2)
# cv2.waitKey()

cv2.waitKey()

cv2.destroyAllWindows()
