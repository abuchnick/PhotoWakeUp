import torch
import pickle as pkl
import cv2
import numpy as np
from Image_renderer import Renderer
import smplx
import torch
import torch.nn.functional as F
from human_body_prior.tools.model_loader import load_vposer
from smplx.lbs import batch_rigid_transform, batch_rodrigues

# COUNTER = 0

vposer, _ = load_vposer('./data/vposer_v1_0')

class Animation:
    def __init__(self, mesh_data, img, smplx_result, skinning_weights):
        self.model = smplx_result['model']
        self.smplx_mesh = smplx_result['mesh']
        self.mesh = mesh_data
        self.img = img
        self.renderer = Renderer(
            vertices=mesh_data['transformed_vertices'],  # result['mesh']['vertices'],
            faces=mesh_data['faces'],  # result['mesh']['faces'],
            img=img,
            camera_translation=smplx_result['camera']['translation'],
            camera_rotation=smplx_result['camera']['rotation']
        )
        self.num_joints = 22
        body_pose = vposer.decode(torch.tensor(smplx_result['params']['body_pose']), output_type='aa')
        self.full_pose = torch.cat([torch.zeros((1, 1, 3), dtype=self.model.dtype),
            body_pose.reshape(-1, self.model.NUM_BODY_JOINTS, 3)], dim=1).reshape(-1, 66)
        weights = torch.tensor(skinning_weights, dtype=self.model.dtype).unsqueeze(dim=0)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        print(f"{torch.all(torch.abs(weights.sum(dim=-1) - 1.0) < 0.01)}")
        weights = self.reduce_weights_dimensions(weights, self.model.parents, stop_index=self.num_joints)
        print(f"{torch.all(torch.abs(weights.sum(dim=-1) - 1.0) < 0.01)}")
        self.weights = weights
        self.zero_pose = self.get_zero_pose()

    def __call__(self, animation_poses: torch.Tensor, video_writer: cv2.VideoWriter, batch_size = 50):
        num_frames_left = animation_poses.shape[0]
        index = 0
        num_joints = 22
        while num_frames_left > 0:
            batch = min(batch_size, num_frames_left)

            rot_mats = smplx.lbs.batch_rodrigues(animation_poses[index:index+batch].view(-1, 3)).view(
                [batch, -1, 3, 3])
            posed_joints, rel_transforms = batch_rigid_transform(rot_mats, self.joints.expand(batch, -1, -1), self.model.parents[:22], dtype=self.joints.dtype)
            transformations = torch.matmul(self.weights, rel_transforms.view(batch, num_joints, 16)) \
                .view(batch, -1, 4, 4)
            v_posed_homo = F.pad(self.zero_pose, (0, 1), value=1)
            # v_posed_homo = F.pad(v, [0, 1], value=1.0)
            v_homo = torch.matmul(transformations, torch.unsqueeze(v_posed_homo, dim=-1))

            v = v_homo[:, :, :3, 0]

            for frame_vertices in v:
                rendered_frame = self.renderer.render_texture(vertices=frame_vertices.detach().squeeze().numpy())
                video_writer.write(rendered_frame)  
            num_frames_left -= batch
            index += batch

    def get_zero_pose(self) -> torch.Tensor:
        batch_size = 1
        v = torch.tensor(np.expand_dims(self.mesh['transformed_vertices'], axis=0), dtype=self.model.dtype)
        inverse_pose = self.full_pose * (-1.0)
        inverse_pose_rot_mats = batch_rodrigues(inverse_pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        num_joints = 22
        posed_joints = torch.tensor(self.smplx_mesh['joints_posed'][:22, :], dtype=self.model.dtype)
        self.view_joints(posed_joints)
        joints, rel_transforms = batch_rigid_transform(inverse_pose_rot_mats, posed_joints.reshape(1, -1, 3), self.model.parents[:22].reshape(-1), dtype=self.model.dtype)
        self.view_joints(joints)
        self.joints = joints
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        transformations = torch.matmul(self.weights, rel_transforms.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        v_posed_homo = F.pad(v, (0, 1), value=1)
        # v_posed_homo = F.pad(v, [0, 1], value=1.0)
        v_homo = torch.matmul(transformations, torch.unsqueeze(v_posed_homo, dim=-1))

        v = v_homo[:, :, :3, 0]

        self.view_texture(v.detach().squeeze().numpy())
        return v

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
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


    def view_texture(self, vertices):
        # global COUNTER
        img = self.renderer.render_texture(vertices=vertices)
        # cv2.imwrite(f'.\\temp\\test\\{COUNTER}.png', (255*(np.flip(normals[0], axis=2)+1.0)/2).astype(np.int8))
        # COUNTER += 1
        cv2.imshow('1', img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


    def view_joints(self, joints):
        # global COUNTER
        render = self.renderer.render_joints(joints.squeeze().detach().numpy())
        # cv2.imwrite(f'.\\temp\\test\\{COUNTER}.png', render)
        # COUNTER += 1
        cv2.imshow('1', render)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    with open('temp/trump/result.pkl', 'rb') as file:
        result = pkl.load(file)

    mesh = np.load('temp/trump/mesh_data.npz')
    img = cv2.imread(r'./data/images_temp/trump.jpg')
    skinning_weights = np.load('temp/trump/skinning_weights.npy')

    animation = Animation(mesh_data=mesh, img=img, smplx_result=result, skinning_weights=skinning_weights)
    exit()
    video_writer = cv2.VideoWriter('temp/trump/video2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 120, tuple(reversed(img.shape[:2])))
    poses = torch.tensor(np.load('skip_to_walk_full_pose.npy'), dtype=animation.model.dtype)
    poses[:, 0:3] = 0.0
    animation(poses, video_writer)
    video_writer.release()