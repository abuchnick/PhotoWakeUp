import numpy as np
from smplx.lbs import batch_rodrigues
import torch


def calc_rot_matrices(pose, framerate):
    return batch_rodrigues(pose.view(-1, 3)).view([framerate, -1, 3, 3])
    
    
def rotation_matrices_extractor():
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    animation = np.load(r'./data/skip_to_walk.npz')
    num_betas = animation['num_betas']
    time_length = len(animation['trans'])

    body_params = {
        'root_orient': torch.Tensor(animation['poses'][:, :3]).to(comp_device), # controls the global root orientation
        'pose_body': torch.Tensor(animation['poses'][:, 3:66]).to(comp_device), # controls the body
        'pose_hand': torch.Tensor(animation['poses'][:, 66:]).to(comp_device), # controls the finger articulation
        'trans': torch.Tensor(animation['trans']).to(comp_device), # controls the global body position
        'betas': torch.Tensor(np.repeat(animation['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    }

    full_pose = torch.cat([body_params['root_orient'], body_params['pose_body']], dim=1)
    rot_matrices = calc_rot_matrices(full_pose, framerate=time_length).detach().cpu().numpy()
    np.save(arr=rot_matrices, file='skip_to_walk_rot_mats.npy')


if __name__ == '__main__':
    rotation_matrices_extractor()