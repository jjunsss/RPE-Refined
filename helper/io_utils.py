# io_utils.py

import os
import pickle
import numpy as np
import torch

def load_obj_vertices(path: str) -> np.ndarray:
    """
    Reads 'v x y z' lines from an .obj file.

    Args:
        path (str): Path to the .obj file.

    Returns:
        np.ndarray: A (V, 3) numpy array of vertices, where V is the vertex count.
    """
    vertices = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append(list(map(float, line.split()[1:4])))
    return np.asarray(vertices, dtype=np.float32)

def load_smplx_regressor(npz_path: str) -> np.ndarray:
    """
    Loads the J_regressor from a SMPL-X .npz file.

    Args:
        npz_path (str): Path to the SMPL-X .npz file.

    Returns:
        np.ndarray: The (J, V) joint regressor matrix.
    """
    with np.load(npz_path) as data:
        return data["J_regressor"].astype(np.float32)

def load_pred_rotmat(pkl_path: str) -> np.ndarray:
    """
    Loads predicted rotation matrices from a .pkl file.
    The file should contain 'global_orient' and 'body_pose'.

    Args:
        pkl_path (str): Path to the .pkl file.

    Returns:
        np.ndarray: A (24, 3, 3) array of rotation matrices, stacked as
                    [global_orient(1), body_pose(23)].
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Ensure tensors are on CPU and converted to numpy
    global_orient = data["global_orient"].cpu().numpy() # (1, 1, 3, 3)
    body_pose = data["body_pose"].cpu().numpy()       # (1, 23, 3, 3)
    
    # Concatenate and remove the batch dimension
    rot_matrices = np.concatenate([global_orient, body_pose], axis=1).squeeze(0) # (24, 3, 3)
    return rot_matrices