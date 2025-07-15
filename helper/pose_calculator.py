# pose_calculator.py

import numpy as np

def vertices_to_joints(vertices: np.ndarray, regressor: np.ndarray) -> np.ndarray:
    """
    Calculates joint locations from vertices using a joint regressor.

    Args:
        vertices (np.ndarray): (V, 3) array of mesh vertices.
        regressor (np.ndarray): (J, V) joint regressor matrix.

    Returns:
        np.ndarray: (J, 3) array of joint locations.
    """
    return regressor @ vertices

def root_align_joints(joints: np.ndarray) -> np.ndarray:
    """
    Aligns joints by subtracting the root joint's (joint 0) position.

    Args:
        joints (np.ndarray): (J, 3) array of joint locations.

    Returns:
        np.ndarray: (J, 3) array of root-aligned joint locations.
    """
    return joints - joints[0:1, :]

def procrustes_alignment(pred_joints: np.ndarray, gt_joints: np.ndarray) -> np.ndarray:
    """
    Performs Procrustes alignment to find the best similarity transform
    (scale, rotation, translation) that maps predicted joints to ground truth joints.

    Args:
        pred_joints (np.ndarray): (J, 3) absolute predicted joint coordinates.
        gt_joints (np.ndarray): (J, 3) absolute ground truth joint coordinates.

    Returns:
        np.ndarray: (J, 3) aligned predicted joints.
    """
    mu_pred = pred_joints.mean(axis=0, keepdims=True)
    mu_gt = gt_joints.mean(axis=0, keepdims=True)

    X1 = (pred_joints - mu_pred).T # (3, J)
    X2 = (gt_joints - mu_gt).T   # (3, J)

    K = X1 @ X2.T
    U, _, Vt = np.linalg.svd(K)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    s = np.trace(R @ K) / np.sum(X1**2)
    t = mu_gt.T - s * R @ mu_pred.T

    aligned_pred = (s * (R @ pred_joints.T) + t).T
    return aligned_pred

def axis_angle_to_rot_mat(axis_angles: np.ndarray) -> np.ndarray:
    """
    Converts a batch of axis-angles to 3x3 rotation matrices using Rodrigues' formula.

    Args:
        axis_angles (np.ndarray): (J, 3) array of axis-angles.

    Returns:
        np.ndarray: (J, 3, 3) array of rotation matrices.
    """
    num_joints = axis_angles.shape[0]
    rot_mats = np.zeros((num_joints, 3, 3), dtype=np.float32)
    for i, a in enumerate(axis_angles):
        theta = np.linalg.norm(a)
        if theta < 1e-8:
            rot_mats[i] = np.eye(3)
            continue
        
        k = a / theta
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]], dtype=np.float32)
        
        rot_mats[i] = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return rot_mats