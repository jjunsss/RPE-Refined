# metrics.py

import numpy as np

def calculate_mpjpe(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
    """
    Calculates the Mean Per Joint Position Error (MPJPE) in millimeters.

    Args:
        pred_joints (np.ndarray): (J, 3) predicted joints.
        gt_joints (np.ndarray): (J, 3) ground truth joints.

    Returns:
        float: The MPJPE value in mm.
    """
    return float(np.mean(np.linalg.norm(pred_joints - gt_joints, axis=1)) * 1000.0)

def calculate_mpjre(pred_rot_mats: np.ndarray, gt_rot_mats: np.ndarray, to_deg: bool = False) -> float:
    """
    Calculates the Mean Per Joint Rotation Error (MPJRE) using geodesic distance.

    Args:
        pred_rot_mats (np.ndarray): (J, 3, 3) predicted rotation matrices.
        gt_rot_mats (np.ndarray): (J, 3, 3) ground truth rotation matrices.
        to_deg (bool): If True, converts the result to degrees. Otherwise, returns radians.

    Returns:
        float: The MPJRE value.
    """
    r_dot_rt = np.einsum('...ij,...kj->...ik', pred_rot_mats, gt_rot_mats)
    traces = np.einsum('...ii->...', r_dot_rt)
    
    cos_theta = np.clip((traces - 1) * 0.5, -1.0, 1.0)
    
    mean_error = np.mean(np.arccos(cos_theta))

    if to_deg:
        return float(np.degrees(mean_error))
    return float(mean_error)