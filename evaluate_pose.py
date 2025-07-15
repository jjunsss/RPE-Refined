# evaluate.py

import os
import json
import argparse
import collections
import numpy as np

from helper import io_utils, pose_calculator, metrics

def evaluate(pred_dir: str, gt_json_path: str, smplx_npz: str, out_dir: str,
             fname_key: str = "real_image",
             joint_field: str = "fitted_joint_cam",
             num_joints: int = 24):
    """
    Main evaluation loop to compute 3D pose metrics.
    This function orchestrates loading, calculation, and saving.
    """
    j_regressor = io_utils.load_smplx_regressor(smplx_npz)[:num_joints]
    with open(gt_json_path, "r") as f:
        gt_records = json.load(f)

    all_results = collections.OrderedDict()
    total_mpjpe, total_pa_mpjpe = 0.0, 0.0
    num_samples = 0

    print(f"Starting evaluation for {len(gt_records)} samples...")

    for record in gt_records:
        if fname_key == "id":
            stub = record["id"]
        else:
            stub = os.path.splitext(os.path.basename(record[fname_key]))[0]

        obj_path = os.path.join(pred_dir, f"{stub}.obj")
        pkl_path = os.path.join(pred_dir, f"{stub}.pkl")

        if not (os.path.isfile(obj_path) and os.path.isfile(pkl_path)):
            print(f"[WARNING] Missing prediction files for sample: {stub}")
            continue

        # ---- Step 1: Load data using io_utils ----
        pred_vertices = io_utils.load_obj_vertices(obj_path)
        gt_joints = np.array(record["ground_truth"][joint_field], dtype=np.float32)[:num_joints]

        # ---- Step 2: Calculate poses using pose_calculator ----
        pred_joints = pose_calculator.vertices_to_joints(pred_vertices, j_regressor)

        # ---- Step 3: Compute final scores using metrics ----
        # 1. MPJPE (requires root alignment from pose_calculator)
        err_mpjpe = metrics.calculate_mpjpe(
            pose_calculator.root_align_joints(pred_joints),
            pose_calculator.root_align_joints(gt_joints)
        )
        
        # 2. PA-MPJPE (requires Procrustes alignment from pose_calculator)
        aligned_pred_joints = pose_calculator.procrustes_alignment(pred_joints, gt_joints)
        err_pa_mpjpe = metrics.calculate_mpjpe(aligned_pred_joints, gt_joints)

        # ---- Step 4: Store and accumulate results ----
        all_results[stub] = {
            "MPJPE_mm": err_mpjpe,
            "PA_MPJPE_mm": err_pa_mpjpe,
        }
        total_mpjpe += err_mpjpe
        total_pa_mpjpe += err_pa_mpjpe
        num_samples += 1

    if num_samples > 0:
        mean_mpjpe = total_mpjpe / num_samples
        mean_pa_mpjpe = total_pa_mpjpe / num_samples

        print("\n" + "="*50)
        print(f"Evaluation finished for {num_samples} samples.")
        print(f"  - Mean MPJPE:      {mean_mpjpe:.2f} mm")
        print(f"  - Mean PA-MPJPE:   {mean_pa_mpjpe:.2f} mm")
        print("="*50 + "\n")

        all_results["__summary__"] = {
            "total_samples": num_samples,
            "mean_MPJPE_mm": mean_mpjpe,
            "mean_PA_MPJPE_mm": mean_pa_mpjpe,
        }
    else:
        print("No samples were processed.")
        return

    os.makedirs(out_dir, exist_ok=True)
    out_filename = f"evaluation_results_{os.path.basename(pred_dir)}.json"
    out_path = os.path.join(out_dir, out_filename)
    
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to: {out_path}")

def main():
    """Command-line interface setup."""
    parser = argparse.ArgumentParser(description="Evaluate 3D human pose and shape predictions.")
    
    parser.add_argument("--pred_dir", required=True, help="Directory containing predicted .obj and .pkl files.")
    parser.add_argument("--gt_json_path", required=True, help="Path to the ground truth JSON annotation file.")
    parser.add_argument("--smplx_npz", required=True, help="Path to the SMPL-X model file (for J_regressor).")
    parser.add_argument("--out_dir", required=True, help="Directory to save the output evaluation JSON file.")
    parser.add_argument("--fname_key", default="real_image", choices=["real_image", "image", "id"])
    parser.add_argument("--joint_field", default="fitted_joint_cam", choices=["fitted_joint_cam", "joint_cam"])
    parser.add_argument("--num_joints", type=int, default=24)
    args = parser.parse_args()

    evaluate(**vars(args))

if __name__ == "__main__":
    main()
