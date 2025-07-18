# Refined Annotations for the RPE Benchmark [![Paper  arXiv](https://img.shields.io/badge/Paper-arXiv-EB5424?style=flat&labelColor=555555)](https://arxiv.org/abs/2507.13314)

This repository provides refined and corrected ground-truth (GT) annotations for the **Reasoning-based Pose Estimation (RPE)** benchmark from *ChatPose (CVPR'24)*. We address critical technical reproducibility issues that hinder fair evaluations of pose-aware MLLMs.

---

## 📍 Motivation

The original RPE benchmark has critical limitations:

1. **Technical Issues**
   - Mismatched image indices that don't align with 3DPW dataset
   - Manual matching required for correct GT annotations

2. **Dataset Quality Issues** *(not addressed here)*
   - Image redundancy and limited scenario diversity
---
## 📁 Refined RPE Annotations

```
RPE-refined/
├── refine_vqa_behavior_withGT.json
├── refine_vqa_outfits_withGT.json
├── refine_vqa_pose_withGT.json
└── refine_vqa_shape_withGT.json
```

Each annotation entry contains the following structure:
```
{
  "id": "000000",
  "image": "000000.png",
  "real_image": "3DPW/imageFiles/downtown_runForBus_01/image_00326.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nThe man is about to sit down, can you give the SMPL pose of this person?"
    },
    {
      "from": "gpt",
      "value": "Sure, it is <POSE>."
    }
  ],
  "ground_truth": {
    "joint_cam": [[x1, y1, z1], ...],           // 24 SMPL joints in camera coords
    "fitted_joint_cam": [[x1, y1, z1], ...],    // Fitted SMPL joints
    "joint_img": [[u1, v1], ...],               // 2D projected keypoints
    "smpl_param": {
      "shape": [...],                            // Shape parameters (10D)
      "pose": [...],                             // Pose parameters (72D => 24 joints * 3D rotation)
      "trans": [tx, ty, tz],                     // Translation
      "gender": "male"
    },
    "bbox": [x, y, width, height]
  }
}
```

### Evaluation code
```
bash

# Example evaluation with ChatPose predictions
python scripts/evaluate_pose.py \
   --pred_dir "annot_output/chatpose-RPE-shape-bf16" \
   --gt_json_path "RPE-refined/refine_vqa_shape_withGT.json" \
   --smplx_npz "data/SMPLX_NEUTRAL_2020.npz" \
   --out_dir "./chatpose-eval-bf16" \
   --fname_key "real_image" \
   --joint_field "fitted_joint_cam" \
   --num_joints "22"
```

### Parameters
   - `--pred_dir`: Directory containing model predictions
   - `--gt_json_path`: Path to refined ground truth annotations
   - `--smplx_npz`: SMPLX model file
   - `--out_dir`: Output directory for evaluation results
   - `--fname_key`: Image filename key in annotations
   - `--joint_field`: Joint field to evaluate (you can choice two options: fitted_joint_cam or joint_cam)
   - `--num_joints`: Number of joints to evaluate


## Human Pose MLLMs & Tools

Validated with state-of-the-art pose-aware MLLMs and annotation tools:
- **UniPose** ([GitHub](https://github.com/liyiheng23/UniPose)) - A unified framework for human pose estimation and reasoning
- **ChatPose** ([GitHub](https://github.com/yfeng95/PoseGPT)) - The original pose-aware MLLM that introduced the RPE benchmark
- **NeuralAnnot** ([GitHub](https://github.com/mks0601/NeuralAnnot_RELEASE)) - The original annotations

## Status

- [x] **Refined annotations** publicly available
- [x] **Evaluation code** provided above
- [x] **Technical report** on arXiv

