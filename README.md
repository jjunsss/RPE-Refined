# Refined Annotations for the RPE Benchmark

This repository provides refined and corrected ground-truth (GT) annotations for the **Reasoning-based Pose Estimation (RPE)** benchmark, which was initially proposed in *ChatPose (CVPR'24)*. Our work addresses critical technical reproducibility issues identified in the original benchmark, which hinder fair and consistent quantitative evaluations.

The goal of this repository is to provide the research community with a reliable resource to enable robust, consistent, and reproducible evaluations of pose-aware Multimodal Large Language Models (MLLMs).

---

## 📍 Motivation

While the RPE benchmark has become influential, we have identified critical limitations in two main categories (which will be detailed in our upcoming technical report):

1. **Technical Reproducibility Issues**
   - **Mismatched Indices:** The benchmark uses custom image indices that do not align with the original 3DPW dataset.
   - **Manual Matching Required:** Researchers must manually align samples to obtain correct GT annotations, making evaluations tedious and error-prone.

2. **Dataset Quality Issues** *(not directly addressed in this repo)*
   - Image redundancy
   - Limited scenario diversity
   - Overly simplistic scenes
   - Ambiguous and repetitive queries

## 🛠 What This Repo Provides

This repository addresses **only the technical reproducibility issues** described above:

1. **Corrected GT Annotations**  
   We manually resolved image index mismatches and verified each sample via visual inspection. The refined JSON files are now aligned with the original 3DPW dataset and include key attributes such as `real_image`, `smpl_params`, `joint_cam`, and `fitted_joint_cam`, freeing pose-aware MLLM researchers from incorrect or labor-intensive evaluation setups.

> Annotations were extracted using the [NeuralAnnot repository](https://github.com/mks0601/NeuralAnnot_RELEASE).

## ✅ Planned Releases

- [ ] **Public release** of issue analysis technical report on arXiv  
- [ ] **Open-source codebase** for pose-aware MLLM evaluation (MPJPE / PA-MPJPE / MPJRE)  

Stay tuned!
