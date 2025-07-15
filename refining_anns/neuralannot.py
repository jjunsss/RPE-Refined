#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# NeuralAnnot SMPL‑X → 실제 이미지 합성 (Multi‑GPU 병렬 처리)
# ----------------------------------------------------------------------
# 이 스크립트는 3DPW(3‑D Pose in the Wild) 데이터셋의 COCO 주석과
# NeuralAnnot로 추정한 SMPL‑X 파라미터를 이용해,
#   1) 사람 메쉬(SMPL‑X)를 생성하고
#   2) 원본 이미지 위에 합성(Render)한 뒤
#   3) 시퀀스별·프레임별 PNG 파일로 저장
# 하는 과정을 자동화합니다.
#
# - GPU가 있으면 Multi‑GPU, 없으면 Multi‑CPU로 자동 전환해
#   여러 시퀀스를 병렬로 빠르게 처리합니다.
# - “3점 조명(Three‑Point Lighting)”을 사용해 자연스러운 광원을 재현합니다.
# - 2025‑06‑18: Multi‑GPU 모드와 CPU 모드를 하나로 통합(코드 단순화)
# ==============================================================================

# ────────────────────── 모듈 import ──────────────────────
import os
import json
from pathlib import Path
import numpy as np
import torch
import trimesh                     # 삼각형 기반 3‑D 메쉬 처리
import pyrender                    # OpenGL(또는 EGL) 기반 3‑D 렌더러
import smplx                       # SMPL‑X 인체 모델
from PIL import Image
from pycocotools.coco import COCO  # COCO 포맷 주석 파서
import torch.multiprocessing as mp # 멀티프로세싱(GPU/CPU)

# ──────────────────── 사용자 설정 영역 ────────────────────
MODEL_PATH          = './smpl_models'                       # SMPL‑X 모델 파일(.pkl) 저장 폴더
COCO_ANNOTATION_PATH = './data/3DPW_test.json'              # 3DPW → COCO 변환 주석
SMPLX_PARAM_PATH     = './SMPL-X/3DPW_test_SMPLX_NeuralAnnot.json'  # NeuralAnnot가 예측한 SMPL‑X 파라미터
IMAGE_FOLDER_PATH    = './imageFiles'                       # 원본 이미지 폴더(시퀀스별 하위디렉터리 포함)

SEQUENCE_NAME   = None  # None이면 모든 시퀀스, 문자열이면 해당 시퀀스만 예: 'downtown_arguing_00'
FRAME_INTERVAL  = 10    # n프레임마다 한 장씩 렌더(속도 vs. 품질 절충)
NUM_GPUS        = 4     # 최대 사용 GPU 개수(CUDA 미사용 시 CPU 프로세스 수도 여기서 결정)
RENDER_OUT_DIR_BASE = Path('./render_output_composite_coco')  # 결과물 저장 루트

# ───────────────────── 보조 함수들 ──────────────────────
def apply_opengl_conversion(mesh: trimesh.Trimesh):
    """
    OpenGL 카메라 좌표계는 +Y가 위, +Z가 카메라(뷰어) 방향인
    ‘카메라 기준 좌표계’입니다.
    SMPL‑X는 +Z가 위인 ‘월드 기준 좌표계’를 쓰므로,
    x축 180° 회전으로 두 좌표계를 맞춰 줍니다.
    """
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    return mesh

def setup_scene_lighting(scene: pyrender.Scene):
    """
    Three‑Point Lighting 세팅:
      • Key Light   : (0, −1, 1)
      • Fill Light  : (0,  1, 1)
      • Back Light  : (1,  1, 2)
    각 광원은 Directional Light로 설정해
    ‘멀리서 동일한 방향으로 쏘는 평행광’을 구현합니다.
    """
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)

    # Key Light
    pose1 = np.eye(4); pose1[:3, 3] = [0, -1, 1]
    scene.add(light, pose=pose1)

    # Fill Light
    pose2 = np.eye(4); pose2[:3, 3] = [0, 1, 1]
    scene.add(light, pose=pose2)

    # Back Light
    pose3 = np.eye(4); pose3[:3, 3] = [1, 1, 2]
    scene.add(light, pose=pose3)

# ───────────────── Worker 프로세스 ─────────────────
def worker(rank: int,
           world_size: int,
           all_seq_names: list[str],
           coco_path: str,
           smplx_path: str,
           use_cuda: bool):
    """
    rank        : 현재 프로세스 번호(0‑based)
    world_size  : 전체 프로세스 개수
    all_seq_names: 전체 시퀀스 목록(이 중 일부만 분담 처리)
    use_cuda    : True면 GPU 전용, False면 CPU 전용

    각 워커는 다음 순서로 동작합니다.
      1) 자신의 장치(GPU/CPU) 지정
      2) 데이터(주석, 파라미터) 로드
      3) 할당된 시퀀스 순회
      4) 프레임마다:
         ‑ SMPL‑X → 정점(verts) 생성
         ‑ trimesh로 메쉬 생성·좌표계 변환
         ‑ pyrender Scene 구성(카메라+조명+메쉬)
         ‑ Off‑screen 렌더(깊이 포함)
         ‑ RGBA 이미지를 원본 배경에 합성
         ‑ PNG 저장
    """
    # (1) EGL 모드로 렌더러 초기화(OpenGL 디스플레이 불필요)
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    if use_cuda:
        # 멀티프로세싱 환경에서 GPU 불가분성을 위해
        # 각 프로세스에 1개 GPU만 노출
        gpu_id = rank
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device("cuda:0")
        prefix = f"[GPU {gpu_id}]"
    else:
        device = torch.device("cpu")
        prefix = f"[CPU‑Proc {rank}]"

    # (2) 이 워커가 담당할 시퀀스 나누기
    seqs_local = np.array_split(all_seq_names, world_size)[rank]
    if len(seqs_local) == 0:
        print(f"{prefix} 처리할 시퀀스 없음 → 바로 종료")
        return
    print(f"{prefix} 담당 시퀀스: {', '.join(seqs_local)}")

    # (3) COCO 및 SMPL‑X 파라미터 로드
    db = COCO(coco_path)
    with open(smplx_path) as f:
        smplx_params_all = json.load(f)

    # 이미지 메타데이터 미리 캐싱(성능 ↑)
    img_ids = db.getImgIds()
    imgs_all = db.loadImgs(img_ids)

    # (4) 시퀀스별 렌더링 루프
    for seq in sorted(seqs_local):
        # frame_idx → 이미지 메타데이터 매핑
        frame_map = {im['frame_idx']: im for im in imgs_all if im['sequence'] == seq}
        if not frame_map:
            print(f"{prefix}[경고] 시퀀스 ‘{seq}’ 없음 → 건너뜀")
            continue

        out_dir = RENDER_OUT_DIR_BASE / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{prefix} ‘{seq}’ 처리 시작 → 저장 경로: {out_dir}")

        # 해상도 고정(시퀀스 내 첫 이미지 사용)
        W, H = frame_map[min(frame_map)]['width'], frame_map[min(frame_map)]['height']
        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

        # n프레임 간격 렌더링
        for fidx in sorted(frame_map)[::FRAME_INTERVAL]:
            im_info = frame_map[fidx]

            # 이미지에 포함된 인물 주석 찾기
            ann_ids = db.getAnnIds(imgIds=im_info['id'])
            anns = db.loadAnns(ann_ids)
            if not anns:
                continue  # 인물 없는 프레임

            # 빈 Scene 시작(투명 배경)
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0],
                                   ambient_light=[.3, .3, .3, 1])
            person_present = False

            # (4‑1) 프레임에 등장하는 사람 반복
            for ann in anns:
                aid = str(ann['id'])
                if aid not in smplx_params_all:
                    continue  # SMPL‑X 파라미터 없으면 스킵

                # SMPL‑X 파라미터 추출
                p = smplx_params_all[aid]

                # SMPL‑X 모델 인스턴스(성별 포함)
                smplx_model = smplx.create(
                    MODEL_PATH, model_type='smplx',
                    gender=ann.get('gender', 'neutral'),
                    use_pca=False,
                    num_betas=len(p['shape'])
                ).to(device)

                # 정점(verts) 예측: with torch.no_grad() → 그래프 비생성
                with torch.no_grad():
                    verts = smplx_model(
                        betas=torch.tensor(p['shape'],   device=device).unsqueeze(0),
                        global_orient=torch.tensor(p['root_pose'], device=device).unsqueeze(0),
                        body_pose=  torch.tensor(p['body_pose'],  device=device).unsqueeze(0),
                        transl=     torch.tensor(p['trans'],      device=device).unsqueeze(0),
                        return_verts=True
                    ).vertices[0].cpu().numpy()  # (N_v, 3)

                # trimesh로 메쉬 구성, 좌표계 변환(OpenGL)
                mesh = trimesh.Trimesh(verts, smplx_model.faces)
                mesh = apply_opengl_conversion(mesh)

                # 머티리얼 및 Scene에 추가
                mat = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    alphaMode='OPAQUE',
                    baseColorFactor=(1.0, 1.0, 0.9, 1.0)  # 살색 톤
                )
                scene.add(pyrender.Mesh.from_trimesh(mesh, material=mat))
                person_present = True

            if not person_present:
                continue  # 렌더할 대상 없음

            # (4‑2) 카메라 Intrinsics 추가
            cam_p = {k: np.array(v) for k, v in im_info['cam_param'].items()}
            camera = pyrender.IntrinsicsCamera(
                fx=cam_p['focal'][0], fy=cam_p['focal'][1],
                cx=cam_p['princpt'][0], cy=cam_p['princpt'][1]
            )
            scene.add(camera)

            # (4‑3) 조명 세팅
            setup_scene_lighting(scene)

            # (4‑4) 오프스크린 렌더(RGBA + depth)
            rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            if not (depth > 0).any():
                continue  # 깊이값 전부 0 → 메쉬가 카메라 밖

            # (4‑5) 배경 이미지와 합성
            render_img = Image.fromarray(rgba, 'RGBA')
            bg_path = Path(IMAGE_FOLDER_PATH) / im_info['sequence'] / im_info['file_name']
            if not bg_path.exists():
                continue  # 원본 이미지 없으면 스킵
            bg = Image.open(bg_path).convert('RGBA')
            bg.paste(render_img, (0, 0), render_img)  # 알파채널 기준 붙여넣기

            # (4‑6) 저장
            out_path = out_dir / f"f{fidx:05d}.png"
            bg.save(out_path)

        renderer.delete()  # 메모리 누수 방지
        print(f"{prefix} ‘{seq}’ 완료")

# ──────────────────────── 메인 ────────────────────────
def main():
    # (1) 필수 경로 체크(사전에 빠르게 오류 감지)
    for p in [MODEL_PATH, COCO_ANNOTATION_PATH, SMPLX_PARAM_PATH, IMAGE_FOLDER_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"필수 경로를 찾을 수 없습니다 → {p}")

    # (2) 전체 시퀀스 목록 확보
    db = COCO(COCO_ANNOTATION_PATH)
    imgs = db.loadImgs(db.getImgIds())
    seq_names = sorted({im['sequence'] for im in imgs})
    all_seq_names = [SEQUENCE_NAME] if SEQUENCE_NAME else seq_names

    # (3) GPU/CPU 자원 확인
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        avail = torch.cuda.device_count()
        world_size = min(NUM_GPUS, avail)
        if avail < NUM_GPUS:
            print(f"[경고] GPU {avail}개만 사용 가능 → {world_size}개로 진행")
        print(f"총 {len(all_seq_names)}개 시퀀스를 {world_size}개 GPU로 병렬 처리")
    else:
        world_size = NUM_GPUS  # CPU 프로세스 수
        print(f"[경고] CUDA 미사용 → CPU {world_size}개 프로세스로 처리")

    print(f"대상 시퀀스: {all_seq_names}")

    # (4) 멀티프로세싱 실행
    mp.spawn(
        worker,
        args=(world_size, all_seq_names,
              COCO_ANNOTATION_PATH, SMPLX_PARAM_PATH, use_cuda),
        nprocs=world_size,
        join=True
    )

    print(f"\n모든 시퀀스 처리 완료! 결과: {RENDER_OUT_DIR_BASE}")

# 스크립트를 직접 실행할 때만 main() 호출
if __name__ == "__main__":
    main()
