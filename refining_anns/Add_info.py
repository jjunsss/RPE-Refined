#usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# temp.json에 3DPW Ground-Truth 데이터 추가 (경로 직접 수정 기능 포함)
# ------------------------------------------------------------------------------
# 특정 'real_image' 경로를 발견하면, 해당 경로 값을 새로운 경로로 직접 수정한 뒤
# 수정된 경로를 기준으로 GT 데이터를 찾아 저장합니다.
# ==============================================================================

import json
from pathlib import Path
import os

# ──────────────────── 사용자 설정 영역 ────────────────────
TEMP_JSON_PATH = 'refining_anns/refine_chatpose_vqa_shape.json'
THREEDPW_GT_PATH = '/home/uvll/jjunsss/naver/vlm/pose/dataset/3DPW/data/3DPW_test.json'

# ──────────────────────── 메인 로직 ────────────────────────
def main():
    """메인 실행 함수"""

    # (1) 파일 존재 여부 확인
    for p in [TEMP_JSON_PATH, THREEDPW_GT_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"필수 파일을 찾을 수 없습니다: {p}")

    # (2) 데이터 로드
    print("데이터 로딩 중...")
    with open(TEMP_JSON_PATH, 'r', encoding='utf-8') as f:
        temp_data = json.load(f)
    with open(THREEDPW_GT_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    # (3) Ground-Truth 데이터 맵 생성
    print("Ground-Truth 데이터 맵 생성 중...")
    gt_map = {}
    image_id_to_info = {img['id']: img for img in gt_data['images']}
    for ann in gt_data['annotations']:
        image_id = ann['image_id']
        if image_id in image_id_to_info:
            img_info = image_id_to_info[image_id]
            seq_name = img_info['sequence']
            file_name = img_info['file_name']
            gt_map[(seq_name, file_name)] = ann

    # (4) temp.json의 각 항목에 Ground-Truth 데이터 추가
    print("JSON 파일 처리 시작...")
    found_count = 0
    not_found_count = 0
    for entry in temp_data:
        real_image_path = Path(entry['real_image'])
        seq_name = real_image_path.parts[-2]
        file_name = real_image_path.name

        # ==========================================================
        # ▼▼▼ 'real_image' 경로 자체를 직접 변경하는 로직 ▼▼▼
        # ==========================================================
        if seq_name == 'downtown_runForBus_00' and file_name == 'image_00373.jpg':
            new_file_name = 'image_00479.jpg'
            original_path_str = entry['real_image']
            
            # 1. entry 딕셔너리의 'real_image' 값을 새로운 경로로 업데이트
            entry['real_image'] = str(real_image_path.parent / new_file_name)
            
            print(f"✅ 경로 수정: '{original_path_str}' -> '{entry['real_image']}'")
            
            # 2. GT를 찾기 위한 파일 이름도 변경
            file_name = new_file_name
        # ==========================================================
        
        lookup_key = (seq_name, file_name)

        if lookup_key in gt_map:
            entry['ground_truth'] = gt_map[lookup_key]
            found_count += 1
        else:
            entry['ground_truth'] = None
            not_found_count += 1
            print(f"⚠️ 경고: '{seq_name}/{file_name}'에 해당하는 Ground-Truth를 찾을 수 없습니다.")

    # (5) 업데이트된 데이터로 새 JSON 파일 저장
    print("\n작업 완료.")
    print(f"✅ 총 {len(temp_data)}개 항목 중 {found_count}개에 Ground-Truth 추가 완료.")
    if not_found_count > 0:
        print(f"❌ {not_found_count}개 항목은 일치하는 데이터를 찾지 못했습니다.")

    root, ext = os.path.splitext(TEMP_JSON_PATH)
    OUTPUT_JSON_PATH = f"{root}_addvalue{ext}"
    print(f"결과를 '{OUTPUT_JSON_PATH}'에 저장합니다.")

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(temp_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()