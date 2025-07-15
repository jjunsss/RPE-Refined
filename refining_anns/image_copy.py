import os
import json
import shutil

SOURCE_ROOT = '/home/uvll/jjunsss/naver/vlm/pose/dataset'

def load_and_copy(json_list_path, use_real_image=False, dest_root=None):
    with open(json_list_path, 'r') as f:
        records = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(json_list_path))
    if dest_root is None:
        dest_root = os.path.join(base_dir, 'copied_images')
    os.makedirs(dest_root, exist_ok=True)
    for rec in records:
        key = 'real_image' if use_real_image else 'image'
        rel_path = rec[key]
        if rel_path.startswith('3DPW'):
            src = os.path.join(SOURCE_ROOT, rel_path)
        else:
            src = os.path.join(base_dir, rel_path)
        if not os.path.isfile(src):
            print(f"Warning: skipping missing {src}")
            continue
        dst = os.path.join(dest_root, os.path.basename(rel_path))
        shutil.copy2(src, dst)
    print(f"Copied {len(records)} images into '{dest_root}'.")

if __name__ == '__main__':
    JSON_PATH = 'refining_anns/refine_chatpose_vqa_behavior_addvalue.json'
    use_real = 'addvalue' in JSON_PATH
    load_and_copy(
        json_list_path=JSON_PATH,
        use_real_image=use_real
    )
