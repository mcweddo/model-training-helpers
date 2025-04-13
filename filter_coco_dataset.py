import os
import json
import shutil
import argparse
from pathlib import Path

import mmcv
import cv2
from PIL import Image, UnidentifiedImageError
from mmengine.fileio import FileClient


def is_image_valid(image_path):
    try:
        file_client = FileClient('disk')  # default backend for local FS
        img_bytes = file_client.get(image_path)

        # Try with OpenCV backend
        mmcv.use_backend('cv2')
        img = mmcv.imfrombytes(img_bytes, flag='color')
        if img is not None and img.size != 0:
            return True

        # Fallback to Pillow backend
        # mmcv.use_backend('pillow')
        # img = mmcv.imfrombytes(img_bytes, flag='color')
        # if img is not None and img.size != 0:
        #     return True

    except Exception as e:
        print(e)
        pass

    return False


def find_and_quarantine_bad_images(image_dir, quarantine_dir=None, delete_bad=False):
    bad_images = []

    for root, _, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                path = os.path.join(root, fname)
                if not is_image_valid(path):
                    print(f"[BAD] {path}")
                    bad_images.append(path)

                    if delete_bad:
                        os.remove(path)
                    elif quarantine_dir:
                        os.makedirs(quarantine_dir, exist_ok=True)
                        shutil.move(path, os.path.join(quarantine_dir, fname))

    return bad_images


def rebuild_coco_annotations(original_json_path, bad_image_paths, output_json_path):
    bad_filenames = {Path(p).name for p in bad_image_paths}

    with open(original_json_path, 'r') as f:
        coco = json.load(f)

    original_images = coco.get("images", [])
    original_annotations = coco.get("annotations", [])

    valid_images = [img for img in original_images if img["file_name"] not in bad_filenames]
    valid_image_ids = {img["id"] for img in valid_images}
    valid_annotations = [ann for ann in original_annotations if ann["image_id"] in valid_image_ids]

    filtered_coco = dict(coco)
    filtered_coco["images"] = valid_images
    filtered_coco["annotations"] = valid_annotations

    with open(output_json_path, 'w') as f:
        json.dump(filtered_coco, f, indent=2)

    print(f"\n[INFO] Filtered annotations saved to: {output_json_path}")
    print(f"[INFO] Retained {len(valid_images)} images and {len(valid_annotations)} annotations.")


def main():
    parser = argparse.ArgumentParser(description="Filter COCO dataset for malformed images.")
    parser.add_argument('--images', nargs='+', required=True,
                        help="Paths to image folders (e.g., train2017 val2017)")
    parser.add_argument('--annots', nargs='+', required=True,
                        help="Paths to COCO annotation JSON files (same order as images)")
    parser.add_argument('--out-annots', nargs='+', required=True,
                        help="Paths to save filtered annotation JSONs (same order as images)")
    parser.add_argument('--quarantine-dir', default=None,
                        help="Folder to move bad images to. Required if --delete is not set.")
    parser.add_argument('--delete', action='store_true',
                        help="Delete bad images instead of moving them.")
    args = parser.parse_args()

    if not args.delete and not args.quarantine_dir:
        parser.error("Either --delete or --quarantine-dir must be specified.")

    if not (len(args.images) == len(args.annots) == len(args.out_annots)):
        parser.error("The number of --images, --annots, and --out-annots must match.")

    for image_dir, annot_path, output_annot_path in zip(args.images, args.annots, args.out_annots):
        print(f"\n[INFO] Processing dataset:")
        print(f"       Image folder : {image_dir}")
        print(f"       Annotations  : {annot_path}")

        quarantine_dir = None
        if args.quarantine_dir:
            quarantine_dir = os.path.join(args.quarantine_dir, Path(image_dir).name)

        bad_images = find_and_quarantine_bad_images(
            image_dir,
            quarantine_dir=quarantine_dir,
            delete_bad=args.delete
        )

        rebuild_coco_annotations(annot_path, bad_images, output_annot_path)

    print("\n[DONE] All datasets processed.")


if __name__ == "__main__":
    main()
