"""
pipeline.py
-----------
Puts all the steps together.

    analyze_image()  -> runs the steps on ONE image already loaded in memory
    run_on_path()    -> handles a single file or a whole folder, saves the results
"""

import os

from src.image_utils import (
    list_images, read_image, resize_for_work, clip_box, scale_box,
    save_image, save_json,
)
from src.preprocessing import (
    make_gray, fix_lighting, find_page, make_ink_mask, find_ruling_lines,
)
from src.line_finder import (
    clean_ink, guess_char_height, split_edge_stamp_text, find_text_blobs, find_line_boxes,
)
from src.classifier import classify_blobs
from src.visualize import draw_regions


def analyze_image(img):
    """
    Runs the whole detection on one colour image.

    Returns:
        regions : list of dicts (label, bbox, confidence, reason) in ORIGINAL image pixels
        page_box: (x1, y1, x2, y2) of the page in original pixels
        steps   : dictionary with the in-between pictures (used by the notebook to show them)
    """
    orig_h, orig_w = img.shape[:2]

    # --- preprocessing (on the smaller copy) ---
    small, scale = resize_for_work(img)
    small_h, small_w = small.shape[:2]

    gray = make_gray(small)
    flat = fix_lighting(gray)
    page_mask, page_box = find_page(gray)
    ink = make_ink_mask(flat, page_mask)
    lines = find_ruling_lines(ink)

    # take the lines out of the ink so they are not counted as writing
    ink_without_lines = ink.copy()
    ink_without_lines[lines > 0] = 0

    # --- finding blobs ---
    cleaned = clean_ink(ink_without_lines)
    char_h = guess_char_height(cleaned)
    stamp_ink, body_ink = split_edge_stamp_text(cleaned, char_h)
    blobs = find_text_blobs(body_ink, char_h) + find_text_blobs(stamp_ink, char_h)
    line_boxes = find_line_boxes(lines)

    # --- deciding what each blob is ---
    regions_small = classify_blobs(blobs, line_boxes, page_mask, small_w, small_h, char_h)

    # --- go back to original image size and keep boxes inside the image ---
    regions = []
    for r in regions_small:
        big_box = scale_box(r["box"], scale)
        big_box = clip_box(big_box, orig_w, orig_h)
        regions.append({
            "label": r["label"],
            "bbox": list(big_box),
            "confidence": r["confidence"],
            "reason": r["reason"],
        })

    # sort so the output reads top to bottom
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    page_box_big = clip_box(scale_box(page_box, scale), orig_w, orig_h)

    steps = {
        "small": small, "gray": gray, "flat": flat, "page_mask": page_mask,
        "ink": ink, "lines": lines, "cleaned": cleaned, "stamp_ink": stamp_ink,
        "body_ink": body_ink, "blobs": blobs,
        "char_h": char_h, "scale": scale,
    }
    return regions, list(page_box_big), steps


def run_on_path(input_path, output_dir):
    """
    input_path : one image file or a folder of images
    output_dir : where annotated images and json files are saved
    Returns a list with one result dictionary per image.
    """
    image_paths = list_images(input_path)
    if len(image_paths) == 0:
        print("No images found in", input_path)
        return []

    os.makedirs(output_dir, exist_ok=True)

    # safety: never write into the folder the inputs come from, so originals stay untouched
    input_folder = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
    if os.path.abspath(input_folder) == os.path.abspath(output_dir):
        raise ValueError("Output folder must be different from the input folder.")

    all_results = []
    for number, path in enumerate(image_paths, start=1):
        name = os.path.basename(path)
        print("[%d/%d] %s" % (number, len(image_paths), name))

        img = read_image(path)
        if img is None:
            print("   could not read this file, skipping it")
            continue

        regions, page_box, _ = analyze_image(img)
        h, w = img.shape[:2]

        result = {
            "image": name,
            "width": w,
            "height": h,
            "page_box": page_box,
            "regions": regions,
        }
        all_results.append(result)

        base = os.path.splitext(name)[0]
        save_json(os.path.join(output_dir, base + ".json"), result)
        save_image(os.path.join(output_dir, base + "_annotated.png"), draw_regions(img, regions))

        counts = {}
        for r in regions:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        print("   found:", counts)

    save_json(os.path.join(output_dir, "all_predictions.json"), all_results)
    print("Done. Results are in", output_dir)
    return all_results
