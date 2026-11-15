"""
image_utils.py
--------------
Small helper functions for reading images, saving things and
playing with boxes. Nothing clever here.

A "box" everywhere in this project is a tuple (x1, y1, x2, y2):
    x1, y1 = top-left corner
    x2, y2 = bottom-right corner
"""

import os
import json

import cv2
import numpy as np

from src.config import IMAGE_EXTENSIONS, WORK_WIDTH


def list_images(input_path):
    """Give back a list of image paths. input_path can be one file or a folder."""
    if os.path.isfile(input_path):
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError("Could not find: " + input_path)

    paths = []
    for name in sorted(os.listdir(input_path)):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            paths.append(os.path.join(input_path, name))
    return paths


def read_image(path):
    """
    Read an image as a colour (BGR) array.
    I use np.fromfile + imdecode instead of cv2.imread because cv2.imread
    fails on Windows if the path has special characters.
    Returns None if the file is broken.
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def resize_for_work(img):
    """
    Shrink a big image so everything runs faster.
    Returns the small image and the scale (original_width / small_width),
    which we need later to put the boxes back on the big image.
    """
    h, w = img.shape[:2]
    if w <= WORK_WIDTH:
        return img.copy(), 1.0

    scale = w / WORK_WIDTH
    new_h = int(round(h / scale))
    small = cv2.resize(img, (WORK_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    return small, scale


def clip_box(box, width, height):
    """Make sure a box never goes outside the image (assignment constraint)."""
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(x1, width - 1)))
    x2 = int(max(0, min(x2, width - 1)))
    y1 = int(max(0, min(y1, height - 1)))
    y2 = int(max(0, min(y2, height - 1)))
    return (x1, y1, x2, y2)


def scale_box(box, factor):
    """Multiply every number in the box by factor (used to go back to the original size)."""
    return tuple(int(round(v * factor)) for v in box)


def save_image(path, img):
    """Save an image (again using imencode so Windows paths are fine)."""
    ext = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise IOError("Could not encode image for " + path)
    encoded.tofile(path)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
