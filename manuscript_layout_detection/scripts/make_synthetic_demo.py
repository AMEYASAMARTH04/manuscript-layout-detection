"""
make_synthetic_demo.py
----------------------
The two real sample pages only have main text + watermarks. To check that the
header / footer / side_text rules also work, this script builds a SYNTHETIC page:
it takes the body of the paper sample, pastes it on a bigger sheet with wide margins
and writes a header, a footer and margin notes with a normal computer font.

It is only a test image (clearly named "synthetic"), not real manuscript data.

Run from the project folder:
    python scripts/make_synthetic_demo.py
"""

import os
import sys

import cv2
import numpy as np

# so "from src..." works when running this file from the project folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_utils import read_image, save_image

SRC = os.path.join("data", "test_images", "sample_paper.png")
OUT = os.path.join("data", "test_images", "synthetic_demo_page.png")


def main():
    paper = read_image(SRC)

    # crop only the handwriting rows (no watermark, no border lines)
    body = paper[92:690, 238:1940]

    h, w = 1000, 2150
    rng = np.random.default_rng(0)
    page = np.full((h, w, 3), (205, 225, 235), dtype=np.uint8)       # parchment colour
    noise = rng.normal(0, 4, (h, w, 1)).astype(np.int16)
    page = np.clip(page.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # paste the body in the middle
    top, left = 200, 230
    page[top:top + body.shape[0], left:left + body.shape[1]] = body

    ink = (30, 30, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # header: running title above the body
    cv2.putText(page, "CHAPTER THREE", (860, 130), font, 1.6, ink, 3, cv2.LINE_AA)

    # footer: catchword + folio number under the body
    cv2.putText(page, "ra-ma-ya", (1600, 880), font, 1.4, ink, 3, cv2.LINE_AA)
    cv2.putText(page, "12", (230, 880), font, 1.4, ink, 3, cv2.LINE_AA)

    # margin notes (marginalia) left and right, smaller writing
    for i, word in enumerate(["note", "see p.4", "check"]):
        cv2.putText(page, word, (20, 330 + i * 60), font, 1.0, ink, 2, cv2.LINE_AA)
    for i, word in enumerate(["later", "added"]):
        cv2.putText(page, word, (1990, 450 + i * 60), font, 1.0, ink, 2, cv2.LINE_AA)

    save_image(OUT, page)
    print("saved", OUT)


if __name__ == "__main__":
    main()
