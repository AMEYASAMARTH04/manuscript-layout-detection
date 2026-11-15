"""
preprocessing.py
----------------
Everything that cleans up the picture BEFORE we try to find text:

    1. make it gray
    2. fix uneven lighting / faded ink
    3. find where the actual page is (palm leaves are photographed on a coloured cloth)
    4. turn the picture into a black/white "ink mask"
    5. find long straight lines (borders, ruling lines) so we don't mistake them for text
"""

import cv2
import numpy as np

from src.config import (
    BACKGROUND_KERNEL_FRACTION,
    PAGE_ERODE_FRACTION,
    RULING_MIN_LENGTH_FRACTION,
    BORDER_MIN_COVERAGE,
    BORDER_TIMES_NORMAL,
)


def make_gray(img):
    """Colour doesn't help much for finding text, so we go to gray."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def fix_lighting(gray):
    """
    Handles uneven lighting, stains and faded ink.

    Idea: if we could see the page WITHOUT any ink, we could divide the
    real page by it and the lighting would cancel out. To guess the
    "page without ink" we use a morphological closing (which erases small dark
    things like letters) and then blur it a bit.
    """
    h, w = gray.shape
    k = int(w * BACKGROUND_KERNEL_FRACTION)
    k = max(15, k) | 1                      # "| 1" makes sure the number is odd

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), k / 2)

    # gray / background * 255  -> paper becomes ~255 and ink stays dark
    flat = cv2.divide(gray, background, scale=255)

    # a tiny blur helps with noise and slightly blurry scans
    flat = cv2.GaussianBlur(flat, (3, 3), 0)
    return flat


def find_page(gray):
    """
    Finds the page (the palm leaf / paper) inside the photo.

    Returns:
        page_mask: white where the page is, black elsewhere
        page_box : (x1, y1, x2, y2) around the page

    If we can't find a clear page (for example a scan that is all paper),
    we just say "the whole image is the page".
    """
    h, w = gray.shape
    whole_image_mask = np.full((h, w), 255, dtype=np.uint8)
    whole_image_box = (0, 0, w - 1, h - 1)

    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # letters make little holes in the page shape, so we close them
    k = max(15, w // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return whole_image_mask, whole_image_box

    biggest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(biggest) / float(h * w)

    # too small = probably not a page, too big = the page IS the image
    if area_ratio < 0.3 or area_ratio > 0.95:
        return whole_image_mask, whole_image_box

    page_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(page_mask, [biggest], -1, 255, thickness=-1)

    # shave the edges a little (edges are shadowy and look like ink)
    e = max(3, int(w * PAGE_ERODE_FRACTION))
    page_mask = cv2.erode(page_mask, np.ones((e, e), np.uint8))

    x, y, bw, bh = cv2.boundingRect(biggest)
    return page_mask, (x, y, x + bw, y + bh)


def make_ink_mask(flat, page_mask):
    """
    Black/white picture where WHITE = ink.
    We use Otsu's method (it picks the best threshold automatically),
    but only look at pixels inside the page to choose the threshold, so the
    background doesn't confuse it.

    Note: we keep ink OUTSIDE the page too, because captions / stamps that
    sit on the background (like the watermark text on the palm leaf photo) have
    to be found and called "filler". We only blank a thin band right at the page
    edge, because the shadowy edge looks like ink but isn't.
    """
    inside = flat[page_mask > 0].reshape(-1, 1)
    threshold, _ = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ink = np.zeros_like(flat)
    ink[flat < threshold] = 255

    # thin band around the page edge (does nothing if the page is the whole image)
    w = flat.shape[1]
    e = max(3, int(w * PAGE_ERODE_FRACTION))
    grown = cv2.dilate(page_mask, np.ones((3 * e, 3 * e), np.uint8))
    edge_band = (grown > 0) & (page_mask == 0)
    ink[edge_band] = 0
    return ink


def find_ruling_lines(ink):
    """
    Finds page borders and ruling lines so we don't mistake them for writing.

    Vertical borders: I first tried morphology (opening with a tall thin kernel)
    but old scans have dashed, slightly crooked borders and it kept missing them.
    So now I just count, for every column of pixels, how much of the page height
    is covered by ink. A border column is inked almost all the way down (40%-80%),
    while a normal column of handwriting is only ~15%. Columns far above the
    normal level are borders.

    Horizontal lines: opening with a long flat kernel (only very long, very
    straight lines survive).

    Returns a mask of the lines (slightly thickened so we cover their edges).
    """
    h, w = ink.shape
    lines = np.zeros_like(ink)

    # ---- vertical borders: column coverage ----
    coverage = (ink > 0).sum(axis=0) / float(h)
    columns_with_ink = coverage[coverage > 0]
    if len(columns_with_ink) > 0:
        normal_level = np.median(columns_with_ink)
        limit = max(BORDER_MIN_COVERAGE, BORDER_TIMES_NORMAL * normal_level)
        is_border = coverage >= limit

        # make each border a bit wider (7 px) so we catch the whole stroke
        is_border = np.convolve(is_border.astype(int), np.ones(7), mode="same") > 0
        lines[:, is_border] = ink[:, is_border]

    # ---- horizontal lines: long flat kernel ----
    horiz_len = max(15, int(w * RULING_MIN_LENGTH_FRACTION))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 3))
    horizontal_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, horiz_kernel)
    lines = cv2.bitwise_or(lines, horizontal_lines)

    lines = cv2.dilate(lines, np.ones((5, 5), np.uint8))
    return lines
