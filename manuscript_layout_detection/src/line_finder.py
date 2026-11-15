"""
line_finder.py
--------------
Takes the black/white ink mask and turns it into "blobs".
A blob is a group of letters that sit close together (a line, a paragraph,
a page number ...). Later, classifier.py decides what each blob is.

Steps:
    1. clean the ink mask (remove dust, remove big solid stains / holes)
    2. guess how tall one character is
    3. pull small stamp text (watermarks) in the top/bottom strip away from the body
    4. glue neighbouring letters together by dilating, and read off each blob
"""

import cv2
import numpy as np

from src.config import (
    MERGE_WIDTH_IN_CHARS,
    MERGE_HEIGHT_IN_CHARS,
    MIN_BLOB_INK_RATIO,
    SMALL_TEXT_RATIO,
    EDGE_STRIP_FRACTION,
)


def clean_ink(ink):
    """
    Removes
      - tiny specks (dust, noise)
      - leftover bits of long thin ruling lines
      - big SOLID blobs (a string hole in a palm leaf, ink stains)
    Real letters are never completely filled in, so "fill ratio" tells them apart.

    Returns the cleaned mask and the list of component stats we computed.
    """
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    min_area = max(12, int(h * w * 0.00002))
    solid_area = int((w * 0.012) ** 2)      # roughly "bigger than a letter"

    keep = np.zeros(n, dtype=bool)          # keep[i] is True if component i is a good one
    for i in range(1, n):                   # component 0 is the background, skip it
        box_w = stats[i, cv2.CC_STAT_WIDTH]
        box_h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue                        # dust

        fill_ratio = area / float(box_w * box_h)
        if area > solid_area and fill_ratio > 0.6:
            continue                        # solid blob = hole or stain

        # leftover pieces of ruling lines: very long and very thin
        thin_limit = w * 0.01
        if box_w > 12 * box_h and box_h < thin_limit:
            continue                        # thin horizontal streak
        if box_h > 12 * box_w and box_w < thin_limit and box_h > 0.15 * h:
            continue                        # thin vertical streak

        keep[i] = True

    cleaned = (keep[labels].astype(np.uint8)) * 255
    return cleaned


def guess_char_height(cleaned):
    """
    How tall is one letter on this page? We need this so the code works
    for both tiny and huge handwriting.

    I only look at the BIGGEST 30% of components (by area), because the
    small ones are often dots, noise or little English letters, and I
    don't want them to fool me. Then I take the median height.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if n <= 1:
        return 20                           # nothing found, just return some number

    areas = stats[1:, cv2.CC_STAT_AREA]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]

    order = np.argsort(areas)[::-1]         # biggest first
    how_many = max(3, int(len(order) * 0.3))
    biggest_heights = heights[order[:how_many]]

    return int(max(8, np.median(biggest_heights)))


def split_edge_stamp_text(cleaned, char_h):
    """
    Watermarks / library stamps (small English text in the top or bottom strip
    of the image) can sit very close to the real writing, and then the "glue" step
    sticks them to the main text. So before gluing, I pull out every letter that
      - is clearly smaller than the real handwriting, and
      - sits inside the top or bottom strip of the image.
    They get glued on their own later.

    Returns (stamp_ink, body_ink), two masks that together make up `cleaned`.
    """
    h, w = cleaned.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    is_stamp = np.zeros(n, dtype=bool)
    for i in range(1, n):
        piece_h = stats[i, cv2.CC_STAT_HEIGHT]
        centre_y = centroids[i][1]
        small = piece_h < SMALL_TEXT_RATIO * char_h
        in_strip = centre_y < h * EDGE_STRIP_FRACTION or centre_y > h * (1 - EDGE_STRIP_FRACTION)
        if small and in_strip:
            is_stamp[i] = True

    stamp_ink = (is_stamp[labels].astype(np.uint8)) * 255
    body_ink = cv2.subtract(cleaned, stamp_ink)
    return stamp_ink, body_ink


def find_text_blobs(cleaned, char_h):
    """
    Glue letters into blobs and measure each blob.

    Returns a list of dictionaries, one per blob:
        box    : tight box around the ink (x1, y1, x2, y2)
        ink    : how many ink pixels are in the blob
        char_h : median height of the letters in this blob
        n_chars: how many separate letter-pieces it has
    """
    # the glue: dilate with a box that is a bit wider than a letter
    kw = max(3, int(char_h * MERGE_WIDTH_IN_CHARS)) | 1
    kh = max(3, int(char_h * MERGE_HEIGHT_IN_CHARS)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    glued = cv2.dilate(cleaned, kernel)

    n_blobs, blob_labels = cv2.connectedComponents(glued, connectivity=8)

    # now look at the letter pieces of the CLEAN mask
    n_pieces, piece_labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    # find out which blob each piece belongs to
    # (every ink pixel of a piece lies in the same blob, so any pixel will do)
    blob_of_piece = np.zeros(n_pieces, dtype=int)
    ink_pixels = piece_labels > 0
    blob_of_piece[piece_labels[ink_pixels]] = blob_labels[ink_pixels]

    # pieces much smaller than a normal one are probably dots / specks, I don't
    # want one far-away dot to stretch the box, so they are ignored for the box size
    median_area = np.median(stats[1:, cv2.CC_STAT_AREA]) if n_pieces > 1 else 0

    blobs = []
    for b in range(1, n_blobs):
        pieces = np.where(blob_of_piece == b)[0]
        pieces = pieces[pieces > 0]
        if len(pieces) == 0:
            continue

        ink = int(stats[pieces, cv2.CC_STAT_AREA].sum())

        solid = pieces[stats[pieces, cv2.CC_STAT_AREA] >= 0.1 * median_area]
        if len(solid) == 0:
            solid = pieces
        xs1 = stats[solid, cv2.CC_STAT_LEFT]
        ys1 = stats[solid, cv2.CC_STAT_TOP]
        xs2 = xs1 + stats[solid, cv2.CC_STAT_WIDTH]
        ys2 = ys1 + stats[solid, cv2.CC_STAT_HEIGHT]

        # throw away blobs that are too small to be real writing
        if ink < MIN_BLOB_INK_RATIO * char_h * char_h:
            continue

        blobs.append({
            "box": (int(xs1.min()), int(ys1.min()), int(xs2.max()), int(ys2.max())),
            "ink": ink,
            "char_h": float(np.median(stats[pieces, cv2.CC_STAT_HEIGHT])),
            "n_chars": int(len(pieces)),
        })

    return blobs


def find_line_boxes(lines_mask):
    """
    Turns the ruling-line mask into a list of boxes (one per long line).
    These end up as 'filler' (decorative borders).
    A real line is THIN, so anything fat is ignored.
    """
    h, w = lines_mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines_mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        if min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) > 0.03 * max(h, w):
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((int(x), int(y), int(x + bw), int(y + bh)))

    # one border often comes out as several pieces, so glue pieces that are close together
    return merge_close_boxes(boxes, gap=40)


def merge_close_boxes(boxes, gap):
    """
    Keeps merging any two boxes that are closer than `gap` pixels
    until nothing can be merged any more.
    """
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a = boxes[i]
                b = boxes[j]
                close_in_x = a[0] - gap <= b[2] and b[0] - gap <= a[2]
                close_in_y = a[1] - gap <= b[3] and b[1] - gap <= a[3]
                if close_in_x and close_in_y:
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes
