# Code explained, file by file
Read in this order: `config.py` -> `image_utils.py` -> `preprocessing.py` -> `line_finder.py` -> `classifier.py` -> `visualize.py` -> `pipeline.py` -> `inference.py`.
The data flows like this:

```
image -> shrink -> gray -> fix lighting -> find page -> ink mask -> remove borders
      -> clean ink -> guess letter size -> split stamp text -> glue into blobs
      -> classify blobs -> scale boxes back -> JSON + annotated image
```

---

## `src/config.py`

All the settings in one place: image width used while working, class names, box colours and every threshold the other files use. If detection looks wrong on a new manuscript, this is the first file to touch. Nothing runs here, other files just import the numbers.

```python
"""
config.py
---------
All the settings live here so I don't have to hunt for random numbers
inside the other files. If the detection looks off on some manuscript,
this is the first place to try changing things.
"""

# Big images are shrunk to this width while we work on them (faster).
# The final boxes are scaled back to the ORIGINAL image size, so nothing is lost.
WORK_WIDTH = 1600

# which file types we accept when a whole folder is given
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# the five classes from the assignment
CLASS_NAMES = ["header", "footer", "main_text", "side_text", "filler"]

# colours used when drawing boxes (OpenCV uses B, G, R order, not R, G, B)
CLASS_COLORS = {
    "header": (255, 120, 0),      # blue
    "footer": (0, 165, 255),      # orange
    "main_text": (0, 170, 0),     # green
    "side_text": (200, 0, 200),   # purple
    "filler": (0, 0, 220),        # red
}

# ---------- preprocessing numbers ----------

# size of the "remove the dark ink" step used to guess the paper brightness,
# written as a fraction of the image width
BACKGROUND_KERNEL_FRACTION = 0.02

# how much of the page edge we shave off, so shadows / edges are not seen as ink
PAGE_ERODE_FRACTION = 0.006

# a horizontal "ruling line" must be at least this long (fraction of image width)
RULING_MIN_LENGTH_FRACTION = 0.5

# a pixel column is a page BORDER if ink covers at least this fraction of the
# image height AND at least this many times more than a normal column of text
BORDER_MIN_COVERAGE = 0.3
BORDER_TIMES_NORMAL = 2.0

# ---------- blob / region finding numbers ----------

# when we glue letters together into blobs we use a box this big,
# measured in "character heights" (so it adapts to big or small writing)
MERGE_WIDTH_IN_CHARS = 1.0
MERGE_HEIGHT_IN_CHARS = 0.5

# blobs with less ink than this (compared to one character square) are dropped as noise
MIN_BLOB_INK_RATIO = 0.15

# ---------- classification numbers ----------

# a blob is "main text" if it has at least this much ink compared to the biggest blob
MAIN_TEXT_INK_RATIO = 0.25

# English / stamp text is smaller than the real writing: below this ratio we call it small
SMALL_TEXT_RATIO = 0.75

# the stamp text sits in the top / bottom strip of the image (fraction of height)
EDGE_STRIP_FRACTION = 0.08

```

---

## `src/image_utils.py`

Small helpers. `list_images` accepts a file or a folder, `read_image`/`save_image` use `imdecode`/`imencode` so Windows paths with special characters work, `resize_for_work` shrinks big scans to 1600px wide (and returns the scale so boxes can be scaled back), `clip_box` guarantees boxes stay inside the page, `scale_box` converts boxes back to original size.

```python
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

```

---

## `src/preprocessing.py`

Cleans the picture before looking for text. `fix_lighting` divides the image by an estimate of the empty page so shadows and stains cancel. `find_page` finds the palm leaf on its cloth background (or says 'the whole image is the page'). `make_ink_mask` makes a black/white picture with Otsu thresholding. `find_ruling_lines` finds page borders by counting how much of each pixel column is inked (borders are inked far more than text).

```python
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

```

---

## `src/line_finder.py`

Turns the ink mask into blobs. `clean_ink` removes dust, thin leftovers of lines and big solid blobs like string holes. `guess_char_height` estimates letter height from the biggest components so the code adapts to big/small writing. `split_edge_stamp_text` separates small stamp/watermark letters in the top and bottom strip. `find_text_blobs` dilates the ink so nearby letters merge, then measures each blob (box, ink amount, letter height). `find_line_boxes` + `merge_close_boxes` turn the border-line mask into one box per border.

```python
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

```

---

## `src/classifier.py`

The decision-maker. Rules in order: outside the page or small centred stamp text -> filler; biggest blobs -> main_text; blob in left/right margin -> side_text; above main text -> header; below -> footer; otherwise filler with low confidence. Confidence is a simple score from how clearly the rule fits (for example the size of the gap), not a real probability.

```python
"""
classifier.py
-------------
This is the "brain" of the project. It looks at every blob and decides
if it is  main_text / header / footer / side_text / filler.

There is NO trained model here (the assignment gives no labelled data), so the
decisions come from simple rules about WHERE a blob is compared to the
biggest block of writing on the page. The rules, in order:

    1. blob is outside the page, or is small centred English/stamp text at the
       very top/bottom of the image          -> filler
    2. biggest blobs of ink                  -> main_text
    3. blob is sitting inside the main text  -> ignored (it is part of the body)
    4. blob is left/right of the main text   -> side_text
    5. blob is above the main text           -> header
    6. blob is below the main text           -> footer
    7. anything else                         -> filler (low confidence)

The confidence number is NOT a probability from a neural network. It is a
score I made up from "how clearly does the rule fit" (for example a bigger gap
between header and body = more sure). It is only good for comparing regions
with each other.
"""

from src.config import (
    MAIN_TEXT_INK_RATIO,
    SMALL_TEXT_RATIO,
    EDGE_STRIP_FRACTION,
)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def make_region(label, box, confidence, reason):
    return {
        "label": label,
        "box": box,
        "confidence": round(float(clamp(confidence, 0.05, 0.99)), 3),
        "reason": reason,
    }


def overlap_length(a1, a2, b1, b2):
    """How much do the ranges [a1, a2] and [b1, b2] overlap? (0 if they don't)"""
    return max(0, min(a2, b2) - max(a1, b1))


def classify_blobs(blobs, line_boxes, page_mask, img_w, img_h, char_h):
    regions = []

    # ------------------------------------------------------------
    # Rule 1: filler by position
    # ------------------------------------------------------------
    candidates = []
    for b in blobs:
        x1, y1, x2, y2 = b["box"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        width = x2 - x1

        outside_page = page_mask[cy, cx] == 0

        in_edge_strip = (cy < img_h * EDGE_STRIP_FRACTION) or (cy > img_h * (1 - EDGE_STRIP_FRACTION))
        is_small = b["char_h"] < SMALL_TEXT_RATIO * char_h
        is_wide = width > img_w * 0.25
        is_centred = abs(cx - img_w / 2) < img_w * 0.2

        if outside_page:
            regions.append(make_region("filler", b["box"], 0.90,
                                       "text lying outside the page (caption / stamp)"))
        elif in_edge_strip and is_small and is_wide and is_centred:
            regions.append(make_region("filler", b["box"], 0.85,
                                       "small centred text in the top/bottom strip (looks like English stamp text)"))
        else:
            candidates.append(b)

    # ------------------------------------------------------------
    # Rule 2: main text = the blobs with the most ink
    # ------------------------------------------------------------
    if len(candidates) == 0:
        return add_line_regions(regions, line_boxes)

    biggest_ink = max(b["ink"] for b in candidates)
    main_blobs = [b for b in candidates if b["ink"] >= MAIN_TEXT_INK_RATIO * biggest_ink]
    other_blobs = [b for b in candidates if b["ink"] < MAIN_TEXT_INK_RATIO * biggest_ink]

    for b in main_blobs:
        confidence = 0.5 + 0.5 * (b["ink"] / float(biggest_ink))
        regions.append(make_region("main_text", b["box"], confidence,
                                   "one of the biggest blocks of writing on the page"))

    # the box around ALL the main text together, we compare everything else to it
    mx1 = min(b["box"][0] for b in main_blobs)
    my1 = min(b["box"][1] for b in main_blobs)
    mx2 = max(b["box"][2] for b in main_blobs)
    my2 = max(b["box"][3] for b in main_blobs)

    # ------------------------------------------------------------
    # Rules 3 to 7: look at the small leftover blobs one by one
    # ------------------------------------------------------------
    for b in other_blobs:
        x1, y1, x2, y2 = b["box"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        blob_w = max(1, x2 - x1)
        blob_h = max(1, y2 - y1)

        # Rule 3: already inside the main text area -> skip it
        inside_w = overlap_length(x1, x2, mx1, mx2)
        inside_h = overlap_length(y1, y2, my1, my2)
        if inside_w / blob_w > 0.8 and inside_h / blob_h > 0.8:
            continue

        shares_height = inside_h / blob_h > 0.5       # sits next to the body, not above/below it
        shares_width = inside_w / blob_w > 0.2        # sits over/under the body

        # Rule 4: left or right of the main text -> side_text
        if (cx < mx1 or cx > mx2) and (shares_height or not shares_width):
            if cx < mx1:
                gap = mx1 - x2
            else:
                gap = x1 - mx2
            confidence = 0.6 + 0.35 * clamp(gap / (2.0 * char_h))
            regions.append(make_region("side_text", b["box"], confidence,
                                       "sits in the left/right margin next to the main text"))
            continue

        # Rule 5: above the main text -> header
        if cy < my1:
            gap = my1 - y2
            confidence = 0.6 + 0.35 * clamp(gap / (2.0 * char_h))
            regions.append(make_region("header", b["box"], confidence,
                                       "sits above the main text with a gap"))
            continue

        # Rule 6: below the main text -> footer
        if cy > my2:
            gap = y1 - my2
            confidence = 0.6 + 0.35 * clamp(gap / (2.0 * char_h))
            regions.append(make_region("footer", b["box"], confidence,
                                       "sits below the main text with a gap"))
            continue

        # Rule 7: we don't know what this is
        regions.append(make_region("filler", b["box"], 0.5,
                                   "does not fit any other rule (decoration / unclear mark)"))

    return add_line_regions(regions, line_boxes)


def add_line_regions(regions, line_boxes):
    """Long border / ruling lines are decorative, so they become 'filler' too."""
    for box in line_boxes:
        regions.append(make_region("filler", box, 0.7, "long straight border / ruling line"))
    return regions

```

---

## `src/visualize.py`

Draws coloured, semi-transparent boxes with 'label confidence' text on a COPY of the image, scaling line width and font with image size.

```python
"""
visualize.py
------------
Draws the predicted boxes on a COPY of the image (the original is never touched).
"""

import cv2

from src.config import CLASS_COLORS


def draw_regions(img, regions):
    """
    img     : the original colour image
    regions : list of dicts with "label", "bbox", "confidence"
    returns : a new image with coloured boxes + labels
    """
    out = img.copy()
    h, w = out.shape[:2]

    # scale line thickness and text with the image size so it is readable on big scans
    thickness = max(2, int(w / 600))
    font_scale = max(0.5, w / 1800.0)

    # first a light colour fill, so you can see the areas
    overlay = out.copy()
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        color = CLASS_COLORS[r["label"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)

    # then the outlines and the text labels
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        color = CLASS_COLORS[r["label"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        text = "%s %.2f" % (r["label"], r["confidence"])
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # put the label just above the box, or inside it if there is no room above
        label_y = y1 - 4 if y1 - th - 8 > 0 else y1 + th + 6
        cv2.rectangle(out, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 4), color, -1)
        cv2.putText(out, text, (x1 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return out

```

---

## `src/pipeline.py`

Glues all steps together. `analyze_image` runs the whole detection on one loaded image and returns regions in original-image pixels (plus the in-between pictures for the notebook). `run_on_path` handles a file or folder, refuses to write into the input folder, saves the annotated PNG and JSON per page and an `all_predictions.json`.

```python
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

```

---

## `inference.py`

The command line entry point from the assignment: `python inference.py --input ./data/test_images --output ./results`. It only reads the two arguments and calls `run_on_path`.

```python
"""
inference.py
------------
Command line script for batch processing.

Example:
    python inference.py --input ./data/test_images --output ./results

--input can be a single image or a folder of images.
"""

import argparse

from src.pipeline import run_on_path


def main():
    parser = argparse.ArgumentParser(
        description="Detect header / footer / main_text / side_text / filler regions in manuscript images.")
    parser.add_argument("--input", required=True, help="an image file or a folder of images")
    parser.add_argument("--output", required=True, help="folder where results will be saved")
    args = parser.parse_args()

    run_on_path(args.input, args.output)


if __name__ == "__main__":
    main()

```

---

## `scripts/make_synthetic_demo.py`

Builds the synthetic test page (real handwriting pasted on a big sheet with a header, footer and margin notes added with a normal font) so the header / footer / side_text rules can be tested.

```python
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

```
