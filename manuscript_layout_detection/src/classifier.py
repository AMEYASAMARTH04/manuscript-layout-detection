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
