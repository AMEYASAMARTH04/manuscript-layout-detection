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
