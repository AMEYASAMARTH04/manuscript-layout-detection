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
