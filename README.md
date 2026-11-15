# manuscript-layout-detection
# Manuscript Layout Region Detection

Finds and labels layout regions on historical manuscript pages (palm-leaf and paper):

| label | meaning |
|---|---|
| `header` | text above the main body (running titles, folio numbers) |
| `footer` | text below the main body (catchwords, page numbers, signatures) |
| `main_text` | the main body of writing |
| `side_text` | marginalia / notes in the left or right margin |
| `filler` | decorations, borders, English / stamp text, anything that fits no other class |

For every page it saves an **annotated image** and a **JSON file** with bounding boxes, labels and confidence scores.

---

## 1. Setup

Needs Python 3.9 or newer.

```bash
# (optional but recommended) make a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
```

## 2. Run from the command line (batch)

```bash
python inference.py --input ./data/test_images --output ./results
```

- `--input` can be **a folder** or **a single image** (`--input ./data/test_images/sample_paper.png`)
- `--output` is created if it does not exist, and must be a different folder from the input folder
- Only relative paths are used, nothing is hard-coded

Supported formats: `.jpg .jpeg .png .tif .tiff .bmp`

## 3. Run in Jupyter Notebook (step by step)

```bash
jupyter notebook
```

Open `manuscript_layout_detection.ipynb` and choose *Kernel -> Restart & Run All*.
It shows the picture after every step of the pipeline.

## 4. Output

For every `page.png` in the input, the output folder gets:

- `page_annotated.png` - the page with coloured boxes, labels and confidence
- `page.json` - the predictions
- `all_predictions.json` - everything together (for batch runs)

```json
{
  "image": "sample_palm_leaf.png",
  "width": 2048,
  "height": 834,
  "page_box": [143, 83, 1778, 698],
  "regions": [
    {
      "label": "main_text",
      "bbox": [278, 93, 1763, 673],
      "confidence": 0.99,
      "reason": "one of the biggest blocks of writing on the page"
    }
  ]
}
```

`bbox` is `[x1, y1, x2, y2]` in **pixels of the original image** (top-left, bottom-right). Boxes are always clipped to the image, and the original input files are never modified.

## 5. Folder structure

```
manuscript_layout_detection/
├── inference.py                       # command line script
├── manuscript_layout_detection.ipynb  # step-by-step notebook
├── requirements.txt
├── README.md
├── CODE_EXPLAINED.md                  # every file explained
├── data/test_images/                  # sample pages (+ one synthetic test page)
├── results/                           # example output
├── scripts/
│   ├── make_synthetic_demo.py         # builds the synthetic test page
│   └── build_notebook.py              # rebuilds the notebook
└── src/
    ├── config.py                      # all tunable numbers
    ├── image_utils.py                 # reading / saving / box helpers
    ├── preprocessing.py               # lighting fix, page finding, ink mask, border lines
    ├── line_finder.py                 # clean ink, glue letters into blobs
    ├── classifier.py                  # rules that give each blob a label + confidence
    ├── visualize.py                   # draws the boxes
    └── pipeline.py                    # puts everything together
```

## 6. How it works and why (model rationale)

The assignment provides no labelled data, so training a detector (YOLO, Faster R-CNN, LayoutLM ...) was not possible honestly. I used **classical computer vision + layout rules**, which needs no training, is easy to explain and runs on a laptop CPU.

1. **Lighting fix** - divide the gray image by an estimate of the empty page (morphological closing + blur). Handles uneven light, shadows, stains and faded ink.
2. **Page finding** - Otsu + largest shape finds the leaf on its cloth background; if the page fills the image the whole image is used.
3. **Ink mask** - Otsu threshold on the lighting-corrected image (only page pixels choose the threshold).
4. **Border lines** - columns of pixels that are inked far more than normal are page borders. They are removed from the ink and reported as `filler`.
5. **Cleaning** - tiny dust, thin streaks and big solid blobs (string holes, stains) are removed.
6. **Stamp text split** - small letters in the top / bottom strip are separated from the body so a watermark can never stick to the handwriting.
7. **Blob forming** - the size of one character is estimated from the page itself, and neighbouring letters are glued with a dilation sized in "characters", so it adapts to big or small writing.
8. **Rules** (`classifier.py`):
   outside page / small centred stamp text -> `filler`; biggest blobs -> `main_text`; left or right of body -> `side_text`; above -> `header`; below -> `footer`; anything else -> `filler` with low confidence.

**Confidence** is a heuristic score (for example: more space between a header and the body = higher). It is **not** a calibrated probability.

### How the robustness items are handled

| Challenge | What the code does |
|---|---|
| faded ink / uneven lighting / stains | background division (`fix_lighting`), Otsu on page pixels, solid-blob removal |
| bleed-through | Otsu on the corrected image keeps only the strong ink |
| blur / noise | small Gaussian blur, dust removal |
| page damage, holes | page mask, solid-blob removal |
| palm-leaf vs paper, aspect ratios | page finding with fallback, all sizes relative to image / character size |
| multi-column | every big blob becomes its own `main_text` box |
| skew | blobs are glued region-wise so mild skew is fine; **there is no explicit deskew step** |

## 7. Tuning

All numbers are in `src/config.py` with comments. The most useful ones:

- `MERGE_WIDTH_IN_CHARS` / `MERGE_HEIGHT_IN_CHARS` - how far apart letters can be and still count as one block
- `MAIN_TEXT_INK_RATIO` - how much ink a blob needs (compared with the biggest) to be `main_text`
- `EDGE_STRIP_FRACTION` - how thick the top/bottom strip is where stamp text is searched
- `BORDER_MIN_COVERAGE` - how inked a column must be to count as a border

## 8. Known limitations

- Rule-based: it assumes one main block of writing with visible gaps around header, footer and margin notes. Notes closer to the body than about one letter height are merged into `main_text`.
- `main_text` boxes can include ink right next to the body (for example a ruling line stub or a flourish that crosses the border).
- Pencil text and decorations are only detected when they look like small stamp text or long lines.
- Evaluated only by eye on the two provided samples and one synthetic page. No quantitative metrics (mAP / IoU) because there is no ground truth.
- Next step if labelled pages become available: fine-tune a detector (for example YOLO) on the five classes and keep this pipeline as a baseline.

## 9. Sample images

`sample_palm_leaf.png` and `sample_paper.png` come from the assignment document. `synthetic_demo_page.png` is generated by `scripts/make_synthetic_demo.py` to exercise the header / footer / side-text rules, it is not real manuscript data.
