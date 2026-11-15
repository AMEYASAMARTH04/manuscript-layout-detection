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
