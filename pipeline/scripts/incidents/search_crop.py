"""Cut a window out of a picture at its own resolution (no scaling), keeping the legend strip on top.

    uv run python scripts/incidents/search_crop.py PICTURE X0 Y0 X1 Y1 OUT [--legend PX]"""
import argparse

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("picture")
ap.add_argument("box", type=int, nargs=4)
ap.add_argument("out")
ap.add_argument("--legend", type=int, default=0, help="the legend strip's height, kept on top")
args = ap.parse_args()
img = cv2.imread(args.picture)
x0, y0, x1, y1 = args.box
crop = img[y0:y1, x0:x1]
if args.legend:
    crop = cv2.vconcat([img[:args.legend, x0:x1], crop])
cv2.imwrite(args.out, crop)
print(args.out, crop.shape)
