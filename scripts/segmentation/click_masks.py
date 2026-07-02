#!/usr/bin/env python3
"""Paint raw images and write matching *.mask.png files."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def images(image_dir):
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--radius", type=int, default=12)
    args = parser.parse_args()

    files = images(args.image_dir)
    out_dir = Path(args.mask_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    mask = None
    img_artist = None
    overlay = None
    fig, ax = plt.subplots(figsize=(12, 8))

    def out_path():
        return out_dir / f"{files[idx].stem}.mask.png"

    def load():
        nonlocal mask, img_artist, overlay
        img = cv2.cvtColor(cv2.imread(str(files[idx])), cv2.COLOR_BGR2RGB)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if img_artist is None:
            img_artist = ax.imshow(img)
            overlay = ax.imshow(np.zeros((*mask.shape, 4), dtype=float))
        else:
            img_artist.set_data(img)
            overlay.set_data(np.zeros((*mask.shape, 4), dtype=float))
        ax.set_title(f"{idx + 1}/{len(files)} {files[idx].name} | drag paint, n/enter next, p prev, q quit")
        fig.canvas.draw_idle()

    def save():
        path = out_path()
        if mask.any():
            cv2.imwrite(str(path), mask)
        elif path.exists():
            path.unlink()

    def refresh():
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[..., 0] = 1.0
        rgba[..., 3] = (mask > 0) * 0.4
        overlay.set_data(rgba)
        fig.canvas.draw_idle()

    def paint(event):
        if event.inaxes != ax or event.button != 1 or event.xdata is None:
            return
        cv2.circle(mask, (round(event.xdata), round(event.ydata)), args.radius, 255, -1)
        refresh()

    def key(event):
        nonlocal idx
        if event.key in ("enter", "n"):
            save()
            idx = min(idx + 1, len(files) - 1)
            load()
        elif event.key == "p":
            save()
            idx = max(idx - 1, 0)
            load()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", paint)
    fig.canvas.mpl_connect("motion_notify_event", paint)
    fig.canvas.mpl_connect("key_press_event", key)
    load()
    plt.show()


if __name__ == "__main__":
    main()
