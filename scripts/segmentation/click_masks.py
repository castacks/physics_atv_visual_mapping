#!/usr/bin/env python3
"""Paint raw images and write typed *.mask.png files."""

import argparse
from pathlib import Path
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np


def images(image_dir):
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)


def clean_label(label):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return label.strip("._-")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--jump", type=int, default=50)
    parser.add_argument(
        "--types",
        default="danger",
        help="Comma-separated starting labels. Press t in the UI to add/select more.",
    )
    args = parser.parse_args()

    files = images(args.image_dir)
    out_dir = Path(args.mask_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [clean_label(x) for x in args.types.split(",") if clean_label(x)]
    labels.extend(p.name for p in sorted(out_dir.iterdir()) if p.is_dir() and p.name not in labels)
    if not labels:
        labels = ["danger"]
    active_label = labels[0]

    idx = 0
    masks = {}
    img_artist = None
    overlay = None
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.7, 1.0],
            [0.2, 1.0, 0.2],
            [1.0, 0.8, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.7],
            [1.0, 0.45, 0.0],
            [0.7, 0.4, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    def out_path(label):
        return out_dir / label / f"{files[idx].stem}.mask.png"

    def load():
        nonlocal masks, img_artist, overlay
        img = cv2.cvtColor(cv2.imread(str(files[idx])), cv2.COLOR_BGR2RGB)
        masks = {label: np.zeros(img.shape[:2], dtype=np.uint8) for label in labels}
        loaded = False
        for label in labels:
            path = out_path(label)
            if path.exists():
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                masks[label] = np.where(mask > 0, 255, 0).astype(np.uint8)
                loaded = True
        flat_path = out_dir / f"{files[idx].stem}.mask.png"
        if flat_path.exists() and not loaded:
            mask = cv2.imread(str(flat_path), cv2.IMREAD_GRAYSCALE)
            masks[active_label] = np.where(mask > 0, 255, 0).astype(np.uint8)
        if img_artist is None:
            img_artist = ax.imshow(img)
            overlay = ax.imshow(np.zeros((*img.shape[:2], 4), dtype=float))
        else:
            img_artist.set_data(img)
            overlay.set_data(np.zeros((*img.shape[:2], 4), dtype=float))
        refresh()

    def save():
        for label, mask in masks.items():
            path = out_path(label)
            if mask.any():
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), mask)
            elif path.exists():
                path.unlink()

    def refresh():
        rgba = np.zeros((*next(iter(masks.values())).shape, 4), dtype=float)
        for label, mask in masks.items():
            color = colors[labels.index(label) % len(colors)]
            alpha = 0.45 if label == active_label else 0.25
            sel = mask > 0
            rgba[sel, :3] = color
            rgba[sel, 3] = alpha
        overlay.set_data(rgba)
        menu = " ".join(f"{i + 1}:{label}" for i, label in enumerate(labels[:9]))
        ax.set_title(
            f"{idx + 1}/{len(files)} {files[idx].name} | active={active_label} | "
            f"L paint, R erase, t type, n/p, [/], q save+quit | {menu}"
        )
        fig.canvas.draw_idle()

    def paint(event):
        if event.inaxes != ax or event.button not in (1, 3) or event.xdata is None:
            return
        value = 255 if event.button == 1 else 0
        cv2.circle(masks[active_label], (round(event.xdata), round(event.ydata)), args.radius, value, -1)
        refresh()

    def key(event):
        nonlocal idx, active_label
        if event.key in ("enter", "n"):
            save()
            idx = min(idx + 1, len(files) - 1)
            load()
        elif event.key == "p":
            save()
            idx = max(idx - 1, 0)
            load()
        elif event.key == "]":
            save()
            idx = min(idx + args.jump, len(files) - 1)
            load()
        elif event.key == "[":
            save()
            idx = max(idx - args.jump, 0)
            load()
        elif event.key == "t":
            save()
            label = clean_label(input("mask type: "))
            if label:
                if label not in labels:
                    labels.append(label)
                    masks[label] = np.zeros_like(next(iter(masks.values())))
                active_label = label
                refresh()
        elif event.key in tuple(str(i) for i in range(1, 10)):
            label_idx = int(event.key) - 1
            if label_idx < len(labels):
                active_label = labels[label_idx]
                refresh()
        elif event.key in ("backspace", "delete"):
            masks[active_label][:] = 0
            refresh()
        elif event.key == "q":
            save()
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", paint)
    fig.canvas.mpl_connect("motion_notify_event", paint)
    fig.canvas.mpl_connect("key_press_event", key)
    load()
    plt.show()


if __name__ == "__main__":
    main()
