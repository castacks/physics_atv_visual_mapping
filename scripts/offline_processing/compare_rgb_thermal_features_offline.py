import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import yaml
import os
import subprocess
import cv2
import torch
from tqdm import tqdm

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.utils import normalize_dino

"""
Offline 2x2 visualization of RGB vs. thermal feature images.
    col 0: RGB     (top: raw image, bottom: featurized image)
    col 1: thermal (top: raw image, bottom: featurized image)

Feature images are visualized using the first 3 channels of the image pipeline output.

Example:
    python3 compare_rgb_thermal_features_offline.py --data_dir <run_dir> --rgb_dir rgb_image \
        --thermal_dir thermal_left_processed --mode video --out_fp rgb_thermal.mp4
"""

IMAGE_PROC_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'image_processing')

RGB_INTRINSICS = np.array([477.6049499511719, 0.0, 499.5, 0.0, 477.6049499511719, 252.0, 0.0, 0.0, 1.0])
THERMAL_INTRINSICS = np.array([347.95429323, 0.0, 323.05630493, 0.0, 347.95429323, 247.92877579, 0.0, 0.0, 1.0])

def get_image_proc_pipeline(img_proc_cfg_fp, device='cpu'):
    image_pipeline_config = yaml.safe_load(open(img_proc_cfg_fp, 'r'))
    config = {
        'image_processing': image_pipeline_config,
        'models_dir': os.environ['TARTANDRIVER_MODELS_DIR'],
        'device': device
    }
    pipeline = setup_image_pipeline(config)
    return pipeline

def load_image(img_fp):
    """
    Returns H x W x 3 float image in [0, 1] (RGB order)
    """
    img = cv2.imread(img_fp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img

def run_pipeline(pipeline, img, intrinsics):
    # B x C x H X W
    img_t = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)
    intrinsics_in = torch.from_numpy(intrinsics).reshape(1, 3, 3)

    # feat_img is 1 x dim x h x w
    with torch.no_grad():
        feat_img, _ = pipeline.run(img_t, intrinsics_in)
    return feat_img

def feat_img_to_rgb(feat_img):
    """
    1 x dim x h x w feature image -> h x w x 3 image in [0, 1] for viz
    """
    return normalize_dino(feat_img[0].permute(1, 2, 0)).cpu().numpy()

def get_frame_names(rgb_img_dir, thermal_img_dir):
    """
    Frames present in both the rgb and thermal directories
    """
    rgb_fps = set(f for f in os.listdir(rgb_img_dir) if f.endswith('.png'))
    thermal_fps = set(f for f in os.listdir(thermal_img_dir) if f.endswith('.png'))
    frame_names = sorted(rgb_fps & thermal_fps)
    if len(frame_names) < len(rgb_fps | thermal_fps):
        print(f"Warning: {len(rgb_fps)} rgb frames, {len(thermal_fps)} thermal frames, using {len(frame_names)} common frames")
    return frame_names

def open_video_writer(out_fp, w, h, fps):
    """
    Pipe raw BGR frames to ffmpeg and encode as H.264 mp4
    """
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
        # yuv420p requires even dims
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_fp
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

def render_frame(fig, axes, frame_name, rgb_img, rgb_feat_viz, thermal_img, thermal_feat_viz):
    """
    Draw 2x2 grid and return it as an H x W x 3 uint8 array
    """
    panels = [
        (axes[0][0], rgb_img, 'RGB'),
        (axes[1][0], rgb_feat_viz, 'RGB Features'),
        (axes[0][1], thermal_img, 'Thermal'),
        (axes[1][1], thermal_feat_viz, 'Thermal Features'),
    ]
    for ax, img, title in panels:
        ax.clear()
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(frame_name)

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='path to run directory containing image subdirectories')
    parser.add_argument('--rgb_dir', type=str, default='rgb_image', help='name of rgb image directory inside data_dir')
    parser.add_argument('--thermal_dir', type=str, default='thermal_left_processed', help='name of thermal image directory inside data_dir')
    parser.add_argument('--rgb_img_proc_cfg', type=str, default='loftup_vlad.yaml', help='image processing config for rgb (relative to config dir)')
    parser.add_argument('--thermal_img_proc_cfg', type=str, default='loftup_vlad.yaml', help='image processing config for thermal (relative to config dir)')
    parser.add_argument('--mode', type=str, choices=['frame', 'video'], default='frame', help='save a single frame or a video of all frames')
    parser.add_argument('--frame_idx', type=int, default=0, help='index into the sorted frame list (mode=frame)')
    parser.add_argument('--stride', type=int, default=1, help='use every n-th frame (mode=video)')
    parser.add_argument('--fps', type=float, default=10., help='video framerate (mode=video)')
    parser.add_argument('--out_fp', type=str, default=None, help='output path (default: rgb_thermal_feats.{png,mp4} in cwd)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    rgb_img_dir = os.path.join(args.data_dir, args.rgb_dir)
    thermal_img_dir = os.path.join(args.data_dir, args.thermal_dir)

    out_fp = args.out_fp
    if out_fp is None:
        out_fp = 'rgb_thermal_feats.png' if args.mode == 'frame' else 'rgb_thermal_feats.mp4'

    frame_names = get_frame_names(rgb_img_dir, thermal_img_dir)
    if len(frame_names) == 0:
        raise RuntimeError(f"No common .png frames in {rgb_img_dir} and {thermal_img_dir}")

    if args.mode == 'frame':
        frame_names = [frame_names[args.frame_idx]]
    else:
        frame_names = frame_names[::args.stride]

    rgb_img_proc_cfg_fp = os.path.join(IMAGE_PROC_CONFIG_DIR, args.rgb_img_proc_cfg)
    rgb_pipeline = get_image_proc_pipeline(rgb_img_proc_cfg_fp, args.device)

    thermal_img_proc_cfg_fp = os.path.join(IMAGE_PROC_CONFIG_DIR, args.thermal_img_proc_cfg)
    thermal_pipeline = get_image_proc_pipeline(thermal_img_proc_cfg_fp, args.device)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    video_writer = None

    for frame_name in tqdm(frame_names):
        rgb_img = load_image(os.path.join(rgb_img_dir, frame_name))
        thermal_img = load_image(os.path.join(thermal_img_dir, frame_name))

        rgb_feat_img = run_pipeline(rgb_pipeline, rgb_img, RGB_INTRINSICS)
        thermal_feat_img = run_pipeline(thermal_pipeline, thermal_img, THERMAL_INTRINSICS)

        rgb_feat_viz = feat_img_to_rgb(rgb_feat_img)
        thermal_feat_viz = feat_img_to_rgb(thermal_feat_img)

        out_frame = render_frame(fig, axes, frame_name, rgb_img, rgb_feat_viz, thermal_img, thermal_feat_viz)
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)

        if args.mode == 'frame':
            cv2.imwrite(out_fp, out_frame)
        else:
            if video_writer is None:
                h, w = out_frame.shape[:2]
                video_writer = open_video_writer(out_fp, w, h, args.fps)
            video_writer.stdin.write(out_frame.tobytes())

    plt.close(fig)
    if video_writer is not None:
        video_writer.stdin.close()
        video_writer.wait()

    print(f"Saved {len(frame_names)} frame(s) to {out_fp}")
