#!/usr/bin/env python3
"""Build hindsight point-cloud supervision from per-frame point clouds."""

import argparse
from pathlib import Path

import torch
import yaml

from physics_atv_visual_mapping.localmapping.hindsight import compute_overlaps
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from ros_torch_converter.datatypes.pointcloud import PointCloudTorch


def load_metadatas(metadata_dir):
    paths = sorted(Path(metadata_dir).glob("*_metadata.yaml"))
    if not paths:
        raise ValueError(f"no voxel metadata in {metadata_dir}")
    return [
        LocalMapperMetadata(
            origin=torch.tensor(metadata["origin"], dtype=torch.float32),
            length=torch.tensor(metadata["length"], dtype=torch.float32),
            resolution=torch.tensor(metadata["resolution"], dtype=torch.float32),
        )
        for path in paths
        for metadata in [yaml.safe_load(path.read_text())]
    ]


def aabb_mask(points, metadata):
    lower = metadata.origin.to(points.device)
    upper = (metadata.origin + metadata.length).to(points.device)
    return ((points >= lower) & (points < upper)).all(dim=-1)


def rolling_metadata(metadata, length):
    center = metadata.origin + metadata.length / 2.0
    base_origin = -1.5 * length
    return LocalMapperMetadata(
        origin=torch.round((center + base_origin) / metadata.resolution) * metadata.resolution,
        length=3.0 * length,
        resolution=metadata.resolution,
    )


def resolve_target_end(total, target_start, target_count):
    if target_start < 0 or target_start >= total:
        raise ValueError(f"target_start must be in [0, {total})")
    end = total if target_count is None else target_start + target_count
    if target_count is not None and target_count < 1:
        raise ValueError("target_count must be positive")
    if end > total:
        raise ValueError(f"target range [{target_start}, {end}) exceeds {total} frames")
    return end


def required_source_end(metadata_dir, target_start=0, target_count=None):
    metadatas = load_metadatas(metadata_dir)
    end = resolve_target_end(len(metadatas), target_start, target_count)
    return max(compute_overlaps(metadatas)[target_start:end].tolist())


def generate_hindsight(pointcloud_dir, metadata_dir, output_dir, target_start=0, target_count=None):
    pointcloud_dir = Path(pointcloud_dir)
    output_dir = Path(output_dir)
    metadatas = load_metadatas(metadata_dir)
    total = len(metadatas)
    target_end = resolve_target_end(total, target_start, target_count)
    length = metadatas[0].length
    if any(not torch.allclose(metadata.length, length) for metadata in metadatas):
        raise ValueError("all metadata volumes must have the same length")
    resolution = metadatas[0].resolution
    if any(not torch.allclose(metadata.resolution, resolution) for metadata in metadatas):
        raise ValueError("all metadata volumes must have the same resolution")

    overlaps = compute_overlaps(metadatas)
    endpoints = overlaps.tolist()
    source_end = max(endpoints[target_start:target_end])
    output_dir.mkdir(parents=True, exist_ok=True)

    active_points = torch.empty((0, 3), dtype=torch.float32)
    active_colors = torch.empty((0, 3), dtype=torch.float32)
    colors_valid = True
    target_headers = {}
    saved = []

    for source_index in range(source_end + 1):
        source = PointCloudTorch.from_kitti(pointcloud_dir, source_index)
        target_headers[source_index] = (source.stamp, source.frame_id)

        rolling_mask = aabb_mask(active_points, rolling_metadata(metadatas[source_index], length))
        active_points = active_points[rolling_mask]
        if colors_valid:
            active_colors = active_colors[rolling_mask]

        source_mask = aabb_mask(source.pts, metadatas[source_index])
        source_points = source.pts[source_mask]
        has_colors = len(source.colors) == len(source.pts)
        if colors_valid and source_points.numel() and not has_colors:
            colors_valid = False
            active_colors = torch.empty((0, 3), dtype=torch.float32)
        active_points = torch.cat([active_points, source_points], dim=0)
        if colors_valid:
            active_colors = torch.cat([active_colors, source.colors[source_mask]], dim=0)

        for target_index in range(target_start, target_end):
            if endpoints[target_index] != source_index:
                continue
            target_mask = aabb_mask(active_points, metadatas[target_index])
            colors = active_colors[target_mask] if colors_valid else None
            output = PointCloudTorch.from_torch(active_points[target_mask], colors)
            output.stamp, output.frame_id = target_headers[target_index]
            output.to_kitti(output_dir, target_index - target_start)
            saved.append(target_index)

    expected = list(range(target_start, target_end))
    if sorted(saved) != expected:
        raise RuntimeError(f"saved targets {saved}, expected [{target_start}, {target_end})")
    (output_dir / "source_target_indices.txt").write_text(
        "\n".join(str(target_index) for target_index in expected) + "\n"
    )
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointcloud-dir", type=Path)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--print-source-end", action="store_true")
    args = parser.parse_args()
    if args.print_source_end:
        print(required_source_end(args.metadata_dir, args.target_start, args.target_count))
        return
    if args.pointcloud_dir is None or args.output_dir is None:
        parser.error("--pointcloud-dir and --output-dir are required unless --print-source-end is used")
    saved = generate_hindsight(
        args.pointcloud_dir,
        args.metadata_dir,
        args.output_dir,
        args.target_start,
        args.target_count,
    )
    print(f"wrote {len(saved)} hindsight point clouds to {args.output_dir}")


if __name__ == "__main__":
    main()
