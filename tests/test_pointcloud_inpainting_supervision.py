import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml

from physics_atv_visual_mapping.localmapping.hindsight import compute_overlaps
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from ros_torch_converter.datatypes.pointcloud import PointCloudTorch


SCRIPT = Path(__file__).parents[1] / "scripts/offline_processing/get_pointcloud_inpainting_supervision.py"
SPEC = importlib.util.spec_from_file_location("pointcloud_inpainting", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metadata(origin, length=2.0):
    return LocalMapperMetadata(
        origin=[origin, -1.0, -1.0],
        length=[length, 2.0, 2.0],
        resolution=[1.0, 1.0, 1.0],
    )


def write_run(tmp_path, metadatas, frames):
    pointcloud_dir = tmp_path / "pointcloud"
    metadata_dir = tmp_path / "metadata"
    pointcloud_dir.mkdir()
    metadata_dir.mkdir()
    for index, (volume, points) in enumerate(zip(metadatas, frames)):
        (metadata_dir / f"{index:08d}_metadata.yaml").write_text(
            yaml.safe_dump(
                {
                    "origin": volume.origin.tolist(),
                    "length": volume.length.tolist(),
                    "resolution": volume.resolution.tolist(),
                }
            )
        )
        pointcloud = PointCloudTorch.from_numpy(np.asarray(points, dtype=np.float32))
        pointcloud.stamp = 100.0 + index
        pointcloud.frame_id = "sensor_init"
        pointcloud.to_kitti(pointcloud_dir, index)
    return pointcloud_dir, metadata_dir


def output_points(output_dir, index):
    return PointCloudTorch.from_kitti(output_dir, index)


def test_contiguous_overlap_stops_at_first_gap_and_current_is_included(tmp_path):
    volumes = [metadata(0.0), metadata(0.5), metadata(4.0), metadata(0.0)]
    assert compute_overlaps(volumes).tolist() == [1, 1, 2, 3]
    pointcloud_dir, metadata_dir = write_run(
        tmp_path,
        volumes,
        [
            [[0.2, 0.0, 0.0]],
            [[0.25, 0.0, 0.0], [1.7, 0.0, 0.0], [2.2, 0.0, 0.0]],
            [[4.2, 0.0, 0.0]],
            [[0.4, 0.0, 0.0]],
        ],
    )
    output_dir = tmp_path / "output"
    MODULE.generate_hindsight(pointcloud_dir, metadata_dir, output_dir)

    first = output_points(output_dir, 0)
    assert torch.allclose(first.pts, torch.tensor([[0.2, 0.0, 0.0], [1.7, 0.0, 0.0]]))
    assert first.stamp == 100.0
    assert first.frame_id == "sensor_init"
    final = output_points(output_dir, 3)
    assert torch.allclose(final.pts, torch.tensor([[0.4, 0.0, 0.0]]))


def test_history_is_permanently_evicted_when_rolling_volume_moves(tmp_path):
    volumes = [metadata(-1.0), metadata(3.0), metadata(-1.0)]
    pointcloud_dir, metadata_dir = write_run(
        tmp_path,
        volumes,
        [[[0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]],
    )
    output_dir = tmp_path / "output"
    assert MODULE.generate_hindsight(pointcloud_dir, metadata_dir, output_dir) == [0, 1, 2]
    last = output_points(output_dir, 2)
    assert torch.allclose(last.pts, torch.tensor([[0.5, 0.0, 0.0]]))


def test_nonzero_target_start_still_processes_history_from_zero(tmp_path):
    volumes = [metadata(0.0), metadata(0.0), metadata(0.0)]
    pointcloud_dir, metadata_dir = write_run(
        tmp_path,
        volumes,
        [[[0.1, 0.0, 0.0]], [[0.2, 0.0, 0.0]], [[0.3, 0.0, 0.0]]],
    )
    output_dir = tmp_path / "output"
    assert MODULE.required_source_end(metadata_dir, 2, 1) == 2
    assert MODULE.generate_hindsight(pointcloud_dir, metadata_dir, output_dir, 2, 1) == [2]
    selected = output_points(output_dir, 0)
    assert torch.allclose(
        selected.pts,
        torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
    )
    assert selected.stamp == 102.0
    assert np.loadtxt(output_dir / "timestamps.txt").reshape(-1).tolist() == [102.0]
    assert (output_dir / "source_target_indices.txt").read_text() == "2\n"
