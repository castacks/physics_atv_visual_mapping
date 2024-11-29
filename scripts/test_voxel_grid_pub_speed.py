import rclpy
from rclpy.node import Node

import torch
import numpy as np

from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelGrid

from std_msgs.msg import Float32MultiArray, UInt8MultiArray
from sensor_msgs.msg import PointCloud2, PointField

class PubTestNode(Node):
    """
    Sample script for trying a couple different permutations of publishing big voxel grids
    """
    def __init__(self, voxel_grid):
        super().__init__("pub_test_node")

        self.voxel_grid = voxel_grid
        self.pts = self.voxel_grid.grid_indices_to_pts(self.voxel_grid.raster_indices_to_grid_indices(self.voxel_grid.indices))
        self.feats = self.voxel_grid.features

        # self.timer = self.create_timer(0.5, self.pub_voxels_as_pc)
        # self.timer = self.create_timer(0.5, self.pub_voxels_as_float32)
        self.timer = self.create_timer(0.5, self.pub_voxels_as_uint8)

        self.pointcloud_pub = self.create_publisher(PointCloud2, "/debug_pc", 10)
        self.uint8_pub = self.create_publisher(UInt8MultiArray, "/debug_uint8", 10)
        self.float32_pub = self.create_publisher(Float32MultiArray, "/debug_float32", 10)

    def pub_voxels_as_float32(self):
        self.get_logger().info('publishing voxels')

        t1 = self.get_clock().now()

        pcdata_npy = torch.cat([self.pts, self.feats], dim=-1).cpu().numpy()
        torch.cuda.synchronize()

        t2 = self.get_clock().now()

        pcdata = pcdata_npy.flatten().tolist()

        t3 = self.get_clock().now()

        msg = Float32MultiArray(
            data=pcdata
        )

        t4 = self.get_clock().now()

        self.float32_pub.publish(msg)

        t5 = self.get_clock().now()

        timing_data = {
            'torch -> npy   ': t2-t1,
            'npy -> string  ': t3-t2,
            'add to f32 msg ': t4-t3,
            'pub f32 msg    ': t5-t4,
        }

        timing_str = ''.join(["\n{}: {:.4f}s".format(k,v.nanoseconds * 1e-9) for k,v in timing_data.items()])
        timing_str += '\n________\ntotal: {:.4f}s'.format(sum([x.nanoseconds*1e-9 for x in timing_data.values()]))
        self.get_logger().info(timing_str)


    def pub_voxels_as_uint8(self):
        self.get_logger().info('publishing voxels')

        t1 = self.get_clock().now()

        pcdata_npy = torch.cat([self.pts, self.feats], dim=-1).cpu().numpy()
        torch.cuda.synchronize()

        t2 = self.get_clock().now()

        self.get_logger().info("proc {} ({} floats)".format(pcdata_npy.shape, np.prod(pcdata_npy.shape)))
        pcdata = pcdata_npy.astype("f").tobytes()

        t3 = self.get_clock().now()

        msg = UInt8MultiArray(
            data=pcdata
        )

        t4 = self.get_clock().now()

        self.uint8_pub.publish(msg)

        t5 = self.get_clock().now()

        timing_data = {
            'torch -> npy   ': t2-t1,
            'npy -> string  ': t3-t2,
            'add to ui8 msg ': t4-t3,
            'pub ui8 msg    ': t5-t4,
        }

        timing_str = ''.join(["\n{}: {:.4f}s".format(k,v.nanoseconds * 1e-9) for k,v in timing_data.items()])
        timing_str += '\n________\ntotal: {:.4f}s'.format(sum([x.nanoseconds*1e-9 for x in timing_data.values()]))
        self.get_logger().info(timing_str)


    def pub_voxels_as_pc(self):
        self.get_logger().info('publishing voxels')

        #init pc msg
        t1 = self.get_clock().now()
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        for i in range(self.feats.shape[-1]):
            fields.append(PointField(name='feat_{}'.format(i), offset=8+4*i, datatype=PointField.FLOAT32, count=1))

        t2 = self.get_clock().now()

        pcdata_npy = torch.cat([self.pts, self.feats], dim=-1).cpu().numpy()
        torch.cuda.synchronize()

        t3 = self.get_clock().now()

        pcdata = pcdata_npy.astype("f").tobytes()

        t4 = self.get_clock().now()

        msg = PointCloud2(
            fields=fields,
            data=pcdata
        )

        t5 = self.get_clock().now()
        
        self.pointcloud_pub.publish(msg)

        t6 = self.get_clock().now()

        timing_data = {
            'init pc msg   ': t2-t1,
            'torch -> npy  ': t3-t2,
            'npy -> string ': t4-t3,
            'add to pc_msg ': t5-t4,
            'pub pc msg    ': t6-t5,
        }

        timing_str = ''.join(["\n{}: {:.4f}s".format(k,v.nanoseconds * 1e-9) for k,v in timing_data.items()])
        timing_str += '\n________\ntotal: {:.4f}s'.format(sum([x.nanoseconds*1e-9 for x in timing_data.values()]))
        self.get_logger().info(timing_str)

def main(args=None):
    rclpy.init(args=args)

    #test params
    N_VOXELS = 500000 #empirically, ~1.5mil voxels is an upper bound of what we see in practice
    N_FEATURES = 16
    DEVICE = 'cuda'

    metadata = LocalMapperMetadata(
        origin=[-100., -100., -50.],
        length=[200., 200., 100.],
        resolution=[0.25, 0.25, 0.25]
    )

    #empirically, fewer than 5% of voxels tend to be occupied
    max_n_voxels = torch.prod(metadata.N).item()
    assert N_VOXELS < max_n_voxels, "too many voxels for given metadata, make a bigger grid or fewer voxels"
    idxs = torch.randperm(n=max_n_voxels)[:N_VOXELS]
    features = torch.rand(N_VOXELS, N_FEATURES)

    print('making voxel grid with {} voxels, {} features ({} total floats)'.format(N_VOXELS, N_FEATURES, N_VOXELS*(N_FEATURES+3)))

    voxel_grid = VoxelGrid(n_features=N_FEATURES, metadata=metadata, device='cpu')
    voxel_grid.indices = idxs
    voxel_grid.features = features
    voxel_grid = voxel_grid.to(DEVICE)

    node = PubTestNode(voxel_grid=voxel_grid)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()