import rclpy
from rclpy.node import Node

import torch
import numpy as np

from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.localmapping.bev.bev_localmapper import BEVGrid

from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, UInt8MultiArray

class PubTestNode(Node):
    """
    Sample script for trying a couple different permutations of publishing big voxel grids
    """
    def __init__(self, bev_grid):
        super().__init__("pub_test_node")

        self.bev_grid = bev_grid
        self.n_feats = len(self.bev_grid.feature_keys)

        self.timer = self.create_timer(0.5, self.pub_gridmap)

        self.gridmap_pub = self.create_publisher(GridMap, "/debug_gridmap", 10)
        self.rawdata_pub = self.create_publisher(UInt8MultiArray, "/debug_rawdata", 10)

    def pub_gridmap(self):
        self.get_logger().info('publishing gridmap')

        t1 = self.get_clock().now()

        msg = GridMap()
        msg.layers = self.bev_grid.feature_keys

        for i in range(self.n_feats):
            msg.data.append(Float32MultiArray())

        t2 = self.get_clock().now()

        data_reshape = torch.flip(self.bev_grid.data, dims=(0, 1)).permute(1,0,2)
        data_reshape = data_reshape.reshape(-1, self.n_feats).T
        torch.cuda.synchronize()

        t3 = self.get_clock().now()

        data_npy = data_reshape.cpu().numpy()

        t4 = self.get_clock().now()

        for i in range(self.n_feats):
            listdata = data_npy[i].tolist()
            # msg.data[i].data = listdata
            msg.data[i] = Float32MultiArray(
                data = listdata
            )

        t5 = self.get_clock().now()

        self.gridmap_pub.publish(msg)

        t6 = self.get_clock().now()

        rawdata_npy = self.bev_grid.data.cpu().numpy()
        self.get_logger().info("proc {} ({} floats)".format(rawdata_npy.shape, np.prod(rawdata_npy.shape)))
        rawdata = rawdata_npy.astype("f").tobytes()
        msg2 = UInt8MultiArray(
            data = rawdata
        )

        t7 = self.get_clock().now()

        self.rawdata_pub.publish(msg2)

        t8 = self.get_clock().now()

        timing_data = {
            'init gridmap   ': t2-t1,
            'reshape gpu    ': t3-t2,
            'gpu -> cpu     ': t4-t3,
            'add layer data ': t5-t4,
            'publish msg    ': t6-t5,
            'make uint8?    ': t7-t6,
            'pub  uint8?    ': t8-t7,
        }

        timing_str = ''.join(["\n{}: {:.4f}s".format(k,v.nanoseconds * 1e-9) for k,v in timing_data.items()])
        timing_str += '\n________\ntotal: {:.4f}s'.format(sum([x.nanoseconds*1e-9 for x in timing_data.values()]))
        self.get_logger().info(timing_str)

def main(args=None):
    rclpy.init(args=args)

    #test params
    # N_CHANNELS = 16 + 12 #we have ~12 geometric topics + N visual topics
    N_CHANNELS = 20 #we have ~12 geometric topics + N visual topics
    DEVICE = 'cuda'

    metadata = LocalMapperMetadata(
        origin=[-100., -100.],
        length=[200., 200.],
        resolution=[0.25, 0.25]
    )

    data = torch.rand(*metadata.N, N_CHANNELS)
    fks = ["feat_{}".format(i) for i in range(N_CHANNELS)]

    bev_grid = BEVGrid(n_features=N_CHANNELS, feature_keys=fks, metadata=metadata, device='cpu')
    bev_grid.data = data
    bev_grid = bev_grid.to(DEVICE)

    node = PubTestNode(bev_grid=bev_grid)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()