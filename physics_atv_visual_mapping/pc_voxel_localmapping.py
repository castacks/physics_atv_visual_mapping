import rclpy
from rclpy.node import Node
import yaml
import copy
import numpy as np
import time 
import cv2

np.float = np.float64  # hack for numpify

import tf2_ros
import torch
import cv_bridge
import os

from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
# from perception_interfaces.msg import FeatureVoxelGrid

# from physics_atv_visual_mapping.image_processing.image_pipeline import (
#     setup_image_pipeline,
# )
# from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.terrain_estimation.terrain_estimation_pipeline import setup_terrain_estimation_pipeline

from physics_atv_visual_mapping.localmapping.bev.bev_localmapper import BEVLocalMapper
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import (
    VoxelLocalMapper,
)
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata

from physics_atv_visual_mapping.utils import *


class PCVoxelMappingNode(Node):
    def __init__(self):
        super().__init__("visual_mapping")

        self.declare_parameter("config_fp", "")

        config_fp = self.get_parameter("config_fp").get_parameter_value().string_value
        config = yaml.safe_load(open(config_fp, "r"))

        self.vehicle_frame = config["vehicle_frame"]

        self.localmap = None
        self.pcl_msg = None
        self.odom_msg = None
        self.img_msg = None
        self.odom_frame = None

        self.device = config["device"]
        self.base_metadata = config["localmapping"]["metadata"]
        self.localmap_ema = config["localmapping"]["ema"]
        self.layer_key = (
            config["localmapping"]["layer_key"]
            if "layer_key" in config["localmapping"].keys()
            else None
        )
        self.layer_keys = (
            self.make_layer_keys(config["localmapping"]["layer_keys"])
            if "layer_keys" in config["localmapping"].keys()
            else None
        )
        self.last_update_time = 0.0

        self.setup_localmapper(config)

        self.do_terrain_estimation = "terrain_estimation" in config.keys()
        if self.do_terrain_estimation:
            self.get_logger().info("doing terrain estimation")
            self.terrain_estimator = setup_terrain_estimation_pipeline(config)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pcl_sub = self.create_subscription(
            PointCloud2, config["pointcloud"]["topic"], self.handle_pointcloud, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, config["odometry"]["topic"], self.handle_odom, 10
        )

        self.voxel_viz_pub = self.create_publisher(
            PointCloud2, "/voxels_viz", 1
        )

        if self.do_terrain_estimation:
            self.gridmap_pub = self.create_publisher(GridMap, "/dino_gridmap", 10)

        self.timing_pub = self.create_publisher(Float32, "/dino_proc_time", 10)

        self.timer = self.create_timer(0.2, self.spin)
        self.viz = config["viz"]

        # for debug
        self.tempind = 0

    def setup_localmapper(self, config):
        """
        check that the localmapper metadata is good
        """
        self.mapper_type = config["localmapping"]["mapper_type"]
        metadata = LocalMapperMetadata(**self.base_metadata)

        assert self.mapper_type == 'voxel', "need mapper type to be either 'voxel'"
        assert metadata.ndims == 3, "need 3d metadata for voxel mapping"
        self.localmapper = VoxelLocalMapper(
            metadata,
            n_features=0,
            ema=config["localmapping"]["ema"],
            device=config["device"],
        )

    def make_layer_keys(self, layer_keys):
        out = []
        for lk in layer_keys:
            for i in range(lk["n"]):
                out.append("{}_{}".format(lk["key"], i))
        return out

    def handle_pointcloud(self, msg):
        self.pcl_msg = msg

    def handle_odom(self, msg):
        if self.odom_frame is None:
            self.odom_frame = msg.header.frame_id
        self.odom_msg = msg

    def preprocess_inputs(self):
        if self.pcl_msg is None:
            self.get_logger().warn("no pcl msg received")
            return None

        pcl_time = (
            self.pcl_msg.header.stamp.sec + self.pcl_msg.header.stamp.nanosec * 1e-9
        )
        if abs(pcl_time - self.last_update_time) < 1e-3:
            return None

        if self.odom_msg is None:
            self.get_logger().warn("no odom msg received")
            return None

        if not self.tf_buffer.can_transform(
            self.odom_frame,
            self.pcl_msg.header.frame_id,
            rclpy.time.Time.from_msg(self.pcl_msg.header.stamp),
        ):
            self.get_logger().warn(
                "cant tf from {} to {} at {}".format(
                    self.odom_frame,
                    self.pcl_msg.header.frame_id,
                    self.pcl_msg.header.stamp,
                )
            )
            return None

        tf_odom_to_pcl_msg = self.tf_buffer.lookup_transform(
            self.odom_frame,
            self.pcl_msg.header.frame_id,
            rclpy.time.Time.from_msg(self.pcl_msg.header.stamp),
        )
        
        pcl = pcl_msg_to_xyz(self.pcl_msg).to(self.device)

        odom_to_pcl_htm = tf_msg_to_htm(tf_odom_to_pcl_msg).to(self.device)
        pcl_in_odom = transform_points(pcl.clone(), odom_to_pcl_htm)

        self.last_update_time = pcl_time

        pos = torch.tensor(
            [
                self.odom_msg.pose.pose.position.x,
                self.odom_msg.pose.pose.position.y,
                self.odom_msg.pose.pose.position.z,
            ],
            device=self.device,
        )

        return {
            "pos": pos,
            "pcl": pcl_in_odom,
        }

    # def make_voxel_msg(self, voxel_grid):
    #     msg = FeatureVoxelGrid()
    #     msg.header.stamp = self.pcl_msg.header.stamp
    #     msg.header.frame_id = self.odom_frame

    #     msg.metadata.origin.x = voxel_grid.metadata.origin[0].item()
    #     msg.metadata.origin.y = voxel_grid.metadata.origin[1].item()
    #     msg.metadata.origin.z = voxel_grid.metadata.origin[2].item()

    #     msg.metadata.length.x = voxel_grid.metadata.length[0].item()
    #     msg.metadata.length.y = voxel_grid.metadata.length[1].item()
    #     msg.metadata.length.z = voxel_grid.metadata.length[2].item()

    #     msg.metadata.resolution.x = voxel_grid.metadata.resolution[0].item()
    #     msg.metadata.resolution.y = voxel_grid.metadata.resolution[1].item()
    #     msg.metadata.resolution.z = voxel_grid.metadata.resolution[2].item()

    #     msg.num_voxels = voxel_grid.features.shape[0]
    #     msg.num_features = voxel_grid.features.shape[1]

    #     if self.layer_keys is None:
    #         msg.feature_keys = [
    #             "{}_{}".format(self.layer_key, i) for i in range(voxel_grid.features.shape[1])
    #         ]
    #     else:
    #         msg.feature_keys = copy.deepcopy(self.layer_keys)

    #     msg.indices = voxel_grid.indices.tolist()

    #     feature_msg = Float32MultiArray()
    #     feature_msg.layout.dim.append(
    #         MultiArrayDimension(
    #             label="column_index",
    #             size=voxel_grid.features.shape[0],
    #             stride=voxel_grid.features.shape[0],
    #         )
    #     )
    #     feature_msg.layout.dim.append(
    #         MultiArrayDimension(
    #             label="row_index",
    #             size=voxel_grid.features.shape[0],
    #             stride=voxel_grid.features.shape[0] * voxel_grid.features.shape[1],
    #         )
    #     )

    #     feature_msg.data = voxel_grid.features.flatten().tolist()

    #     return msg

    def make_gridmap_msg(self, bev_grid):
        """
        convert dino into gridmap msg

        Publish all the feature channels, plus a visualization and elevation layer

        Note that we assume all the requisite stuff is available (pcl, img, odom) as this
        should only be called after a dino map is successfully produced
        """
        gridmap_msg = GridMap()

        gridmap_data = bev_grid.data.cpu().numpy()
        np.save(str(self.tempind).zfill(6) + '.npy',gridmap_data)
        self.tempind += 1

        # setup metadata
        gridmap_msg.header.stamp = self.pcl_msg.header.stamp #self.img_msg.header.stamp # TODO: Wenshan
        gridmap_msg.header.frame_id = self.odom_frame

        gridmap_msg.layers = bev_grid.feature_keys

        #temp hack
        gridmap_msg.basic_layers = ["min_elevation_filtered_inflated_mask"]
        mask_idx = gridmap_msg.layers.index("min_elevation_filtered_inflated_mask")
        mask = gridmap_data[..., mask_idx] > 0.1
        gridmap_data[..., mask_idx][~mask] = float('nan')

        gridmap_msg.info.resolution = self.localmapper.metadata.resolution[0].item()
        gridmap_msg.info.length_x = self.localmapper.metadata.length[0].item()
        gridmap_msg.info.length_y = self.localmapper.metadata.length[1].item()
        gridmap_msg.info.pose.position.x = (
            self.localmapper.metadata.origin[0].item() + 0.5 * gridmap_msg.info.length_x
        )
        gridmap_msg.info.pose.position.y = (
            self.localmapper.metadata.origin[1].item() + 0.5 * gridmap_msg.info.length_y
        )
        gridmap_msg.info.pose.position.z = self.odom_msg.pose.pose.position.z
        gridmap_msg.info.pose.orientation.w = 1.0
        # transposed_layer_data = np.transpose(gridmap_data, (0, 2,1))
        # flipped_layer_data = np.flip(np.flip(transposed_layer_data, axis=1), axis=2)

        # gridmap_data has the shape (rows, cols, layers)
        # Step 1: Flip the 2D grid layers in both directions (reverse both axes)
        flipped_data = np.flip(gridmap_data, axis=(0, 1))  # Flips along both axes

        # Step 2: Transpose the first two dimensions (x, y) for each layer
        transposed_data = np.transpose(
            flipped_data, axes=(1, 0, 2)
        )  # Transpose rows and cols

        # Step 3: Flatten each 2D layer, maintaining the layers' structure (flattening across x, y)
        flattened_data = transposed_data.reshape(-1, gridmap_data.shape[-1])
        accum_time = 0
        for i in range(gridmap_data.shape[-1]):
            layer_data = gridmap_data[..., i]
            gridmap_layer_msg = Float32MultiArray()
            gridmap_layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="column_index",
                    size=layer_data.shape[0],
                    stride=layer_data.shape[0],
                )
            )
            gridmap_layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="row_index",
                    size=layer_data.shape[0],
                    stride=layer_data.shape[0] * layer_data.shape[1],
                )
            )

            # gridmap reverses the rasterization
            start_time = time.time()
            gridmap_layer_msg.data = flattened_data[:, i].tolist()
            end_time = time.time()
            accum_time += end_time - start_time

            # gridmap_layer_msg.data = flipped_layer_data[i].flatten().tolist()
            gridmap_msg.data.append(gridmap_layer_msg)
        self.get_logger().info("time to flatten layer {}: {}".format(i, accum_time))
        # add dummy elevation
        gridmap_msg.layers.append("elevation")
        layer_data = (
            np.zeros_like(gridmap_data[..., 0])
            + self.odom_msg.pose.pose.position.z
            - 1.73
        )
        gridmap_layer_msg = Float32MultiArray()
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0],
            )
        )
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0] * layer_data.shape[1],
            )
        )

        gridmap_layer_msg.data = layer_data.flatten().tolist()
        gridmap_msg.data.append(gridmap_layer_msg)

        return gridmap_msg

    def make_pcl_msg(self, pcl):
        """
        Convert dino pcl into message
        """
        self.get_logger().info("{}".format(pcl.shape))
        start_time = time.time()

        points = pcl.cpu().numpy()
        # Prepare the data array with XYZ and RGB
        xyz = np.zeros(
            points.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
            ],
        )

        # Assign XYZ values
        xyz["x"] = points[:, 0]
        xyz["y"] = points[:, 1]
        xyz["z"] = points[:, 2]

        msg = ros2_numpy.msgify(PointCloud2, xyz)
        msg.header.frame_id = self.odom_frame
        msg.header.stamp = self.pcl_msg.header.stamp
        self.get_logger().info("pcl total time: {}".format(time.time() - start_time))

        return msg

    def make_pcl_color_msg(self, pcl, vmin=None, vmax=None):
        """
        Convert dino pcl into message
        """
        self.get_logger().info("{}".format(pcl.shape))
        start_time = time.time()
        pcl_pos = pcl[:, :3].cpu().numpy()

        pcl_cs = pcl[:, 3:6]
        if vmin is None or vmax is None:
            vmin = pcl_cs.min(dim=0)[0].view(1, 3)
            vmax = pcl_cs.max(dim=0)[0].view(1, 3)
        else:
            vmin = vmin.view(1, 3)
            vmax = vmax.view(1, 3)
        pcl_cs = ((pcl_cs - vmin) / (vmax - vmin)).cpu().numpy()

        after_init_time = time.time()

        points = pcl_pos
        rgb_values = (pcl_cs * 255.0).astype(np.uint8)
        # Prepare the data array with XYZ and RGB
        xyzcolor = np.zeros(
            points.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.float32),
            ],
        )

        # Assign XYZ values
        xyzcolor["x"] = points[:, 0]
        xyzcolor["y"] = points[:, 1]
        xyzcolor["z"] = points[:, 2]

        color = np.zeros(
            points.shape[0], dtype=[("r", np.uint8), ("g", np.uint8), ("b", np.uint8)]
        )
        color["r"] = rgb_values[:, 0]
        color["g"] = rgb_values[:, 1]
        color["b"] = rgb_values[:, 2]
        xyzcolor["rgb"] = ros2_numpy.point_cloud2.merge_rgb_fields(color)

        msg = ros2_numpy.msgify(PointCloud2, xyzcolor)
        msg.header.frame_id = self.odom_frame
        msg.header.stamp = self.pcl_msg.header.stamp
        self.get_logger().info("pcl total time: {}".format(time.time() - start_time))

        return msg
    
    def make_img_msg(self, dino_img, vmin=None, vmax=None):
        if vmin is None or vmax is None:
            viz_img = normalize_dino(dino_img[..., :3])
        else:
            vmin = vmin.view(1, 1, 3)
            vmax = vmax.view(1, 1, 3)
            viz_img = (dino_img[..., :3] - vmin) / (vmax-vmin)
            viz_img = viz_img.clip(0., 1.)

        viz_img = viz_img.cpu().numpy() * 255
        img_msg = self.bridge.cv2_to_imgmsg(viz_img.astype(np.uint8), "rgb8")
        img_msg.header.stamp = self.img_msg.header.stamp
        return img_msg

    def simple_costmap(self, bev_grid, pos):
        # 'min_elevation', 'mean_elevation', 'max_elevation', 'num_voxels', 
        # 'min_elevation_filtered', 'min_elevation_filtered_mask', 'min_elevation_filtered_inflated', 'min_elevation_filtered_inflated_mask', 
        # 'diff'
        data = bev_grid.data.cpu().numpy()

        num_voxels = data[:,:,3]
        mask1 = num_voxels > 0
        mask2 = data[:,:,7] > 0

        robot_height = pos[2] - 1.8 # hard code
        data[:,:,:3] -= robot_height
        data[:,:,4] -= robot_height
        data[:,:,6] -= robot_height

        costmap = np.zeros_like(num_voxels, dtype=np.uint8)
        costmap[~mask2] = 128
        obstacle = data[:,:, 6] > 1.0
        obstacle[~mask2] = False
        costmap[obstacle] = 255

        rough = data[:,:, 8] > 0.5
        rough[~mask2] = False
        costmap[rough] = 180

        vis = cv2.applyColorMap(costmap, cv2.COLORMAP_JET)
        cv2.imshow('img',vis)
        cv2.waitKey(1)

    def vis_gridmap(self, bev_grid, pos):
        # 'min_elevation', 'mean_elevation', 'max_elevation', 'num_voxels', 
        # 'min_elevation_filtered', 'min_elevation_filtered_mask', 'min_elevation_filtered_inflated', 'min_elevation_filtered_inflated_mask', 
        # 'diff'
        data = bev_grid.data.cpu().numpy()

        num_voxels = data[:,:,3]
        mask1 = num_voxels > 0

        robot_height = pos[2] - 1.8 # hard code
        data[:,:,:3] -= robot_height
        data[:,:,4] -= robot_height
        data[:,:,6] -= robot_height

        visall = []
        for f in range(3):
            # print(data[mask1, f].max(), data[mask1, f].min(), data[mask1, f].mean(), data[mask1, f].std())
            vis = np.zeros((600, 600), dtype=np.uint8)
            vis[mask1] = np.clip((data[mask1, f] + 5) * 25, 0, 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            visall.append(vis)
            # cv2.imshow('img',vis)
            # cv2.waitKey(0)

        for f in [4,6]:
            vis = np.zeros((600, 600), dtype=np.uint8)
            mask = data[:,:,f+1] > 0
            vis[mask] = np.clip((data[mask, f] + 5) * 25, 0, 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            visall.append(vis)
            # cv2.imshow('img',vis)
            # cv2.waitKey(0)

        mask = data[:,:,7] > 0
        # print(data[mask, 8].max(), data[mask, 8].min(), data[mask, 8].mean(), data[mask, 8].std())
        vis = np.zeros((600, 600), dtype=np.uint8)
        vis[mask] = np.clip((data[mask, 8]) * 128 , 0, 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        visall.append(vis)

        visall = np.array(visall).reshape((2,3,600,600,3))
        visall = visall.transpose(0,2,1,3,4)
        visall = visall.reshape(1200,1800,3)
        cv2.imshow('img',visall)
        cv2.waitKey(1)


    def publish_messages(self, res):
        """
        Publish the dino pcl and dino map
        """
        pts = self.localmapper.voxel_grid.grid_indices_to_pts(
            self.localmapper.voxel_grid.raster_indices_to_grid_indices(
                self.localmapper.voxel_grid.all_indices
            )
        )
        pts_msg = self.make_pcl_msg(pts)
        self.voxel_viz_pub.publish(pts_msg)

        # colors = self.localmapper.voxel_grid.features[:, :3]
        # vmin = colors.min(dim=0)[0]
        # vmax = colors.max(dim=0)[0]

        # all_idxs = torch.cat([self.localmapper.voxel_grid.indices, self.localmapper.voxel_grid.all_indices])
        # unique, cnts = torch.unique(all_idxs, return_counts=True)
        # non_colorized_idxs = unique[cnts==1]

        # non_colorized_pts = self.localmapper.voxel_grid.grid_indices_to_pts(
        #     self.localmapper.voxel_grid.raster_indices_to_grid_indices(non_colorized_idxs)
        # )

        # color_placeholder = 0.1 * torch.ones(non_colorized_pts.shape[0], 3, device=non_colorized_pts.device)

        # pts = torch.cat([pts, non_colorized_pts], dim=0)
        # colors = torch.cat([colors, color_placeholder], dim=0)

        # voxel_viz_msg = self.make_pcl_msg(
        #     torch.cat([pts, colors], axis=-1), vmin=vmin, vmax=vmax
        # )
        # self.voxel_viz_pub.publish(voxel_viz_msg)

        # voxel_msg = self.make_voxel_msg(self.localmapper.voxel_grid)
        # self.voxel_pub.publish(voxel_msg)

        # pcl_msg = self.make_pcl_msg(res["dino_pcl"], vmin=vmin, vmax=vmax)
        # self.pcl_pub.publish(pcl_msg)

        # img_msg = self.make_img_msg(res["dino_image"], vmin=vmin, vmax=vmax)
        # self.image_pub.publish(img_msg)

        gridmap_msg = self.make_gridmap_msg(self.bev_grid)
        self.gridmap_pub.publish(gridmap_msg)

        timing_msg = Float32()
        self.timing_pub.publish(timing_msg)

    def spin(self):
        self.get_logger().info("spinning...")

        start_time = time.time()
        res = self.preprocess_inputs()
        after_preprocess_time = time.time()
        if res:
            self.get_logger().info("updating localmap...")

            pts = res["pcl"]
            # features = res["dino_pcl"][:, 3:]

            # self.get_logger().info(
            #     "Got {} features, mapping first {}".format(
            #         features.shape[-1], self.localmapper.n_features
            #     )
            # )

            self.get_logger().info(str(res.keys()))

            self.localmapper.update_pose(res["pos"])
            # self.localmapper.add_feature_pc(
            #     pts=pts, features=features[:, : self.localmapper.n_features]
            # )
            self.localmapper.add_pc(pts)

            if self.do_terrain_estimation:
                # 'min_elevation', 'mean_elevation', 'max_elevation', 'num_voxels', 
                # 'min_elevation_filtered', 'min_elevation_filtered_mask', 'min_elevation_filtered_inflated', 'min_elevation_filtered_inflated_mask', 
                # 'diff'
                self.bev_grid = self.terrain_estimator.run(self.localmapper.voxel_grid)

            after_update_time = time.time()
            # self.publish_messages(res)
            # self.vis_gridmap(self.bev_grid, res['pos'].cpu().numpy())
            self.simple_costmap(self.bev_grid, res['pos'].cpu().numpy())
            after_publish_time = time.time()

            self.get_logger().info(
                "preprocess time: {}".format(after_preprocess_time - start_time)
            )
            self.get_logger().info(
                "update time: {}".format(after_update_time - start_time)
            )
            self.get_logger().info(
                "publish time: {}".format(after_publish_time - after_update_time)
            )
            self.get_logger().info("total time: {}".format(time.time() - start_time))


def main(args=None):
    rclpy.init(args=args)

    visual_mapping_node = PCVoxelMappingNode()
    rclpy.spin(visual_mapping_node)

    visual_mapping_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()