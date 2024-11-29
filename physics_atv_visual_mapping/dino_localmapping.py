import rclpy
from rclpy.node import Node
import yaml
import copy
import array
import numpy as np

import ctypes

np.float = np.float64  # hack for numpify

import tf2_ros
import torch
import cv_bridge
import os

from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from perception_interfaces.msg import FeatureVoxelGrid

from physics_atv_visual_mapping.image_processing.image_pipeline import (
    setup_image_pipeline,
)
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *

from physics_atv_visual_mapping.localmapping.bev.bev_localmapper import BEVLocalMapper
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import (
    VoxelLocalMapper,
)
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata

from physics_atv_visual_mapping.utils import *

from physics_atv_visual_mapping.ros2_numpy_cpp import ros2_numpy_cpp

# approximate maximum per time stamp
# used for efficient dual buffering
MAX_N_VOXELS = 500000 # 0.5M

class DinoMappingNode(Node):
    def __init__(self):
        super().__init__("visual_mapping")

        self.get_logger().info("ORIGINAL VOXEL MAPPING")

        self.declare_parameter("config_fp", "")
        self.declare_parameter("models_dir", "")

        self.get_logger().info(
            "Looking for models in {}".format(
                self.get_parameter("models_dir").get_parameter_value().string_value
            )
        )

        config_fp = self.get_parameter("config_fp").get_parameter_value().string_value
        config = yaml.safe_load(open(config_fp, "r"))
        config["models_dir"] = (
            self.get_parameter("models_dir").get_parameter_value().string_value
        )

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

        self.image_pipeline = setup_image_pipeline(config)

        self.setup_localmapper(config)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = cv_bridge.CvBridge()

        self.compressed_img = config["image"]["image_compressed"]
        if self.compressed_img:
            self.image_sub = self.create_subscription(
                CompressedImage, config["image"]["image_topic"], self.handle_img, 1
            )
        else:
            self.image_sub = self.create_subscription(
                Image, config["image"]["image_topic"], self.handle_img, 1
            )

        self.intrinsics = (
            torch.tensor(config["intrinsics"]["P"], device=config["device"])
            .reshape(3, 3)
            .float()
        )
        self.dino_intrinsics = None

        self.extrinsics = pose_to_htm(
            np.concatenate(
                [
                    np.array(config["extrinsics"]["p"]),
                    np.array(config["extrinsics"]["q"]),
                ],
                axis=-1,
            )
        )

        self.pcl_sub = self.create_subscription(
            PointCloud2, config["pointcloud"]["topic"], self.handle_pointcloud, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, config["odometry"]["topic"], self.handle_odom, 10
        )

        self.pcl_pub = self.create_publisher(PointCloud2, "/dino_pcl", 10)
        self.image_pub = self.create_publisher(Image, "/dino_image", 10)

        self.toggler = 0
        if self.mapper_type == "bev":
            self.gridmap_pub = self.create_publisher(GridMap, "/dino_gridmap", 10)
        else:
            self.voxel_pub = self.create_publisher(FeatureVoxelGrid, "/dino_voxels", 10)
            self.voxel_viz_pub = self.create_publisher(
                PointCloud2, "/dino_voxels_viz", 1
            )
            self.features_l, self.indices_l = [], []
            self.set_dual_buffers(config)            

        self.timing_pub = self.create_publisher(Float32, "/dino_proc_time", 10)

        self.timer = self.create_timer(0.1, self.spin)
        self.viz = config["viz"]


    def set_dual_buffers(self, cfg) :
        cfg = cfg['localmapping']
        if self.mapper_type == "bev" :
            raise NotImplementedError
        
        else :
            n_features = cfg['n_features']

            for _ in range(2) :
                self.features_l.append(
                    array.array('f', [0] * MAX_N_VOXELS * n_features)
                )

                self.indices_l.append(
                    array.array('Q', [0] * MAX_N_VOXELS)
                )



    def setup_localmapper(self, config):
        """
        check that the localmapper metadata is good
        """
        self.mapper_type = config["localmapping"]["mapper_type"]
        metadata = LocalMapperMetadata(**self.base_metadata)

        assert self.mapper_type in [
            "bev",
            "voxel",
        ], "need mapper type to be either ['bev', 'voxel']"
        if self.mapper_type == "bev":
            assert metadata.ndims == 2, "need 2d metadata for BEV mapping"
            assert (
                metadata.resolution[0] - metadata.resolution[1]
            ).abs() < 1e-4, "need x,y resolution to match for ROS node"

            self.localmapper = BEVLocalMapper(
                metadata,
                n_features=config["localmapping"]["n_features"],
                ema=config["localmapping"]["ema"],
                device=config["device"],
            )
        elif self.mapper_type == "voxel":
            assert metadata.ndims == 3, "need 3d metadata for voxel mapping"
            self.localmapper = VoxelLocalMapper(
                metadata,
                n_features=config["localmapping"]["n_features"],
                ema=config["localmapping"]["ema"],
                device=config["device"],
            )
        else:
            self.get_logger().error("unsupported mapping type!")
            exit(1)

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

    def handle_img(self, msg):
        self.img_msg = msg

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

        if self.img_msg is None:
            self.get_logger().warn("no img msg received")
            return None

        if not self.tf_buffer.can_transform(
            self.vehicle_frame,
            self.pcl_msg.header.frame_id,
            rclpy.time.Time.from_msg(self.pcl_msg.header.stamp),
        ):
            self.get_logger().warn(
                "cant tf from {} to {} at {}".format(
                    self.vehicle_frame,
                    self.pcl_msg.header.frame_id,
                    self.pcl_msg.header.stamp,
                )
            )
            return None

        tf_vehicle_to_pcl_msg = self.tf_buffer.lookup_transform(
            self.vehicle_frame,
            self.pcl_msg.header.frame_id,
            rclpy.time.Time.from_msg(self.pcl_msg.header.stamp),
        )

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

        vehicle_to_pcl_htm = tf_msg_to_htm(tf_vehicle_to_pcl_msg).to(self.device)
        pcl_in_vehicle = transform_points(pcl.clone(), vehicle_to_pcl_htm)

        odom_to_pcl_htm = tf_msg_to_htm(tf_odom_to_pcl_msg).to(self.device)
        pcl_in_odom = transform_points(pcl.clone(), odom_to_pcl_htm)

        self.last_update_time = pcl_time

        # Camera processing
        if self.compressed_img:
            img = (
                self.bridge.compressed_imgmsg_to_cv2(
                    self.img_msg, desired_encoding="rgb8"
                )
                / 255.0
            )
        else:
            img = (
                self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding="rgb8").astype(
                    np.float32
                )
                / 255.0
            )

        img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)

        dino_img, dino_intrinsics = self.image_pipeline.run(
            img, self.intrinsics.unsqueeze(0)
        )
        # dino_img = img.to(self.device)
        # dino_intrinsics = self.intrinsics.unsqueeze(0).to(self.device)
        dino_img = dino_img[0].permute(1, 2, 0)
        dino_intrinsics = dino_intrinsics[0]

        I = get_intrinsics(dino_intrinsics).to(self.device)
        E = get_extrinsics(self.extrinsics).to(self.device)
        P = obtain_projection_matrix(I, E)

        pixel_coordinates = get_pixel_from_3D_source(pcl_in_vehicle[:, :3], P)
        (
            lidar_points_in_frame,
            pixels_in_frame,
            ind_in_frame,
        ) = get_points_and_pixels_in_frame(
            pcl_in_vehicle[:, :3], pixel_coordinates, dino_img.shape[0], dino_img.shape[1]
        )

        dino_features = dino_img[pixels_in_frame[:, 1], pixels_in_frame[:, 0]]
        dino_pcl = torch.cat([pcl_in_odom[ind_in_frame][:, :3], dino_features], dim=-1)

        pos = torch.tensor(
            [
                self.odom_msg.pose.pose.position.x,
                self.odom_msg.pose.pose.position.y,
                self.odom_msg.pose.pose.position.z,
            ],
            device=self.device,
        )

        self.get_logger().info('colorized {}/{} points'.format(dino_pcl.shape[0], pcl_in_vehicle.shape[0]))

        return {
            "pos": pos,
            "pcl": pcl_in_odom,
            "image": img,
            "dino_image": dino_img,
            "dino_pcl": dino_pcl,
            "pixel_projection": pixel_coordinates[ind_in_frame],
        }

    def make_voxel_msg(self, voxel_grid):
        msg = FeatureVoxelGrid()
        msg.header.stamp = self.pcl_msg.header.stamp
        msg.header.frame_id = self.odom_frame

        msg.metadata.origin.x = voxel_grid.metadata.origin[0].item()
        msg.metadata.origin.y = voxel_grid.metadata.origin[1].item()
        msg.metadata.origin.z = voxel_grid.metadata.origin[2].item()

        msg.metadata.length.x = voxel_grid.metadata.length[0].item()
        msg.metadata.length.y = voxel_grid.metadata.length[1].item()
        msg.metadata.length.z = voxel_grid.metadata.length[2].item()

        msg.metadata.resolution.x = voxel_grid.metadata.resolution[0].item()
        msg.metadata.resolution.y = voxel_grid.metadata.resolution[1].item()
        msg.metadata.resolution.z = voxel_grid.metadata.resolution[2].item()      

        n_voxels = voxel_grid.features.shape[0]
        n_features = voxel_grid.features.shape[1]

        msg.num_voxels = n_voxels
        msg.num_features = n_features

        if self.layer_keys is None:
            msg.feature_keys = [
                "{}_{}".format(self.layer_key, i) for i in range(voxel_grid.features.shape[1])
            ]
        else:
            msg.feature_keys = copy.deepcopy(self.layer_keys)

        msg.features.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=voxel_grid.features.shape[0],
                stride=voxel_grid.features.shape[0],
            )
        )
        msg.features.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=voxel_grid.features.shape[0],
                stride=voxel_grid.features.shape[0] * voxel_grid.features.shape[1],
            )
        )

        self.get_logger().info("Voxel grid shape = {}".format(voxel_grid.features.shape))

        # msg.indices = voxel_grids.indices.tolist()
        # msg.features.data = voxel_grid.features.flatten().tolist()        
       
        msg.features.data = self.features_l[self.toggler]
        msg.indices = self.indices_l[self.toggler]
        self.toggler = (self.toggler + 1) % 2

        ros2_numpy_cpp.npcpp_voxel_grid(
            np.ascontiguousarray(voxel_grid.features.data.cpu().numpy()),
            np.ascontiguousarray(voxel_grid.indices.data.cpu().numpy()).astype(np.uint64),
            msg.features.data.buffer_info()[0],
            msg.indices.buffer_info()[0],
            msg.num_voxels,
            msg.num_features,
        )

        return msg    


    def make_gridmap_msg(self, vmin=None, vmax=None):
        """
        convert dino into gridmap msg

        Publish all the feature channels, plus a visualization and elevation layer

        Note that we assume all the requisite stuff is available (pcl, img, odom) as this
        should only be called after a dino map is successfully produced
        """
        gridmap_msg = GridMap()

        gridmap_data = self.localmapper.bev_grid.data.cpu().numpy()

        # setup metadata
        print("gridmap_msg", dir(gridmap_msg.info))
        gridmap_msg.header.stamp = self.img_msg.header.stamp
        gridmap_msg.header.frame_id = self.odom_frame

        if self.layer_keys is None:
            gridmap_msg.layers = [
                "{}_{}".format(self.layer_key, i) for i in range(gridmap_data.shape[-1])
            ]
        else:
            gridmap_msg.layers = copy.deepcopy(self.layer_keys)

        gridmap_msg.info.resolution = self.localmapper.metadata.resolution.mean().item()
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

        # TODO: figure out how to support multiple viz output types
        gridmap_rgb = gridmap_data[..., :3]
        if vmin is None or vmax is None:
            vmin = vmin.cpu().numpy().reshape(1, 1, 3)
            vmax = vmax.cpu().numpy().reshape(1, 1, 3)
        else:
            vmin = gridmap_rgb.reshape(-1, 3).min(axis=0).reshape(1, 1, 3)
            vmax = gridmap_rgb.reshape(-1, 3).max(axis=0).reshape(1, 1, 3)

        gridmap_cs = ((gridmap_rgb - vmin) / (vmax - vmin)).clip(0.0, 1.0)
        gridmap_cs = (gridmap_cs * 255.0).astype(np.int32)

        gridmap_color = (
            gridmap_cs[..., 0] * (2 ** 16)
            + gridmap_cs[..., 1] * (2 ** 8)
            + gridmap_cs[..., 2]
        )
        gridmap_color = gridmap_color.view(dtype=np.float32)

        gridmap_msg.layers.append("rgb_viz")
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

        gridmap_layer_msg.data = gridmap_color.T[::-1, ::-1].flatten().tolist()
        gridmap_msg.data.append(gridmap_layer_msg)

        return gridmap_msg

    def make_pcl_msg(self, pcl, vmin=None, vmax=None):
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

    def xyz_array_to_point_cloud_msg(
        self, points, frame, timestamp=None, rgb_values=None
    ):
        """
        Modified from: https://github.com/castacks/physics_atv_deep_stereo_vo/blob/main/src/stereo_node_multisense.py
        Please refer to this ros answer about the usage of point cloud message:
            https://answers.ros.org/question/234455/pointcloud2-and-pointfield/
        :param points:
        :param header:
        :return:
        """
        # Create the message header
        header = Header()
        header.frame_id = frame
        if timestamp is None:
            timestamp = self.get_clock().now().to_msg()  # ROS2 time function
        header.stamp = timestamp

        msg = PointCloud2()
        msg.header = header

        # Determine point cloud dimensions (organized vs unorganized)
        if len(points.shape) == 3:  # Organized point cloud
            msg.width = points.shape[0]
            msg.height = points.shape[1]
        else:  # Unorganized point cloud
            msg.width = points.shape[0]
            msg.height = 1

        msg.is_bigendian = False
        msg.is_dense = False  # Set to False since organized clouds are non-dense

        if rgb_values is None:
            # XYZ only
            msg.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            msg.point_step = 12  # 3 fields (x, y, z) each 4 bytes (float32)
            msg.row_step = msg.point_step * msg.width
            xyz = points.astype(np.float32)
            msg.data = xyz.tobytes()  # Convert to bytes
        else:
            # XYZ and RGB
            msg.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            ]
            msg.point_step = (
                16  # 4 fields (x, y, z, rgb) with rgb as a packed 32-bit integer
            )
            msg.row_step = msg.point_step * msg.width

            # Prepare the data array with XYZ and RGB
            xyzcolor = np.zeros(
                (points.shape[0],),
                dtype={
                    "names": ("x", "y", "z", "rgba"),
                    "formats": ("f4", "f4", "f4", "u4"),
                },
            )

            # Assign XYZ values
            xyzcolor["x"] = points[:, 0]
            xyzcolor["y"] = points[:, 1]
            xyzcolor["z"] = points[:, 2]

            # Prepare RGB values (packed into a 32-bit unsigned int)
            rgb_uint32 = np.zeros(points.shape[0], dtype=np.uint32)
            rgb_uint32 = (
                np.left_shift(rgb_values[:, 0].astype(np.uint32), 16)
                | np.left_shift(rgb_values[:, 1].astype(np.uint32), 8)
                | rgb_values[:, 2].astype(np.uint32)
            )
            xyzcolor["rgba"] = rgb_uint32

            msg.data = xyzcolor.tobytes()  # Convert to bytes

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

    def publish_messages(self, res):
        """
        Publish the dino pcl and dino map
        """
        if self.mapper_type == "bev":
            #use the aggregated embeddings for min/max to keep consistency across image/pcl/map
            bev_viz_data = self.localmapper.bev_grid.data[..., :3].view(-1, 3)
            vmin = bev_viz_data.min(dim=0)[0]
            vmax = bev_viz_data.max(dim=0)[0]

            gridmap_msg = self.make_gridmap_msg(vmin=vmin, vmax=vmax)
            self.gridmap_pub.publish(gridmap_msg)

            pcl_msg = self.make_pcl_msg(res["dino_pcl"], vmin=vmin, vmax=vmax)
            self.pcl_pub.publish(pcl_msg)

            img_msg = self.make_img_msg(res["dino_image"], vmin=vmin, vmax=vmax)
            self.image_pub.publish(img_msg)

        elif self.mapper_type == "voxel":
            pts = self.localmapper.voxel_grid.grid_indices_to_pts(
                self.localmapper.voxel_grid.raster_indices_to_grid_indices(
                    self.localmapper.voxel_grid.indices
                )
            )
            colors = self.localmapper.voxel_grid.features[:, :3]
            vmin = colors.min(dim=0)[0]
            vmax = colors.max(dim=0)[0]

            voxel_viz_msg = self.make_pcl_msg(
                torch.cat([pts, colors], axis=-1), vmin=vmin, vmax=vmax
            )
            self.voxel_viz_pub.publish(voxel_viz_msg)

            after_update_time = time.time()
            voxel_msg = self.make_voxel_msg(self.localmapper.voxel_grid)
            voxel_msg.features.data = voxel_msg.features.data[:voxel_msg.num_voxels*voxel_msg.num_features]
            voxel_msg.indices = voxel_msg.indices[:voxel_msg.num_voxels]
            self.get_logger().info(str(len(self.features_l[self.toggler])))
            after_msg_creation_time = time.time()
            self.get_logger().info(
                "msg creation time: {}".format(after_msg_creation_time - after_update_time)
            )            
            self.voxel_pub.publish(voxel_msg)
            after_publish_time = time.time()            

            pcl_msg = self.make_pcl_msg(res["dino_pcl"], vmin=vmin, vmax=vmax)
            self.pcl_pub.publish(pcl_msg)

            img_msg = self.make_img_msg(res["dino_image"], vmin=vmin, vmax=vmax)
            self.image_pub.publish(img_msg)

        timing_msg = Float32()
        self.timing_pub.publish(timing_msg)

    def spin(self):
        self.get_logger().info("spinning...")

        start_time = time.time()
        res = self.preprocess_inputs()
        after_preprocess_time = time.time()
        if res:
            self.get_logger().info("updating localmap...")

            pts = res["dino_pcl"][:, :3]
            features = res["dino_pcl"][:, 3:]

            self.get_logger().info(
                "Got {} features, mapping first {}".format(
                    features.shape[-1], self.localmapper.n_features
                )
            )

            self.localmapper.update_pose(res["pos"])
            self.localmapper.add_feature_pc(
                pts=pts, features=features[:, : self.localmapper.n_features]
            )

            after_update_time = time.time()
            self.publish_messages(res)
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

    visual_mapping_node = DinoMappingNode()
    rclpy.spin(visual_mapping_node)

    visual_mapping_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()