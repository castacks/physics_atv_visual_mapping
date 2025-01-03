import rclpy
from rclpy.node import Node
import yaml
import copy
import numpy as np
np.float = np.float64  # hack for numpify

import tf2_ros
import torch
import cv_bridge
import os
from sensor_msgs.msg import PointCloud2, CameraInfo, Image, CompressedImage
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.localmapping.localmapping import *
from physics_atv_visual_mapping.utils import *

class DinoMappingNode(Node):
    def __init__(self):
        super().__init__('visual_mapping')
        
        self.declare_parameter('config_fp', '')

        config_fp = self.get_parameter('config_fp').get_parameter_value().string_value
        config = yaml.safe_load(open(config_fp, 'r'))

        self.localmap = None
        self.pcl_msg = None
        self.odom_msg = None
        self.img_msg = None
        self.odom_frame = None

        self.device = config['device']
        self.base_metadata = config['localmapping']['metadata']
        self.localmap_ema = config['localmapping']['ema']
        self.layer_key = config['localmapping']['layer_key'] if 'layer_key' in config['localmapping'].keys() else None
        self.layer_keys = self.make_layer_keys(config['localmapping']['layer_keys']) if 'layer_keys' in config['localmapping'].keys() else None
        self.last_update_time = 0.

        self.image_pipeline = setup_image_pipeline(config)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = cv_bridge.CvBridge()

        self.compressed_img = config['image']['image_compressed']
        if self.compressed_img:
            self.image_sub = self.create_subscription(CompressedImage, config['image']['image_topic'], self.handle_img, 1)
        else:
            self.image_sub = self.create_subscription(Image, config['image']['image_topic'], self.handle_img, 1)

        self.intrinsics = torch.tensor(config['intrinsics']['P'], device=config['device']).reshape(3, 3).float()
        self.dino_intrinsics = None

        self.extrinsics = pose_to_htm(np.concatenate([
            np.array(config['extrinsics']['p']),
            np.array(config['extrinsics']['q'])
        ], axis=-1))

        self.pcl_sub = self.create_subscription(PointCloud2, config['pointcloud']['topic'], self.handle_pointcloud, 1)
        self.odom_sub = self.create_subscription(Odometry, config['odometry']['topic'], self.handle_odom, 10)

        self.pcl_pub = self.create_publisher(PointCloud2, '/dino_pcl', 1)
        self.gridmap_pub = self.create_publisher(GridMap, '/dino_gridmap', 1)
        self.image_pub = self.create_publisher(Image, '/dino_image', 1)

        self.timer = self.create_timer(0.2, self.spin)
        self.viz = config['viz']

    def make_layer_keys(self, layer_keys):
        out = []
        for lk in layer_keys:
            for i in range(lk['n']):
                out.append('{}_{}'.format(lk['key'], i))
        return out

    def handle_pointcloud(self, msg):
        self.pcl_msg = msg
        # self.pcl_msg.header.frame_id = 'zed_camera_link' # TODO: parametrize
        self.pcl_msg.header.frame_id = 'zed_left_camera_frame'

    def handle_odom(self, msg):
        self.odom_msg = msg
        self.odom_msg.child_frame_id = 'zed_left_camera_frame' # TODO: parametrize
        if self.odom_frame is None:
            self.odom_frame = msg.header.frame_id

    def handle_img(self, msg):
        self.img_msg = msg

    def preprocess_inputs(self):
        if self.pcl_msg is None:
            self.get_logger().warn('no pcl msg received')
            return None

        pcl_time = self.pcl_msg.header.stamp.sec + self.pcl_msg.header.stamp.nanosec * 1e-9
        if abs(pcl_time - self.last_update_time) < 1e-3:
            return None

        if self.odom_msg is None:
            self.get_logger().warn('no odom msg received')
            return None

        if self.img_msg is None:
            self.get_logger().warn('no img msg received')
            return None

        if self.odom_msg.child_frame_id != self.pcl_msg.header.frame_id:
            self.get_logger().warn('for now, need pcls in the child frame of odom')
            return None

        if not self.tf_buffer.can_transform(self.odom_frame, self.pcl_msg.header.frame_id, rclpy.time.Time.from_msg(self.pcl_msg.header.stamp)):
            self.get_logger().warn('cant tf from {} to {} at {}'.format(self.odom_frame, self.pcl_msg.header.frame_id, self.pcl_msg.header.stamp))
            return None

        tf_msg = self.tf_buffer.lookup_transform(self.odom_frame, self.pcl_msg.header.frame_id, rclpy.time.Time.from_msg(self.pcl_msg.header.stamp))
        

        pcl_htm = tf_msg_to_htm(tf_msg).to(self.device)
        pcl = pcl_msg_to_xyz(self.pcl_msg).to(self.device)
        pcl_odom = transform_points(pcl.clone(), pcl_htm)

        _metadata = {
            'origin': torch.tensor([
                self.odom_msg.pose.pose.position.x-0.5*self.base_metadata['length_x'],
                self.odom_msg.pose.pose.position.y-0.5*self.base_metadata['length_y']
            ]).float().to(self.device),
            'length_x': torch.tensor(self.base_metadata['length_x']).float().to(self.device),
            'length_y': torch.tensor(self.base_metadata['length_y']).float().to(self.device),
            'resolution': torch.tensor(self.base_metadata['resolution']).float().to(self.device),
        }
        self.last_update_time = pcl_time

        # Camera processing
        if self.compressed_img:
            img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8')/255.
        else:
            img = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8').astype(np.float32)/255.

        img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)

        # Get time 
        torch.cuda.synchronize()
        start_time = time.time()
        dino_img, dino_intrinsics = self.image_pipeline.run(img, self.intrinsics.unsqueeze(0))
        torch.cuda.synchronize()
        self.get_logger().info('size of img: {}, time to process image: {}'.format(img.shape, time.time()-start_time))
        # dino_img = img.to(self.device)
        # dino_intrinsics = self.intrinsics.unsqueeze(0).to(self.device)
        dino_img = dino_img[0].permute(1, 2, 0)
        dino_intrinsics = dino_intrinsics[0]

        torch.cuda.synchronize()
        start_time = time.time()
        I = get_intrinsics(dino_intrinsics).to(self.device)
        E = get_extrinsics(self.extrinsics).to(self.device)
        P = obtain_projection_matrix(I, E)
        
        pixel_coordinates = get_pixel_from_3D_source(pcl[:, :3], P)
        torch.cuda.synchronize()
        self.get_logger().info('time to project pcl: {} PCL shape: {}, Pixel shape: {}'.format(time.time()-start_time, pcl.shape, pixel_coordinates.shape))

        torch.cuda.synchronize()
        start_time = time.time()
        lidar_points_in_frame, pixels_in_frame, ind_in_frame = get_points_and_pixels_in_frame(
            pcl[:, :3],
            pixel_coordinates,
            dino_img.shape[0],
            dino_img.shape[1]
        )
        torch.cuda.synchronize()
        self.get_logger().info('time to get points and pixels in frame: {}'.format(time.time()-start_time))

        torch.cuda.synchronize()
        start_time = time.time()
        dino_features = dino_img[pixels_in_frame[:, 1], pixels_in_frame[:, 0]]
        dino_pcl = torch.cat([pcl_odom[ind_in_frame][:, :3], dino_features], dim=-1)
        torch.cuda.synchronize()
        self.get_logger().info('time to get features: {}'.format(time.time()-start_time))
        
        return {
            'pcl': pcl_odom,
            'metadata': _metadata,
            'image': img,
            'dino_image': dino_img,
            'dino_pcl': dino_pcl,
            'pixel_projection': pixel_coordinates[ind_in_frame]
        }
        
    def update_localmap(self, pcl, metadata):
        pcl_pos = pcl[:, :3]
        pcl_data = pcl[:, 3:]
        localmap, known_mask, metadata_out = localmap_from_pointcloud(pcl_pos, pcl_data, metadata)
        localmap_update = {
            'data': localmap,
            'known': known_mask,
            'metadata': metadata_out
        }
        if self.localmap is None:
            return localmap_update

        else:
            return aggregate_localmaps(localmap_update, self.localmap, ema=self.localmap_ema)
    def make_gridmap_msg(self, localmap):
        """
        convert dino into gridmap msg

        Publish all the feature channels, plus a visualization and elevation layer

        Note that we assume all the requisite stuff is available (pcl, img, odom) as this
        should only be called after a dino map is successfully produced
        """
        gridmap_msg = GridMap()

        gridmap_data = localmap['data'].cpu().numpy()

        #setup metadata
        print("gridmap_msg", dir(gridmap_msg.info))
        gridmap_msg.header.stamp = self.img_msg.header.stamp
        gridmap_msg.header.frame_id = self.odom_frame
        
        # gridmap_msg.layers = ["VLAD_1", "VLAD_2", "VLAD_3", "VLAD_4", "VLAD_5", "VLAD_6", "VLAD_7", "VLAD_8","height"]

        # TODO (cherie): clean up below for self.layer_keys making
        if self.layer_keys is None:
            gridmap_msg.layers = ['{}_{}'.format(self.layer_key, i) for i in range(gridmap_data.shape[-1])]
        else:
            gridmap_msg.layers = copy.deepcopy(self.layer_keys)
            

        gridmap_msg.info.resolution = localmap['metadata']['resolution'].item()
        gridmap_msg.info.length_x = localmap['metadata']['length_x'].item()
        gridmap_msg.info.length_y = localmap['metadata']['length_y'].item()
        gridmap_msg.info.pose.position.x = localmap['metadata']['origin'][0].item() + 0.5*gridmap_msg.info.length_x
        gridmap_msg.info.pose.position.y = localmap['metadata']['origin'][1].item() + 0.5*gridmap_msg.info.length_y
        gridmap_msg.info.pose.position.z = self.odom_msg.pose.pose.position.z
        gridmap_msg.info.pose.orientation.w = 1.
        # transposed_layer_data = np.transpose(gridmap_data, (0, 2,1))
        # flipped_layer_data = np.flip(np.flip(transposed_layer_data, axis=1), axis=2)
        
        
        # gridmap_data has the shape (rows, cols, layers)
        # Step 1: Flip the 2D grid layers in both directions (reverse both axes)
        flipped_data = np.flip(gridmap_data, axis=(0, 1))  # Flips along both axes

        # Step 2: Transpose the first two dimensions (x, y) for each layer
        transposed_data = np.transpose(flipped_data, axes=(1, 0, 2))  # Transpose rows and cols

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
                    stride=layer_data.shape[0]
                )
            )
            gridmap_layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="row_index",
                    size=layer_data.shape[0],
                    stride=layer_data.shape[0] * layer_data.shape[1]
                )
            )

            #gridmap reverses the rasterization
            start_time = time.time()
            gridmap_layer_msg.data = flattened_data[:, i].tolist()
            end_time = time.time()
            accum_time += end_time - start_time

            # gridmap_layer_msg.data = flipped_layer_data[i].flatten().tolist()
            gridmap_msg.data.append(gridmap_layer_msg)
        self.get_logger().info('time to flatten layer {}: {}'.format(i, accum_time))
        #add dummy elevation
        gridmap_msg.layers.append('elevation')
        layer_data = np.zeros_like(gridmap_data[...,0]) + self.odom_msg.pose.pose.position.z - 1.73
        gridmap_layer_msg = Float32MultiArray()
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0]
            )
        )
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0] * layer_data.shape[1]
            )
        )

        gridmap_layer_msg.data = layer_data.flatten().tolist()
        gridmap_msg.data.append(gridmap_layer_msg)

        """
        #Add rgb viz of top 3
        ###
        if self.mode == 'pca':
            gridmap_rgb = gridmap_data[..., :3]
            vmin = gridmap_rgb.reshape(-1, 3).min(axis=0).reshape(1,1,3)
            vmax = gridmap_rgb.reshape(-1, 3).max(axis=0).reshape(1,1,3)
            gridmap_cs = (gridmap_rgb-vmin)/(vmax-vmin)
            gridmap_cs = (gridmap_cs*255.).astype(np.int32)
        elif self.mode == 'vlad':
            gridmap_da = np.argmin(gridmap_data,axis=-1)
            mins = np.min(gridmap_data,axis=-1)
            gridmap_da[mins/26 > .85] = -1
            gridmap_cs = (self.Csub[gridmap_da]*255).astype(np.int32)
        """

        #TODO: figure out how to support multiple viz output types
        gridmap_rgb = gridmap_data[..., :3]
        vmin = gridmap_rgb.reshape(-1, 3).min(axis=0).reshape(1,1,3)
        vmax = gridmap_rgb.reshape(-1, 3).max(axis=0).reshape(1,1,3)
        gridmap_cs = ((gridmap_rgb-vmin)/(vmax-vmin)).clip(0., 1.)
        gridmap_cs = (gridmap_cs*255.).astype(np.int32)

        gridmap_color = gridmap_cs[..., 0] * (2**16) + gridmap_cs[..., 1] * (2**8) + gridmap_cs[..., 2]
        gridmap_color = gridmap_color.view(dtype=np.float32)

        gridmap_msg.layers.append('rgb_viz')
        gridmap_layer_msg = Float32MultiArray()
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0]
            )
        )
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0] * layer_data.shape[1]
            )
        )

        gridmap_layer_msg.data = gridmap_color.T[::-1, ::-1].flatten().tolist()
        gridmap_msg.data.append(gridmap_layer_msg)

        return gridmap_msg

    def make_pcl_msg(self, pcl):
        """
        Convert dino pcl into message
        """
        start_time = time.time()
        pcl_pos = pcl[:, :3].cpu().numpy()

        pcl_cs = pcl[:, [5,4,3]]
        vmin = pcl_cs.min(dim=0)[0].view(1,3)
        vmax = pcl_cs.max(dim=0)[0].view(1,3)
        pcl_cs =((pcl_cs - vmin) / (vmax - vmin)).cpu().numpy()
        
        after_init_time = time.time()

        # msg = self.xyz_array_to_point_cloud_msg(
        #     points=pcl_pos,
        #     frame=self.odom_frame,
        #     timestamp=self.pcl_msg.header.stamp,
        #     rgb_values=(pcl_cs*255.).astype(np.uint8)
        # )
        
        points = pcl_pos 
        rgb_values = (pcl_cs*255.).astype(np.uint8)
        # Prepare the data array with XYZ and RGB
        xyzcolor = np.zeros(points.shape[0], dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32), 
            ('rgb', np.float32)
        ])

        # Assign XYZ values
        xyzcolor['x'] = points[:, 0]
        xyzcolor['y'] = points[:, 1]
        xyzcolor['z'] = points[:, 2]
        
        color = np.zeros(points.shape[0], dtype=[('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
        color['r'] = rgb_values[:, 0]
        color['g'] = rgb_values[:, 1]
        color['b'] = rgb_values[:, 2]
        xyzcolor['rgb'] = ros2_numpy.point_cloud2.merge_rgb_fields(color)
        
        
        msg = ros2_numpy.msgify(PointCloud2, xyzcolor)
        msg.header.frame_id = self.odom_frame
        msg.header.stamp = self.pcl_msg.header.stamp
        self.get_logger().info('pcl total time: {}'.format(time.time()-start_time))

        return msg

    def xyz_array_to_point_cloud_msg(self, points, frame, timestamp=None, rgb_values=None):
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
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            msg.point_step = 12  # 3 fields (x, y, z) each 4 bytes (float32)
            msg.row_step = msg.point_step * msg.width
            xyz = points.astype(np.float32)
            msg.data = xyz.tobytes()  # Convert to bytes
        else:
            # XYZ and RGB
            msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
            ]
            msg.point_step = 16  # 4 fields (x, y, z, rgb) with rgb as a packed 32-bit integer
            msg.row_step = msg.point_step * msg.width

            # Prepare the data array with XYZ and RGB
            xyzcolor = np.zeros((points.shape[0],), dtype={
                'names': ('x', 'y', 'z', 'rgba'),
                'formats': ('f4', 'f4', 'f4', 'u4')
            })

            # Assign XYZ values
            xyzcolor['x'] = points[:, 0]
            xyzcolor['y'] = points[:, 1]
            xyzcolor['z'] = points[:, 2]

            # Prepare RGB values (packed into a 32-bit unsigned int)
            rgb_uint32 = np.zeros(points.shape[0], dtype=np.uint32)
            rgb_uint32 = np.left_shift(rgb_values[:, 0].astype(np.uint32), 16) | \
                         np.left_shift(rgb_values[:, 1].astype(np.uint32), 8) | \
                         rgb_values[:, 2].astype(np.uint32)
            xyzcolor['rgba'] = rgb_uint32

            msg.data = xyzcolor.tobytes()  # Convert to bytes

        return msg

    def publish_messages(self, res):
        """
        Publish the dino pcl and dino map
        """
        gridmap_msg = self.make_gridmap_msg(self.localmap)
        self.gridmap_pub.publish(gridmap_msg)
        pcl_msg = self.make_pcl_msg(res['dino_pcl'])
        self.pcl_pub.publish(pcl_msg)

        """
        if self.mode == 'pca':
            vmin = self.localmap['data'][..., :3].view(-1, 3).min(dim=0)[0].view(1,1,3)
            vmax = self.localmap['data'][..., :3].view(-1, 3).max(dim=0)[0].view(1,1,3)
            viz_img = ((res['dino_image'][..., :3]-vmin)/(vmax-vmin)).clip(0., 1.).cpu().numpy() * 255
        elif self.mode == 'vlad':
            da = np.argmin(res['dino_image'].cpu().numpy(),axis=-1)
            # mins = np.min(da,axis=-1)
            # gridmap_da[mins/26 > .85] = -1
            viz_img = (self.Csub[da]*255).astype(np.int32)
        """

        #TODO: figure out how to support multiple viz types
        vmin = self.localmap['data'][..., :3].view(-1, 3).min(dim=0)[0].view(1,1,3)
        vmax = self.localmap['data'][..., :3].view(-1, 3).max(dim=0)[0].view(1,1,3)
        viz_img = ((res['dino_image'][..., :3]-vmin)/(vmax-vmin)).clip(0., 1.).cpu().numpy() * 255

        # img_msg = self.bridge.cv2_to_imgmsg(vimg, "bgr8")
        img_msg = self.bridge.cv2_to_imgmsg(viz_img.astype(np.uint8), "rgb8")
        img_msg.header.stamp = pcl_msg.header.stamp
        self.image_pub.publish(img_msg)
    def spin(self):
        self.get_logger().info('spinning...')

        start_time = time.time()
        res = self.preprocess_inputs()
        after_preprocess_time = time.time()
        if res:
            
            self.get_logger().info('updating localmap...')

            self.localmap = self.update_localmap(
                pcl=res['dino_pcl'],
                metadata=res['metadata']
            )
            after_update_time = time.time()
            self.publish_messages(res)
            after_publish_time = time.time()
            self.get_logger().info('preprocess time: {}'.format(after_preprocess_time-start_time))
            self.get_logger().info('update time: {}'.format(after_update_time-start_time))
            self.get_logger().info('publish time: {}'.format(after_publish_time-after_update_time))
            self.get_logger().info('total time: {}'.format(time.time()-start_time))


def main(args=None):
    rclpy.init(args=args)

    visual_mapping_node = DinoMappingNode()
    rclpy.spin(visual_mapping_node)

    visual_mapping_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
