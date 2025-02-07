#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
import numpy as np
from threading import Lock
import time
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import matplotlib.pyplot as plt
from rosbag_to_dataset.dtypes.gridmap import GridMapConvert
import matplotlib
import torch
import torch.nn.functional as F
import cv2 

CMAP = matplotlib.cm.get_cmap('magma')

class LethalHeightCost(Node):
    def __init__(self, odom_topic, gridmap_topic, costmap_topic, uncmap_topic):
        super().__init__('lethal_height_cost')

        self._lock = Lock()
        self.odom_msg = None
        self.hz_counter = 0

        self.gridmap_sub = self.create_subscription(GridMap, gridmap_topic, self.handle_map, 1)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.handle_odom, 1)

        self.timer = self.create_timer(0.1, self.run_map)
        # self.costmap_pub = self.create_publisher(GridMap, costmap_topic, 2)
        # self.uncmap_pub = self.create_publisher(GridMap, uncmap_topic, 2)
        self.allmap_pub = self.create_publisher(GridMap, '/cherie_allmap', 2) # Feature, cost, and uncertainty maps

        self.new_msg = False
        self.cost = 0.0
        self.feat_names = []
        self.grid_map_cvt = GridMapConvert(channels=self.feat_names, size=[1, 1])
        
        # TODO: parametrize
        self.residual_max = 22.5 # from GP_wheelchair.yaml
        self.uc_thresh = .9 # from GP_wheelchair.yaml

        print('DONE WITH INIT')

    def handle_odom(self, msg):
        self.velocity = np.linalg.norm([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        self.odom_msg = msg

    def handle_map(self, msg):
        self.get_logger().info('handling map...')
        with self._lock:
            self.dino_map_msg = msg
            self.new_msg = True
            if len(self.feat_names) == 0:
                for layer in msg.layers:
                    if 'height' in layer:
                        self.feat_names.append(layer)
                    if 'VLAD' in layer:
                        self.feat_names.append(layer)
                    if 'known' in layer:
                        self.feat_names.append(layer)
                self.grid_map_cvt.channels = self.feat_names

    def run_map(self):
        now = time.perf_counter()
        print('----')
        if self.odom_msg is not None:
            if not self.new_msg:
                self.get_logger().info('no new map')
                return

            with self._lock:
                self.get_logger().info('got map')
                if self.dino_map_msg is not None:
                    info = self.dino_map_msg.info
                    header = self.dino_map_msg.header
                    nx = round(info.length_x / info.resolution)
                    ny = round(info.length_y / info.resolution)
                    self.get_logger().info(f"Map size: {nx} x {ny} info x: {info.length_x} y: {info.length_y} res: {info.resolution}")
                    self.grid_map_cvt.size = [nx, ny]

                    gridmap = self.grid_map_cvt.ros_to_numpy(self.dino_map_msg)

                    self.new_msg = False
                else:
                    gridmap = None

            if gridmap is None:
                print("NO MAP")
                return

            # Get uncertainty map from residual values
            unc_map = gridmap['data'][:8,:,:].min(axis=0)
            unc_map /= self.residual_max
            unc_map[unc_map < self.uc_thresh] = 0
            e_kernel = np.ones((2, 2), np.float32)
            d_kernel = np.ones((3, 3), np.float32)
            unc_map = cv2.erode(unc_map, e_kernel, iterations=1)
            unc_map = cv2.dilate(unc_map, d_kernel, iterations=1)
            # Mask out unknown 
            unc_map[gridmap['data'][8] == -1] = 0
            
            # self.get_logger().info(f"gridmap['data'] shape: {gridmap['data'].shape}")
            # self.get_logger().info(f"Uncertainty map min: {unc_map.min()}, max: {unc_map.max()}, shape: {unc_map.shape}")
            
            costmap_mode = 'features' # empty, height, features, anomaly
            if costmap_mode == 'features':
                # avoid_feature = torch.Tensor([22.887554, 21.481354, 22.915676, 19.23652,  23.831785, 21.27125,  19.956055, 22.428432]).cuda()
                avoid_feature = torch.Tensor([25.197876, 21.696243, 24.205647, 24.736038, 22.71544,  24.884506, 20.79713, 24.430073]).cuda() # radio, grass
                # grass_feature = torch.Tensor([23.964779, 21.991943, 23.726662, 19.904432, 22.468143, 21.320164, 20.323324, 23.249199]).cuda()
                grass_feature = torch.Tensor([25.197876, 21.696243, 24.205647, 24.736038, 22.71544,  24.884506, 20.79713, 24.430073]).cuda() # radio
                # sidewalk_feature = torch.Tensor([23.582233, 22.66328,  16.452255, 22.246119, 24.866558, 21.518925, 22.776405, 20.603878]).cuda() # grey sidewalk
                # sidewalk_feature = torch.Tensor([[23.802433, 22.701805, 18.775259, 22.595041, 23.969284, 21.344238, 21.642178, 20.242914]]).cuda() # sand colored sidewalk
                sidewalk_feature = torch.Tensor([25.016464, 21.799082, 18.736214, 22.177929, 23.700327, 20.390478, 22.705978, 22.43816 ]).cuda() # radio
                gridmap_features = torch.Tensor(gridmap['data'][:8, ...]).cuda()
                avoid_similarity_map = self.pixelwise_euclidean_distance(gridmap_features, avoid_feature).cpu().numpy()
                grass_sim_map = self.pixelwise_euclidean_distance(gridmap_features, grass_feature).cpu().numpy()
                sidewalk_sim_map = self.pixelwise_euclidean_distance(gridmap_features, sidewalk_feature).cpu().numpy()
                
                # self.get_logger().info(f"similarity_map: {similarity_map}")
                # costmap = similarity_map
                
                costmap = self.create_costmap(avoid_similarity_map, grass_sim_map, sidewalk_sim_map)
            elif costmap_mode == 'empty':
                costmap = gridmap['data'][0]
                costmap[:,:] = 0
            elif costmap_mode == 'height':
                height_thresh = 0
                costmap = (gridmap['data'][8] > height_thresh).astype(float) # TODO: ideallly later can query by keys                
            else:
                raise NotImplementedError('costmap mode not yet implemented')
                
            # Save data 
            # np.save('data_gridmap.npy', gridmap['data'])
            # np.save('data_costmap.npy', costmap)
            # np.save('data_unc.npy', unc_map)
            
            # self.get_logger().info(f"Costmap min: {costmap.min()}, costmap max: {costmap.max()}")
            # self.get_logger().info(f"Costmap shape: {costmap.shape}")

            # costmap_msg = self.costmap_to_gridmap(costmap, info, header)
            # uncmap_msg = self.costmap_to_gridmap(unc_map, info, header)
            # self.costmap_pub.publish(costmap_msg)
            # self.uncmap_pub.publish(uncmap_msg)
            
            # Concatenate all maps for publishing 
            all_map = np.concatenate((gridmap['data'], np.expand_dims(costmap, axis=0), np.expand_dims(unc_map, axis=0)), axis=0)
            all_map_msg = self.gridmap_npy_to_rosmsg(all_map, info, header, layer_names=self.feat_names + ['costmap', 'uncertainty'])

            # all_map_msg = self.gridmap_npy_to_rosmsg(np.expand_dims(costmap, axis=0), info, header, layer_names=['costmap'])
            # all_map_msg = self.costmap_to_gridmap(costmap, info, header)
            self.allmap_pub.publish(all_map_msg)
            
            if self.hz_counter == 50000:
                self.hz_counter = 0
            self.hz_counter += 1

            print(time.perf_counter() - now, 'time')
        else:
            print('no odom')
            return

    def create_costmap(self, avoid_sim_map, grass_sim_map, sidewalk_sim_map, 
                    high_cost=1.0, medium_cost=0.5, low_cost=0.0, threshold=7.0):
        """
        Create a costmap based on similarity maps for avoid, grass, and sidewalk features, with distance thresholding.
        
        Args:
            avoid_sim_map (np.ndarray): Similarity map for avoid feature.
            grass_sim_map (np.ndarray): Similarity map for grass feature.
            sidewalk_sim_map (np.ndarray): Similarity map for sidewalk feature.
            high_cost (float): Cost to assign for avoid areas (default is 1.0).
            medium_cost (float): Cost to assign for grass areas (default is 0.5).
            low_cost (float): Cost to assign for sidewalk areas (default is 0.0).
            threshold (float): Threshold for the distance maps (default is 5.0).
        
        Returns:
            costmap (np.ndarray): The resulting costmap.
        """
        # Threshold the similarity maps to the given threshold value
        avoid_sim_map = np.minimum(avoid_sim_map, threshold)
        grass_sim_map = np.minimum(grass_sim_map, threshold)
        sidewalk_sim_map = np.minimum(sidewalk_sim_map, threshold)

        # Normalize similarity maps (invert distances so lower distances correspond to higher costs)
        avoid_norm = 1 - avoid_sim_map / threshold  # Higher similarity to avoid gets higher cost
        grass_norm = 1 - grass_sim_map / threshold  # Higher similarity to grass gets medium cost
        sidewalk_norm = 1 - sidewalk_sim_map / threshold  # Higher similarity to sidewalk gets lower cost

        # Initialize the costmap with .5 
        costmap = np.ones_like(avoid_sim_map) * 0.5

        # Areas similar to sidewalk get low cost
        costmap = sidewalk_norm * low_cost
        
        # Apply cost based on the highest similarity to features
        # Areas similar to avoid get the highest cost
        # costmap += avoid_norm * high_cost

        # Areas similar to grass get medium cost
        costmap += grass_norm * medium_cost

        return costmap
    
    def pixelwise_euclidean_distance(self, input_data, target_vector):
        """
        Calculate pixelwise Euclidean distance between each pixel's feature vector and a target vector.
        
        Args:
            input_data (torch.Tensor or np.ndarray): Input data of shape (C, H, W), where C is the number of feature channels.
            target_vector (torch.Tensor or np.ndarray): A target vector of shape (C,) to calculate distance against.
        
        Returns:
            distance_map (torch.Tensor or np.ndarray): Euclidean distance map of shape (H, W), where each value represents 
                                                    the Euclidean distance between the corresponding pixel and the target vector.
        """
        # If input_data is a NumPy array, convert it to a torch.Tensor
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
        if isinstance(target_vector, np.ndarray):
            target_vector = torch.from_numpy(target_vector)
        
        # Permute input to have shape (H, W, C), where each pixel contains a C-dimensional feature vector
        input_data_perm = input_data.permute(1, 2, 0)  # Shape: (H, W, C)
        
        # Compute the difference between each pixel vector and the target vector
        diff = input_data_perm - target_vector  # Shape: (H, W, C)
        
        # Compute the Euclidean distance (L2 norm) for each pixel
        distance_map = torch.norm(diff, p=2, dim=-1)  # Shape: (H, W)
        
        return distance_map

    def gridmap_npy_to_rosmsg(self, gridmaps_npy, info, header, layer_names=None):
        gridmap_msg = GridMap()
        gridmap_msg.header = header
        gridmap_msg.info = info
        gridmap_msg.layers = layer_names
        
        for layer_name, layer_data in zip(layer_names, gridmaps_npy):
            layer_msg = Float32MultiArray()
            layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="column_index",
                    size=layer_data.shape[0],
                    stride=layer_data.shape[0]
                )
            )
            layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="row_index",
                    size=layer_data.shape[0],
                    stride=layer_data.shape[0] * layer_data.shape[1]
                )
            )
            layer_msg.data = layer_data[::-1, ::-1].flatten().tolist()
            gridmap_msg.data.append(layer_msg)

        gridmap_msg.layers.append('elevation')
        layer_data = np.zeros_like(layer_data) + self.odom_msg.pose.pose.position.z - 1.73
        elevation_layer_msg = Float32MultiArray()
        elevation_layer_msg.layout.dim = layer_msg.layout.dim
        elevation_layer_msg.data = layer_data.flatten().tolist()
        gridmap_msg.data.append(elevation_layer_msg)
        return gridmap_msg

    def costmap_to_gridmap(self, costmap, info, header, costmap_layer='costmap'):
        costmap_msg = GridMap()
        costmap_msg.header = header
        costmap_msg.info = info
        costmap_msg.layers = [costmap_layer]

        costmap_layer_msg = Float32MultiArray()
        costmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=costmap.shape[0],
                stride=costmap.shape[0]
            )
        )
        costmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=costmap.shape[0],
                stride=costmap.shape[0] * costmap.shape[1]
            )
        )
        costmap_layer_msg.data = costmap[::-1, ::-1].flatten().tolist()
        costmap_msg.data.append(costmap_layer_msg)

        costmap_msg.layers.append('elevation')
        layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73
        elevation_layer_msg = Float32MultiArray()
        elevation_layer_msg.layout.dim = costmap_layer_msg.layout.dim
        elevation_layer_msg.data = layer_data.flatten().tolist()
        costmap_msg.data.append(elevation_layer_msg)

        gridmap_cs = (CMAP(costmap / .7) * 255).astype(np.int32)
        gridmap_color = gridmap_cs[..., 0] * (2**16) + gridmap_cs[..., 1] * (2**8) + gridmap_cs[..., 2]
        gridmap_color = gridmap_color.view(dtype=np.float32)

        costmap_msg.layers.append('rgb_viz')
        rgb_viz_msg = Float32MultiArray()
        rgb_viz_msg.layout.dim = costmap_layer_msg.layout.dim
        rgb_viz_msg.data = gridmap_color[::-1, ::-1].flatten().tolist()
        costmap_msg.data.append(rgb_viz_msg)

        return costmap_msg


def main(args=None):
    rclpy.init(args=args)
    node = LethalHeightCost('/zed/map_frame_odom', '/dino_gridmap', '/cherie_costmap', '/cherie_uncmap')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
