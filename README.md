# physics_atv_visual_mapping

Our implementation of voxel and BEV mapping, given lidar and camera.

## Usage

ROS2: ```ros2 launch physics_atv_visual_mapping voxel_localmapping.launch.py```

Python example can be found in ```scripts/offline_processing/postprocess_tartandrive.py```

## Examples

Example config can be found in ```config/ros/super_odometry_integration.yaml```

## ROS Nodes

### VoxelMappingNode

#### Publishers:
|       Topic       | Msg Type |    Description          |
| :---------------- | :------  | :---------------------- |
| ```/dino_image```  | ```sensor_msgs/Image``` | A visualization of the image features extracted from the raw image|
| ```/dino_pcl```  | ```sensor_msgs/PointCloud2``` | The pointcloud, colorized with the VFM feature visualization|
| ```/dino_voxels```  | ```sensor_msgs/PointCloud2``` | The voxel grid with all visual features |
| ```/dino_voxels_viz```  | ```sensor_msgs/PointCloud2``` | The voxel grid with visualization features |
| ```/dino_gridmap```  | ```grid_map_msgs/GridMap``` | A BEV-projection of the voxel grid. Only publishes if terrain estimation is enabled |

#### Subscribers:

|       Topic       | Msg Type |    Description          |
| :---------------- | :------  | :---------------------- |
| ```image_topic```  | ```sensor_msgs/{Image/CompressedImage}``` | The topic to get images from (configurable)|
| ```camera_info_topic```  | ```sensor_msgs/CameraInfo``` | The topic to get intrinsics from (configurable)|
| ```pointcloud_topic```  | ```sensor_msgs/PointCloud2``` | The topic to get pointclouds from (configurable)|
| ```odometry_topic```  | ```nav_msgs/Odometry``` | The topic to get odometry from (configurable). The map will be in this message's base frame|