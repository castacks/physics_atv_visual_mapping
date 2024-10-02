import yaml
import copy
import rospy
import numpy as np
np.float = np.float64 #hack for numpify

import ros_numpy
import tf2_ros
import torch
import cv_bridge
import rospkg
import os

import distinctipy as COLORS

from sensor_msgs.msg import PointCloud2, CameraInfo, Image, CompressedImage
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.localmapping.localmapping import *
from physics_atv_visual_mapping.utils import *

class DinoMappingNode:
    """
    Hacky implementation of visual mapping node for debug
    """
    def __init__(self, config):
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
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.bridge = cv_bridge.CvBridge()

        self.compressed_img = config['image']['image_compressed']
        if self.compressed_img:
            self.image_sub = rospy.Subscriber(config['image']['image_topic'], CompressedImage, self.handle_img, queue_size=1)
        else:
            self.image_sub = rospy.Subscriber(config['image']['image_topic'], Image, self.handle_img, queue_size=1)

        # self.intrinsics = torch.tensor(rospy.wait_for_message(config['image']['camera_info_topic'], CameraInfo).P, device=config['device']).reshape(3,3).float()
        #use P if using image_rect_color
        self.intrinsics = torch.tensor(config['intrinsics']['P'], device=config['device']).reshape(3,3).float()
        self.Csub = np.array(COLORS.get_colors(32, rng=1))
        self.Csub = np.concatenate([self.Csub,np.zeros([1,3])],axis=0)

        self.dino_intrinsics = None

        self.extrinsics = pose_to_htm(np.concatenate([
            np.array(config['extrinsics']['p']),
            np.array(config['extrinsics']['q'])
        ], axis=-1))

        self.pcl_sub = rospy.Subscriber(config['pointcloud']['topic'], PointCloud2, self.handle_pointcloud, queue_size=1)
        self.odom_sub = rospy.Subscriber(config['odometry']['topic'], PoseWithCovarianceStamped, self.handle_odom, queue_size=10)

        self.pcl_pub = rospy.Publisher('/dino_pcl', PointCloud2, queue_size=1)
        self.gridmap_pub = rospy.Publisher('/dino_gridmap', GridMap, queue_size=1)
        self.image_pub = rospy.Publisher('/dino_image', Image, queue_size=1)

        self.rate = rospy.Rate(10)
        self.viz = config['viz']

    def make_layer_keys(self, layer_keys):
        out = []
        for lk in layer_keys:
            for i in range(lk['n']):
                out.append('{}_{}'.format(lk['key'], i))
        return out

    def handle_pointcloud(self, msg):
        #temp hack
        self.pcl_msg = msg
        self.pcl_msg.header.frame_id = 'lidar'

    def handle_odom(self, msg):
        if self.odom_frame is None:
            self.odom_frame = msg.header.frame_id

        self.odom_msg = msg

    def handle_img(self, msg):
        self.img_msg = msg

    def preprocess_inputs(self):
        """
        Return the update pcl and new metadata
        """
        if self.pcl_msg is None:
            rospy.logwarn_throttle(1.0, 'no pcl msg received')
            return None

        pcl_time = self.pcl_msg.header.stamp.to_sec()
        if abs(pcl_time - self.last_update_time) < 1e-3:
            return None

        if self.odom_msg is None:
            rospy.logwarn_throttle(1.0, 'no odom msg received')
            return None

        if self.img_msg is None:
            rospy.logwarn_throttle(1.0, 'no img msg received')
            return None

        # if self.odom_msg.child_frame_id != self.pcl_msg.header.frame_id:
        #     rospy.logwarn_throttle(1.0, 'for now, need pcls in the child frame of odom (got {}, expected {})'.format(self.pcl_msg.header.frame_id, self.odom_msg.child_frame_id))
        #     return None

        if not self.tf_buffer.can_transform(self.odom_frame, self.pcl_msg.header.frame_id, self.pcl_msg.header.stamp):
            rospy.logwarn_throttle(1.0, 'cant tf from {} to {} at {}'.format(self.odom_frame, self.pcl_msg.header.frame_id, self.pcl_msg.header.stamp))
            return None

        tf_msg = self.tf_buffer.lookup_transform(self.odom_frame, self.pcl_msg.header.frame_id, self.pcl_msg.header.stamp)

        pcl_htm = tf_msg_to_htm(tf_msg).to(self.device)
#        pcl = pcl_msg_to_xyzrgb(self.pcl_msg).to(self.device)
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

        self.last_update_time = self.pcl_msg.header.stamp.to_sec()

        #camera stuff
        if self.compressed_img:
            img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8')/255.
        else:
            img = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8').astype(np.float32)/255.

        # img = torch.tensor(img).unsqueeze(0).permute(0,3,1,2)
        img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)

        dino_img, dino_intrinsics = self.image_pipeline.run(img, self.intrinsics.unsqueeze(0))

        # dino_img = img.to(self.device)
        # dino_intrinsics = self.intrinsics.unsqueeze(0).to(self.device)

        #move back to channels-last
        dino_img = dino_img[0].permute(1,2,0)
        dino_intrinsics = dino_intrinsics[0]

        #need to compute the transform from the odom frame to the image frame
        #do it this way to account for pcl time sync
        # if not self.tf_buffer.can_transform(self.pcl_msg.header.frame_id, self.img_msg.header.frame_id, self.img_msg.header.stamp):
        #     rospy.logwarn_throttle(1.0, 'cant tf from {} to {} at {}'.format(self.pcl_msg.header.frame_id, self.img_msg.header.frame_id, self.img_msg.header.stamp))
        #     return None

        I = get_intrinsics(dino_intrinsics).to(self.device)
        E = get_extrinsics(self.extrinsics).to(self.device)

        # E = torch.linalg.inv(E)
        # print("EXTRINSICS", E)
        # E = torch.Tensor([[ 0.0319, -0.9985, -0.0443, -0.1077],
        # [-0.2237,  0.0361, -0.9740,  0.0741],
        # [ 0.9741,  0.0410, -0.2222, -0.6000],
        # [ 0.0000,  0.0000,  0.0000,  1.0000]]).to(self.device)

        P = obtain_projection_matrix(I, E)

        print(pcl.mean(dim=0))

        pixel_coordinates = get_pixel_from_3D_source(pcl[:, :3], P)
        lidar_points_in_frame, pixels_in_frame, ind_in_frame = get_points_and_pixels_in_frame(
            pcl[:, :3],
            pixel_coordinates,
            dino_img.shape[0],
            dino_img.shape[1],
            height_thresh = 1.5
        )

        dino_features = dino_img[pixels_in_frame[:, 1], pixels_in_frame[:, 0]]
        dino_pcl = torch.cat([pcl_odom[ind_in_frame][:, :3], dino_features], dim=-1)

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
        gridmap_msg.info.header.stamp = self.img_msg.header.stamp
        gridmap_msg.info.header.frame_id = self.odom_frame

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
            gridmap_layer_msg.data = layer_data[::-1, ::-1].T.flatten()
            gridmap_msg.data.append(gridmap_layer_msg)

        #add dummy elevation
        gridmap_msg.layers.append('elevation')
        layer_data = np.zeros_like(gridmap_data[...,0]) + self.odom_msg.pose.pose.position.z - .5
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

        gridmap_layer_msg.data = layer_data.flatten()
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
        print(gridmap_rgb.min(), gridmap_rgb.max())
        vmin = gridmap_rgb.reshape(-1, 3).min(axis=0).reshape(1,1,3)
        vmax = gridmap_rgb.reshape(-1, 3).max(axis=0).reshape(1,1,3)
        gridmap_cs = (gridmap_rgb-vmin)/(vmax-vmin)
        # gridmap_cs = gridmap_rgb
        gridmap_cs = (gridmap_cs*255.).astype(np.int32)

        # gridmap_da = np.argmin(gridmap_data,axis=-1)
        # mins = np.min(gridmap_data,axis=-1)
        # # gridmap_da[mins/26 > .85] = -1
        # gridmap_cs = (self.Csub[gridmap_da]*255).astype(np.int32)

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

        gridmap_layer_msg.data = gridmap_color.T[::-1, ::-1].flatten()
        gridmap_msg.data.append(gridmap_layer_msg)

        return gridmap_msg

    def make_pcl_msg(self, pcl):
        """
        Convert dino pcl into message
        """
        pcl_pos = pcl[:, :3].cpu().numpy()

        pcl_cs = pcl[:, [5,4,3]]
        vmin = pcl_cs.min(dim=0)[0].view(1,3)
        vmax = pcl_cs.max(dim=0)[0].view(1,3)
        pcl_cs =((pcl_cs - vmin) / (vmax - vmin)).cpu().numpy()

        msg = self.xyz_array_to_point_cloud_msg(
            points=pcl_pos,
            frame=self.odom_frame,
            timestamp=self.pcl_msg.header.stamp,
            rgb_values=(pcl_cs*255.).astype(np.uint8)
        )

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
        header = Header()
        header.frame_id = frame
        if timestamp is None:
            timestamp = rospy.Time().now()
        header.stamp = timestamp
        msg = PointCloud2()
        msg.header = header
        if len(points.shape)==3:
            msg.width = points.shape[0]
            msg.height = points.shape[1]
        else:
            msg.width = points.shape[0]
            msg.height = 1
        msg.is_bigendian = False
        # organized clouds are non-dense, since we have to use std::numeric_limits<float>::quiet_NaN()
        # to fill all x/y/z value; un-organized clouds are dense, since we can filter out unfavored ones
        msg.is_dense = False

        if rgb_values is None:
            msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                          PointField('y', 4, PointField.FLOAT32, 1),
                          PointField('z', 8, PointField.FLOAT32, 1), ]
            msg.point_step = 12
            msg.row_step = msg.point_step * msg.width
            xyz = points.astype(np.float32)
            msg.data = xyz.tostring()
        else:
            msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                          PointField('y', 4, PointField.FLOAT32, 1),
                          PointField('z', 8, PointField.FLOAT32, 1),
                          PointField('rgb', 12, PointField.UINT32, 1),]
            msg.point_step = 16
            msg.row_step = msg.point_step * msg.width

            xyzcolor = np.zeros( (points.shape[0], 1), \
            dtype={
                "names": ( "x", "y", "z", "rgba" ),
                "formats": ( "f4", "f4", "f4", "u4" )} )
            xyzcolor["x"] = points[:, 0].reshape((-1, 1))
            xyzcolor["y"] = points[:, 1].reshape((-1, 1))
            xyzcolor["z"] = points[:, 2].reshape((-1, 1))
            color_rgba = np.zeros((points.shape[0], 4), dtype=np.uint8) + 255
            color_rgba[:,:3] = rgb_values[:,:3]
            xyzcolor["rgba"] = color_rgba.view('uint32')
            msg.data = xyzcolor.tostring()

        return msg

    def publish_messages(self, res):
        """
        Publish the dino pcl and dino map
        """
        gridmap_msg = self.make_gridmap_msg(self.localmap)

        t1 = time.time()
        self.gridmap_pub.publish(gridmap_msg)
        print(time.time() - t1)

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

        # # print(res['dino_image'])
        # da = np.argmin(res['dino_image'].cpu().numpy(),axis=-1)
        # # print("max", np.max(res['dino_image'].cpu().numpy()))
        # mins = np.min(res['dino_image'].cpu().numpy(),axis=-1)
        # mins /= 26
        # # gridmap_da[mins/26 > .85] = -1
        # # da[mins < .8] = -1
        # # viz_img = (self.Csub[da]*255).astype(np.int32)
        #
        # print("RES", res['image'].shape)
        #
        # mins[mins < .88] = 0
        # cv2.imshow('mins', mins)
        # cv2.waitKey(1)
        # mins = cv2.resize(mins,res['image'].shape[-2:][::-1], 0,0, interpolation = cv2.INTER_NEAREST)
        # print(mins.shape)
        # viz_img = res['image'][0]
        # mask = mins!=0
        # viz_img[0,mask] *= 2.5
        # viz_img = viz_img.permute(1,2,0).clip(0., 1.).cpu().numpy() * 255

        # img_msg = self.bridge.cv2_to_imgmsg(vimg, "bgr8")
        img_msg = self.bridge.cv2_to_imgmsg(viz_img.astype(np.uint8), "rgb8")
        img_msg.header.stamp = pcl_msg.header.stamp
        self.image_pub.publish(img_msg)

    def spin(self):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['figure.raise_window'] = False
        import time

        if self.viz:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()
            plt.show(block=False)

        while not rospy.is_shutdown():
            rospy.loginfo('spinning...')

            t1 = time.time()
            res = self.preprocess_inputs()
            t2 = time.time()

            if res:
                rospy.loginfo('updating localmap...')

                t3 = time.time()
                self.localmap = self.update_localmap(
                    pcl=res['dino_pcl'],
                    metadata=res['metadata']
                )
                t4 = time.time()

                self.publish_messages(res)

                t5 = time.time()

                rospy.loginfo('Timing:\n\tpreproc: {:.6f}s\n\tlocalmap: {:.6f}s\n\tserialize: {:.6f}s'.format(t2-t1, t4-t3, t5-t4))

                #debug viz
                if self.viz:
                    extent = (
                        self.localmap['metadata']['origin'][0].item(),
                        self.localmap['metadata']['origin'][0].item() + self.localmap['metadata']['length_x'].item(),
                        self.localmap['metadata']['origin'][1].item(),
                        self.localmap['metadata']['origin'][1].item() + self.localmap['metadata']['length_y'].item()
                    )

                    img_extent = (
                        0,
                        res['dino_image'].shape[1],
                        0,
                        res['dino_image'].shape[0],
                    )

                    for ax in axs:
                        ax.cla()

                    vmin = self.localmap['data'][..., :3].view(-1, 3).min(dim=0)[0].view(1,1,3)
                    vmax = self.localmap['data'][..., :3].view(-1, 3).max(dim=0)[0].view(1,1,3)
                    localmap_viz = (self.localmap['data'][..., :3] - vmin) / (vmax-vmin)

                    dino_img_viz = ((res['dino_image'][..., :3]-vmin)/(vmax-vmin)).clip(0., 1.)

                    axs[0].imshow(localmap_viz.permute(1,0,2).cpu(), extent=extent, origin='lower')
                    axs[0].scatter(
                        self.odom_msg.pose.pose.position.x,
                        self.odom_msg.pose.pose.position.y,
                        marker='x',
                        c='r'
                    )
                    axs[0].set_title('Dino map')

                    axs[1].imshow(res['image'], extent=img_extent)
                    axs[1].imshow(dino_img_viz.cpu(), extent=img_extent, alpha=0.5)
                    axs[1].set_title('Dino + FPV')

                    axs[2].imshow(res['image'], extent=img_extent)
                    axs[2].scatter(res['pixel_projection'][:, 0].cpu(), dino_img_viz.shape[0]-res['pixel_projection'][:, 1].cpu(), c='r', s=1., alpha=0.02)
                    axs[2].set_title('pcl projection')

                    dino_pt_cs = ((res['dino_pcl'][::10, 3:6]-vmin[0])/(vmax[0]-vmin[0])).clip(0., 1.)
                    axs[3].scatter(res['dino_pcl'][::10, 0].cpu(), res['dino_pcl'][::10, 1].cpu(), c=dino_pt_cs.cpu(), s=1.)
                    axs[3].set_xlim(extent[0], extent[1])
                    axs[3].set_ylim(extent[2], extent[3])
                    axs[3].set_aspect(1.)
                    axs[3].set_title('dino PCL')

                    plt.pause(1e-2)

            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('visual_mapping')

    config_fp = rospy.get_param("~config_fp")
    config = yaml.safe_load(open(config_fp, 'r'))

    visual_mapping_node = DinoMappingNode(config)

    visual_mapping_node.spin()
