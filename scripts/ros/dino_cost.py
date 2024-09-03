#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
# from learned_cost_map.msg import FloatStamped
import numpy as np
import rospkg
from threading import Lock

import scipy
import scipy.signal
from scipy.signal import welch
from scipy.integrate import simps
from cv_bridge import CvBridge
import os
import yaml

import roslib
# roslib.load_manifest('learning_tf')
import rospy
import numpy as np
import math
# import tf

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32, Float32, Float32MultiArray
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32

from grid_map_msgs.msg import GridMap

# import skimage
import time
import cv2
import yaml
from yaml.loader import SafeLoader
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal, stats

from matplotlib.animation import FuncAnimation, ArtistAnimation
from rosbag_to_dataset.dtypes.gridmap import GridMapConvert

import matplotlib

# CMAP = matplotlib.cm.get_cmap('plasma')
CMAP = matplotlib.cm.get_cmap('magma')

class LethalHeightCost(object):
    def __init__(self,odom_topic , costmap_topic):

        self._lock = Lock()

        rp = rospkg.RosPack()


        self.odom_msg = None

        self.hz_counter = 0

        # rospy.Subscriber(gridmap_topic, GridMap, self.handle_map, queue_size=1)

        rospy.Subscriber(odom_topic, Odometry, self.handle_odom, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0/10.0), self.run_map)
        self.image = None
        self.new_msg = False
        self.cost = 0.0

        self.costmap_pub = rospy.Publisher(costmap_topic,GridMap,queue_size=2)
        # self.max_vel_pub = rospy.Publisher(vel_pub_topic,Float32,queue_size=2)


        self.channels = []
        self.grid_map_cvt = GridMapConvert(channels=self.channels, size=[1, 1])

        print('DONE WITH INIT')


    def update_plot(self, frame):
        # self.ln.set_data(self.x_data, self.y_data)
        # return self.ln
        return

    def handle_odom(self, msg):
        self.velocity = np.linalg.norm([msg.twist.twist.linear.x,msg.twist.twist.linear.y])
        self.odom_msg = msg

    def handle_map(self,msg):
        print("handling map")
        with self._lock:
            self.dino_map = msg
            self.new_msg = True

            # print(msg.layers)
            if len(self.channels) == 0:
                for layer in msg.layers:
                    if 'height' in layer:
                        self.channels.append(layer)
                    if 'VLAD' in layer:
                        self.channels.append(layer)

                self.grid_map_cvt.channels = self.channels

    def run_map(self, event):
        now = time.perf_counter()
        print('----')
        if self.odom_msg is not None:

            # if not self.new_msg:
            #     print('no new map')
            #     return

            # with self._lock:
            #     print('got map')
            #     if self.dino_map is not None:
            #         # lx = self.dino_map.info.length_x
            #         # ly = self.dino_map.info.length_y
            #         # res = self.dino_map.info.resolution
            #         # origin = self.dino_map.info.pose
            #         #
            #         # # idx = self.dino_map.layers.index('terrain')
            #         # data = self.dino_map.data
            #         #
            #         # nx = data.layout.dim[0].size
            #         # ny = data.layout.dim[1].size
            #         #
            #         # map_data = np.copy(np.array(data.data).reshape(nx, ny)[::-1, ::-1])
            #         #

            #         info = self.dino_map.info
            #         nx = int(info.length_x / info.resolution)
            #         ny = int(info.length_y / info.resolution)
            #         self.grid_map_cvt.size = [nx, ny]

            #         gridmap = self.grid_map_cvt.ros_to_numpy(self.dino_map)

            #         msg_header = self.dino_map.info.header
            #         self.new_msg = False
            #     else:
            #         gridmap = None

            # if gridmap is None:
            #     print("NO MAP")
            #     return

            # # da = gridmap['data'].argmin(axis=0)
            # # unc_map = gridmap['data'].min(axis=0)

            # # unc_map /= 26
            # # unc_map[unc_map < .82] = 0

            # # P = gridmap['metadata']['origin']
            # # res = gridmap['metadata']['resolution']
            # # import pdb; pdb.set_trace()
            # gridmap_height = gridmap['data'][0] # featured used for thresholding
            # height_threshold = -0.2 # TODO: need to handle height of camera
            # costmap = gridmap_height > height_threshold
            # # ids = np.where(unc_map != 0)
            # # costmap[ids] = unc_map[ids]

            # # costmap[xloc,yloc] = 1
            gridmap_msg = GridMap()
            gridmap_msg.info.header.stamp = rospy.Time.now()
            gridmap_msg.info.header.frame_id = 'odom'
            resolution = 0.1
            length_x = 25
            length_y = 25

            # if self.layer_keys is None:
            #     gridmap_msg.layers = ['{}_{}'.format(self.layer_key, i) for i in range(gridmap_data.shape[-1])]
            # else:
            #     gridmap_msg.layers = copy.deepcopy(self.layer_keys)
                

            gridmap_msg.info.resolution = resolution
            gridmap_msg.info.length_x = length_x
            gridmap_msg.info.length_y = length_y
            gridmap_msg.info.pose.position.x = self.odom_msg.pose.pose.position.x - 0.5*gridmap_msg.info.length_x
            gridmap_msg.info.pose.position.y = self.odom_msg.pose.pose.position.y - 0.5*gridmap_msg.info.length_y
            gridmap_msg.info.pose.position.z = 0
            gridmap_msg.info.pose.orientation.w = 1.
            
            costmap = np.zeros((int(length_x/resolution), int(length_y/resolution)))
            gridmap_info = gridmap_msg.info

            costmap_msg = self.costmap_to_gridmap(costmap, gridmap_info)
            self.costmap_pub.publish(costmap_msg)




            if self.hz_counter == 50000:
                self.hz_counter = 0
            self.hz_counter += 1
            # if self.hz_counter % 4 != 0:
            #     return


            print(time.perf_counter() - now, 'time')
            
        else:
            print('no odom')
            return

    def costmap_to_gridmap(self, costmap, info, costmap_layer='costmap'):
        """
        convert costmap into gridmap msg

        Args:
            costmap: The data to load into the gridmap
            msg: The input msg to extrach metadata from
            costmap: The name of the layer to get costmap from
        """
        costmap_msg = GridMap()
        costmap_msg.info = info
        # print("FRAME _ ", costmap_msg.info.header.frame_id)
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

        costmap_layer_msg.data = costmap[::-1, ::-1].flatten()
        costmap_msg.data.append(costmap_layer_msg)

        #add dummy elevation
        costmap_msg.layers.append('elevation')
        layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73 #+ costmap
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
        costmap_msg.data.append(gridmap_layer_msg)
        gridmap_cs = (CMAP(costmap/.7) * 255).astype(np.int32)
        gridmap_color = gridmap_cs[..., 0] * (2**16) + gridmap_cs[..., 1] * (2**8) + gridmap_cs[..., 2]
        gridmap_color = gridmap_color.view(dtype=np.float32)

        costmap_msg.layers.append('rgb_viz')
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

        gridmap_layer_msg.data = gridmap_color[::-1, ::-1].flatten()
        costmap_msg.data.append(gridmap_layer_msg)

        return costmap_msg

def main():
    rospy.init_node("context_publisher", log_level=rospy.INFO)
    rospy.loginfo("Initialized lethal height publisher node")
    odom_topic = '/odom' #rospy.get_param("~odom_topic")
    # gridmap_topic = '/dino_gridmap' #rospy.get_param("~gridmap_topic")
    # viz = rospy.get_param("~viz")
    costmap_topic = '/shortrange_costmap' #rospy.get_param("~costmap_topic")
    # vel_pub_topic = # rospy.get_param("~vel_pub_topic")

    rp = rospkg.RosPack()
    node = LethalHeightCost(odom_topic , costmap_topic)
    rate = rospy.Rate(10)

    # if viz:
    #     print("+++++++++++++++++++++++++++++++")
    #     ani = FuncAnimation(node.fig, node.update_plot)
    #     # ani = ArtistAnimation(node.fig, node.ax)
    #     plt.show(block=True)

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == "__main__":
    main()