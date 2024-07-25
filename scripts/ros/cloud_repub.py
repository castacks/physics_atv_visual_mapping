#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
# import sensor_msgs.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt

MPLIB_CMAP = plt.get_cmap('jet')


type_mappings = [(PointField.INT8, np.dtype('int8')),
                 (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')),
                 (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')),
                 (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
# print(pftype_to_nptype)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)
# prefix to the names of dummy fields we add to get byte alignment
# correct. this needs to not clash with any actual field names
DUMMY_FIELD_PREFIX  = '__'
TOTAL_TIME_SYNCS    = 0

def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    translation_matrix = np.array([[1, 0, 0, x],
                                   [0, 1, 0, y],
                                   [0, 0, 1, z],
                                   [0, 0, 0, 1]])

    rotation_matrix = np.array([[np.cos(yaw) * np.cos(pitch),
                                 np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
                                 np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll),
                                 0],
                                [np.sin(yaw) * np.cos(pitch),
                                 np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
                                 np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll),
                                 0],
                                [-np.sin(pitch),
                                 np.cos(pitch) * np.sin(roll),
                                 np.cos(pitch) * np.cos(roll),
                                 0],
                                [0, 0, 0, 1]])

    transformation_matrix = np.dot(translation_matrix, rotation_matrix)
    return transformation_matrix

# Function to transform a point cloud using a transformation matrix
def transform_pointcloud(pointcloud, transformation_matrix):
    # Assuming the input pointcloud is Nx3 (x, y, z) numpy array
    # Add a column of ones to represent homogeneous coordinates
    num_points = pointcloud.shape[0]
    homogeneous_coords = np.hstack((pointcloud, np.ones((num_points, 1))))

    # Perform the transformation by multiplying the point cloud with the transformation matrix
    transformed_pointcloud = np.dot(transformation_matrix, homogeneous_coords.T).T

    # Extract the transformed (x, y, z) coordinates
    transformed_xyz = transformed_pointcloud[:, :3]
    return transformed_xyz

V2_V1 = create_transformation_matrix(-0.06511276, -0.00034881, 0.21072615 ,.000146970671, .168982719 ,-.0103227238)
L_V1 = create_transformation_matrix(-0.14460644, 0.09369309, -0.21868021 ,0.01003703, 0.22189415,-0.01221614 )

CM_V1 = create_transformation_matrix(0.01784539, -0.09222747, -0.38195659, 3.141501405045616081, -1.3483089608851129, -1.5401517876898885)



# print(CM_V1.tolist())

def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(
                ('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        if f.datatype == 0:
            continue
        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_to_nptype[f.datatype].itemsize * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list

def dtype_to_fields(dtype):
    '''Convert a numpy record datatype into a list of PointFields.
    '''
    fields = []
    for field_name in dtype.names:
        np_field_type, field_offset = dtype.fields[field_name]
        pf = PointField()
        pf.name = field_name
        if np_field_type.subdtype:
            item_dtype, shape = np_field_type.subdtype
            pf.count = int(np.prod(shape))
            np_field_type = item_dtype
        else:
            pf.count = 1

        pf.datatype = nptype_to_pftype[np_field_type]
        pf.offset = field_offset
        fields.append(pf)
    return fields

def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray
​
    Reshapes the returned array to have shape (height, width), even if the
    height is 1.
​
    The reason for using np.frombuffer rather than struct.unpack is
    speed... especially for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    # print(cloud_msg.fields)
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (
            fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))
DTYPE = np.dtype(
            [(name, np.float32) for name in ("x", "y", "z", "intensity", "ring", "time")]
        )

pub1 = rospy.Publisher("/full_cloud", PointCloud2, queue_size=10)
pub2 = rospy.Publisher("/newv2", PointCloud2, queue_size=10)
pub3 = rospy.Publisher("/newl", PointCloud2, queue_size=10)
fake_pub = rospy.Publisher("/livox/lidar_fake", PointCloud2, queue_size=10)



def np_points_to_pointcloud2(points, header,color=True):
        """ Creates a point cloud message. Modified from https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
        Args:
            points: Nx3 array of xyz positions (m)
            header: Header msg we want to use for published pointcloud2
        Returns:
            sensor_msgs/PointCloud2 message
        """
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = points.astype(dtype).tobytes()

        if color:
            fields = [PointField(
                name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
                for i, n in enumerate('xyzrgba')]

            return PointCloud2(
                header=header,
                height=1,
                width=points.shape[0],
                is_dense=False,
                is_bigendian=False,
                fields=fields,
                point_step=(itemsize * 7),
                row_step=(itemsize * 7 * points.shape[0]),
                data=data
            )
        if points.shape[1] == 4:
            # print('intensity')
            fields = [PointField(
                name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
                for i, n in enumerate('xyzi')]

            return PointCloud2(
                header=header,
                height=1,
                width=points.shape[0],
                is_dense=False,
                is_bigendian=False,
                fields=fields,
                point_step=(itemsize * 4),
                row_step=(itemsize * 4 * points.shape[0]),
                data=data
            )
        else:
            fields = [PointField(
                name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
                for i, n in enumerate('xyz')]
            # print('here')
            return PointCloud2(
                header=header,
                height=1,
                width=points.shape[0],
                is_dense=False,
                is_bigendian=False,
                fields=fields,
                point_step=(itemsize * 3),
                row_step=(itemsize * 3 * points.shape[0]),
                data=data
            )

V1 = np.zeros((3,4))
V2 = np.zeros((3,4))
L = np.zeros((3,4))

def pointcloud_callback1(msg):
    global V1,V2,L
    pc = pointcloud2_to_array(msg)
    pts=np.ones((pc.shape[0],4),dtype=np.float32)
    pts[:,0]=pc['x']
    pts[:,1]=pc['y']
    pts[:,2]=pc['z']
    pts[:,3]=pc['intensity']
    # print(pts.shape)
    V1 = pts[:,:4]

    ids = np.where(np.linalg.norm(V1[:,:3],axis=1) > 2)[0]
    V1 = V1[ids]

    v2 = np.copy(V2)
    v2[:,:3] = transform_pointcloud(v2[:,:3],V2_V1)

    l = np.copy(L)
    l[:,:3] = transform_pointcloud(l[:,:3],L_V1)

    V1 = np.vstack([V1,v2,l])
    # V1 = np.vstack([V1,v2])

    # V1 = np.vstack([V1,V2,L])
    # V1 = l

    out_msg = np_points_to_pointcloud2(V1,msg.header,color=False)
    # fake_msg = np_points_to_pointcloud2(V1[0].reshape(1,-1),msg.header,color=False)

    pub1.publish(out_msg)
    # fake_pub.publish(fake_msg)
    # print('hereee')

def pointcloud_callback2(msg):
    global V2
    pc = pointcloud2_to_array(msg)
    pts=np.ones((pc.shape[0],4),dtype=np.float32)
    pts[:,0]=pc['x']
    pts[:,1]=pc['y']
    pts[:,2]=pc['z']
    pts[:,3]=pc['intensity']
    V2 = pts

    V2 = pts[:,:4]

    # print('Velodyne ',V2[:,3].min(),V2[:,3].max())

    ids = np.where(np.linalg.norm(V2[:,:3],axis=1) > 2)[0]
    V2 = V2[ids]



    # out_msg = np_points_to_pointcloud2(V2,msg.header,color=False)
    #
    # pub2.publish(out_msg)

def pointcloud_callback3(msg):
    global L
    pc = pointcloud2_to_array(msg)
    pts=np.ones((pc.shape[0],4),dtype=np.float32)
    pts[:,0]=pc['x']
    pts[:,1]=pc['y']
    pts[:,2]=pc['z']
    pts[:,3]=pc['intensity']
    L = pts

    L = pts[:,:4]



    ids = np.where(np.linalg.norm(L[:,:3],axis=1) > 2)[0]
    L = L[ids]


    # ids = np.where(L[:,3] > 150)[0]
    # L[L[:,3] <=150,3] *= (2./3.)
    # L[ids,3] -= 150
    # L[ids,3] *= (3./2.)
    # L[ids,3] += 100

    # ids = np.where(L[:,3] <= 150)[0]
    # L = L[ids]
    # L *= 2./3.

    # L[:,3] = np.rint(L[:,3])

    # print('Livox ',L[:,3].min(),L[:,3].max())

    # L[L[:,3] > 150,3] = 200
    # L = L[L[:,3] > 150]

    # L[:,:3] = transform_pointcloud(L[:,:3],L_V1)

    # out_msg = np_points_to_pointcloud2(L,msg.header,color=False)
    #
    # pub3.publish(out_msg)

def main():
    rospy.init_node('pointcloud_subscriber')

    # Subscribe to the three PointCloud topics
    rospy.Subscriber("/velodyne_1/velodyne_points", PointCloud2, pointcloud_callback1)
    # rospy.Subscriber("/velodyne_2/velodyne_points", PointCloud2, pointcloud_callback2)
    rospy.Subscriber("/livox/lidar", PointCloud2, pointcloud_callback3)



    # Spin the node to keep it running
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
