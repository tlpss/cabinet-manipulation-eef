import numpy as np 
from airo_typing import NumpyDepthMapType, NumpyFloatImageType, CameraIntrinsicsMatrixType, ColoredPointCloudType

def create_pointcloud_from_depth_map(depth_map, rgb, intrinsics_matrix):
    # for every pixel in the image, get the corresponding 3D point
    diy_pointcloud_points = []
    diy_pointcloud_colors = []
    inverse_intrinsics = np.linalg.inv(intrinsics_matrix)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            # get the 3D point
            image_ray = np.array([j,i,1])
            camera_ray = inverse_intrinsics @ image_ray
            t = depth_map[i,j] / camera_ray[2]
            point = t * camera_ray
            diy_pointcloud_points.append(point)
            diy_pointcloud_colors.append(rgb[i,j,:])
    
    # combine the points and colors into a single array
    diy_pointcloud_points = np.array(diy_pointcloud_points)
    diy_pointcloud_colors = np.array(diy_pointcloud_colors)
    diy_pointcloud = np.concatenate((diy_pointcloud_points, diy_pointcloud_colors), axis=1)
    return diy_pointcloud