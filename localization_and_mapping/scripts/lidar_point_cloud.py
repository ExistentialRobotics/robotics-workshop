import numpy as np

from robotics_workshop.utils.util import read_data
from robotics_workshop.utils.visualization import visualize_lidar_pc


if __name__ == '__main__':
    ranges, angles, poses = read_data()

    render_interval = 15

    for i in range(ranges.shape[0]):

        angles_i = poses[i, 2] + angles[i, :]
        point_cloud_i = poses[i, :2][None].T + ranges[i, :] * np.vstack((np.cos(angles_i), np.sin(angles_i)))

        if i % render_interval == 0:
            visualize_lidar_pc(point_cloud_i, poses[i, :], i)