import numpy as np
import os, cv2


def read_data():
    file_address = os.path.abspath(os.path.join("", os.pardir)) + "/data/"
    ranges = np.load(file_address + 'ranges.npy', allow_pickle=True)
    angles = np.load(file_address + 'angles.npy', allow_pickle=True)
    poses = np.load(file_address + 'poses.npy', allow_pickle=True)
    return ranges, angles, poses


def read_velocity_data():
    file_address = os.path.abspath(os.path.join("", os.pardir)) + "/data/"
    v = np.load(file_address + 'linear_velocities.npy', allow_pickle=True)
    omega = np.load(file_address + 'angular_velocities.npy', allow_pickle=True)
    return v, omega


def load_map(xmin, xmax, ymin, ymax, resolution):
    file_address = os.path.abspath(os.path.join("", os.pardir)) + "/data/"
    occ_map = cv2.imread(file_address + 'occ_map.png', 0)

    MAP = {}
    MAP['res'] = resolution  # meters
    MAP['xmin'] = xmin  # meters
    MAP['ymin'] = ymin
    MAP['xmax'] = xmax
    MAP['ymax'] = ymax
    MAP['sizex'] = int(np.floor((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.floor((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = occ_map

    return MAP


def obtain_velocities(tau):
    _, _, poses = read_data()
    omega = (poses[1:, 2] - poses[:-1, 2]) / tau
    v = np.diag((poses[1:, :2] - poses[:-1, :2]) @ np.array([np.cos(poses[:-1, 2]), np.sin(poses[:-1, 2])])) / tau

    np.save('linear_velocities.npy', v)
    np.save('angular_velocities.npy', omega)


def diff_drive(pose, v, omega, tau):
    new_x = pose[0] + tau * v * np.cos(pose[2])
    new_y = pose[1] + tau * v * np.sin(pose[2])
    new_theta = pose[2] + tau * omega
    return np.array([new_x, new_y, new_theta])


def xy_to_rc(pose, MAP):
    r = int(np.floor((pose[0] - MAP['xmin']) / MAP['res'] + 1))
    c = int(np.floor((pose[1] - MAP['ymin']) / MAP['res'] + 1))
    if pose.shape[0] > 2:
        return np.array([r, c, pose[2]])
    else:
        return np.array([r, c])


def xy_to_rc_multiple(poses, MAP):
    r = np.floor((poses[:, 0] - MAP['xmin']) / MAP['res'] + 1).astype(int)
    c = np.floor((poses[:, 1] - MAP['ymin']) / MAP['res'] + 1).astype(int)
    if poses.shape[1] == 3:
        return np.vstack((r, c, poses[:, 2])).T
    else:
        return np.vstack((r, c)).T


if __name__ == '__main__':
    obtain_velocities(0.1)
