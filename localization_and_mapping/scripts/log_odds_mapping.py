import numpy as np

from robotics_workshop.utils.util import read_data, xy_to_rc
from robotics_workshop.utils.visualization import visualize_log_odds_map


def define_map(xmin, xmax, ymin, ymax, resolution):
    MAP = {}
    MAP['res'] = resolution  # meters
    MAP['xmin'] = xmin  # meters
    MAP['ymin'] = ymin
    MAP['xmax'] = xmax
    MAP['ymax'] = ymax
    MAP['sizex'] = int(np.floor((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.floor((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']))

    return MAP


def bresenham2D(sx, sy, ex, ey):
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap

    if dy == 0:
        q = np.zeros((dx+1,1))
    else:
        q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x,y))


if __name__ == '__main__':
    ranges, angles, poses = read_data()

    render_interval = 15
    delta_lambda = 1
    lambda_max = 10
    max_range = 10

    MAP = define_map(xmin=-7.5, xmax=17.5, ymin=-15, ymax=7.5, resolution=0.15)

    for i in range(ranges.shape[0]):

        angles_i = poses[i, 2] + angles[i, :]
        point_cloud_i = poses[i, :2][None].T + ranges[i, :] * np.vstack((np.cos(angles_i), np.sin(angles_i)))

        pose_rc = xy_to_rc(poses[i, :], MAP)
        for ray_ind in range(0, ranges.shape[1], 10):
            if np.isnan(point_cloud_i[0, ray_ind]):
                v_point = poses[i, :2][None].T + max_range *\
                          np.vstack((np.cos(angles_i[ray_ind]), np.sin(angles_i[ray_ind])))
                point_rc = xy_to_rc(v_point, MAP)
                ray_rc = bresenham2D(pose_rc[0], pose_rc[1], point_rc[0], point_rc[1]).astype(int)
                MAP['map'][ray_rc[0, :], ray_rc[1, :]] = np.clip(MAP['map'][ray_rc[0, :], ray_rc[1, :]] -
                                                                 delta_lambda, -lambda_max, lambda_max)
            else:
                point_rc = xy_to_rc(point_cloud_i[:, ray_ind], MAP)
                ray_rc = bresenham2D(pose_rc[0], pose_rc[1], point_rc[0], point_rc[1]).astype(int)
                MAP['map'][ray_rc[0, :-1], ray_rc[1, :-1]] = np.clip(MAP['map'][ray_rc[0, :-1], ray_rc[1, :-1]] -
                                                                     delta_lambda, -lambda_max, lambda_max)
                MAP['map'][ray_rc[0, -1], ray_rc[1, -1]] = np.clip(MAP['map'][ray_rc[0, -1], ray_rc[1, -1]] +
                                                                   delta_lambda, -lambda_max, lambda_max)

        if i % render_interval == 0:
            visualize_log_odds_map(MAP, pose_rc)