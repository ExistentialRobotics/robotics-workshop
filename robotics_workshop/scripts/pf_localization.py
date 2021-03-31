import numpy as np

from robotics_workshop.utils.util import read_data, read_velocity_data, load_map, diff_drive, \
    xy_to_rc, xy_to_rc_multiple
from robotics_workshop.utils.visualization import visualize_pf_localization


def prediction(particles, v, omega, std_v, std_omega, tau):
    N = particles.shape[0]
    n_v = np.random.normal(0, std_v, size=N)
    n_omega = np.random.normal(0, std_omega, size=N)

    for k in range(N):
        particles[k, :3] = diff_drive(particles[k, :3], v + n_v[k], omega + n_omega[k], tau)

    return particles


def update(particles, ranges, angles, MAP):
    occ_map = np.zeros((MAP['sizex'], MAP['sizey']))
    np.place(occ_map, MAP['map'] == 0, 1)
    corr_list = []
    for k in range(particles.shape[0]):
        angles_k = particles[k, 2] + angles
        point_cloud_k = particles[k, :2][None].T + ranges * np.vstack((np.cos(angles_k), np.sin(angles_k)))
        point_cloud_k_rc = xy_to_rc_multiple(point_cloud_k.T, MAP)
        good_inds = np.logical_and(np.logical_and(point_cloud_k_rc[:, 0] >= 0,
                                   point_cloud_k_rc[:, 0] < MAP['sizex']), np.logical_and(point_cloud_k_rc[:, 1] >= 0,
                                   point_cloud_k_rc[:, 1] < MAP['sizey']))
        corr = np.sum(occ_map[point_cloud_k_rc[good_inds, 0], point_cloud_k_rc[good_inds, 1]]) / 5
        corr_list.append(corr)
        particles[k, 3] = np.exp(corr) * particles[k, 3]

    particles[:, 3] = particles[:, 3] / np.sum(particles[:, 3])

    return particles


def resample(particles):
    rand_choice = np.random.choice(np.arange(particles.shape[0]), size=particles.shape[0], p=particles[:, 3])
    particles[:, :3] = particles[rand_choice, :3]
    particles[:, 3] = 1 / particles.shape[0]
    return particles


if __name__ == '__main__':
    np.random.seed(3)

    ranges, angles, poses = read_data()
    v, omega = read_velocity_data()
    MAP = load_map(xmin=-7.5, xmax=17.5, ymin=-15, ymax=7.5, resolution=0.15)

    render_interval = 5
    tau = 0.1
    std_v = 3
    std_omega = 0.03
    N = 100
    N_eff = 2

    particles = np.empty((N, 4))
    particles[:, 0] = poses[0, 0]
    particles[:, 1] = poses[0, 1]
    particles[:, 2] = poses[0, 2]
    particles[:, 3] = 1 / N

    particles_rc = xy_to_rc_multiple(particles[:, :3], MAP)
    gt_pose_rc = xy_to_rc(poses[0, :], MAP)
    best_ind = np.argmax(particles[:, 3])
    visualize_pf_localization(MAP, particles_rc[:, :2], gt_pose_rc, best_ind)

    for i in range(poses.shape[0] - 1):
        particles = prediction(particles, v[i], omega[i], std_v, std_omega, tau)
        particles = update(particles, ranges[i], angles[i], MAP)

        if (1 / np.sum(particles[:, 3]**2)) < N_eff:
            particles = resample(particles)
            print('Resampling...')

        if i % render_interval == 0:

            particles_rc = xy_to_rc_multiple(particles[:, :3], MAP)
            gt_pose_rc = xy_to_rc(poses[i + 1, :], MAP)
            best_ind = np.argmax(particles[:, 3])
            visualize_pf_localization(MAP, particles_rc[:, :2], gt_pose_rc, best_ind)