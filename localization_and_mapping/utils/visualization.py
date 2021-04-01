import numpy as np
import matplotlib.pyplot as plt


plt.ion()
fig, ax = plt.subplots(figsize=(7, 7))

UNEXPLORED = 127
OCCUPIED = 0
FREE = 255

def visualize_lidar_pc(point_cloud, pose, i):
    good_inds = np.logical_not(np.isnan(point_cloud[0, :]))
    point_cloud = point_cloud[:, good_inds]
    if i == 0:
        ax.scatter(point_cloud[0, :], point_cloud[1, :], c='blue', s=0.1)

        ax.scatter(pose[0], pose[1], c='red', s=10, alpha=0.25)
        ax.plot(pose[0] + np.array([0, 0.5 * np.cos(pose[2])]),
                pose[1] + np.array([0, 0.5 * np.sin(pose[2])]), c='red', alpha=0.25)

        plt.xlim(-7.5, 17.5)
        plt.ylim(-15, 7.5)

        plt.draw()
    else:
        ax.scatter(point_cloud[0, :], point_cloud[1, :], c='blue', s=0.1)

        ax.scatter(pose[0], pose[1], c='red', s=10, alpha=0.25)
        ax.plot(pose[0] + np.array([0, 0.5 * np.cos(pose[2])]),
                pose[1] + np.array([0, 0.5 * np.sin(pose[2])]), c='red', alpha=0.25)

        fig.canvas.draw_idle()
        plt.pause(0.1)


def visualize_log_odds_map(MAP, pose_rc):
    occ_map = UNEXPLORED * np.ones((MAP['sizex'], MAP['sizey']), dtype=np.uint8)
    np.place(occ_map, MAP['map'] > 0, OCCUPIED)
    np.place(occ_map, MAP['map'] < 0, FREE)

    ax.imshow(np.flipud(occ_map.T), cmap='gray')

    ax.scatter(pose_rc[0], MAP['sizey'] - pose_rc[1], c='red', s=10, alpha=0.25)
    ax.plot(pose_rc[0] + np.array([0, 3.3 * np.cos(pose_rc[2])]),
            MAP['sizey'] - pose_rc[1] - np.array([0, 3.3 * np.sin(pose_rc[2])]), c='red', alpha=0.25)

    plt.show()
    plt.pause(0.1)


def visualize_dead_reckoning(MAP, pred_pose_rc, gt_pose_rc):
    ax.clear()
    ax.imshow(np.flipud(MAP['map'].T), cmap='gray')

    ax.scatter(gt_pose_rc[0], MAP['sizey'] - gt_pose_rc[1], c='red', s=10, alpha=0.25)
    ax.plot(gt_pose_rc[0] + np.array([0, 3.3 * np.cos(gt_pose_rc[2])]),
            MAP['sizey'] - gt_pose_rc[1] - np.array([0, 3.3 * np.sin(gt_pose_rc[2])]), c='red', alpha=0.25)

    ax.scatter(pred_pose_rc[0], MAP['sizey'] - pred_pose_rc[1], c='blue', s=10, alpha=0.25)
    ax.plot(pred_pose_rc[0] + np.array([0, 3.3 * np.cos(pred_pose_rc[2])]),
            MAP['sizey'] - pred_pose_rc[1] - np.array([0, 3.3 * np.sin(pred_pose_rc[2])]), c='blue', alpha=0.25)

    plt.show()
    plt.pause(0.1)


def visualize_pf_localization(MAP, particles_rc, gt_pose_rc, best_ind):
    ax.clear()
    ax.imshow(np.flipud(MAP['map'].T), cmap='gray')

    ax.scatter(gt_pose_rc[0], MAP['sizey'] - gt_pose_rc[1], c='red', s=10, alpha=0.25)
    ax.plot(gt_pose_rc[0] + np.array([0, 3.3 * np.cos(gt_pose_rc[2])]),
            MAP['sizey'] - gt_pose_rc[1] - np.array([0, 3.3 * np.sin(gt_pose_rc[2])]), c='red', alpha=0.25)

    ax.scatter(particles_rc[:, 0], MAP['sizey'] - particles_rc[:, 1], c='blue', s=0.5, alpha=0.25)
    ax.scatter(particles_rc[best_ind, 0], MAP['sizey'] - particles_rc[best_ind, 1], c='green', s=10, alpha=1)

    plt.show()
    plt.pause(0.1)