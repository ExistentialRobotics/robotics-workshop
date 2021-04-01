from robotics_workshop.utils.util import read_data, read_velocity_data, load_map, diff_drive, xy_to_rc
from robotics_workshop.utils.visualization import visualize_dead_reckoning


if __name__ == '__main__':
    _, _, poses = read_data()
    v, omega = read_velocity_data()
    MAP = load_map(xmin=-7.5, xmax=17.5, ymin=-15, ymax=7.5, resolution=0.15)

    render_interval = 5
    tau = 0.1

    robot_pose = poses[0, :]
    pred_pose_rc = xy_to_rc(robot_pose, MAP)
    gt_pose_rc = xy_to_rc(poses[0, :], MAP)
    visualize_dead_reckoning(MAP, pred_pose_rc, gt_pose_rc)

    for i in range(poses.shape[0] - 1):
        robot_pose = diff_drive(robot_pose, v[i], omega[i], tau)

        if i % render_interval == 0:
            pred_pose_rc = xy_to_rc(robot_pose, MAP)
            gt_pose_rc = xy_to_rc(poses[i + 1, :], MAP)
            visualize_dead_reckoning(MAP, pred_pose_rc, gt_pose_rc)
