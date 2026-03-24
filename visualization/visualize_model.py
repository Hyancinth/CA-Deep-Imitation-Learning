import matplotlib.pyplot as plt

def plot_train_test_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

def plot_ee_trajectories(ee_positions_gt, ee_positions_pred, goal, obstacle, link_pos, a):
    plt.figure(figsize=(8, 8))
    plt.plot(ee_positions_gt[:, 0], ee_positions_gt[:, 1], label='Ground Truth Trajectory', marker='o')
    plt.plot(ee_positions_pred[:, 0], ee_positions_pred[:, 1], label='Predicted Trajectory', marker='x')
    plt.plot([0, link_pos[0]], [0, link_pos[1]], color='blue', lw=3)
    plt.plot([link_pos[0], ee_positions_gt[0, 0]], [link_pos[1], ee_positions_gt[0, 1]], color='blue', lw=3)
    plt.scatter(goal[0], goal[1], color='green', label='Target', s=100)
    plt.scatter(obstacle[0], obstacle[1], color='red', label='Obstacle', s=100)
    plt.hlines(0, -3, 3, colors='black')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('End-Effector Trajectories')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()