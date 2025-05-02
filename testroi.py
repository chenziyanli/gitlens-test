#!/usr/bin/env python3
import rospy
import numpy as np
import math
import random
import traceback
import sys
from functools import partial
from collections import defaultdict
import time # 用于计时

# ROS Msgs/Srvs
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
try:
    from scipy.ndimage import binary_dilation, generate_binary_structure
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    print("Warning: SciPy library not found. C-Space calculation for A* will be unavailable.", file=sys.stderr)
try:
    from pathfinding.core.grid import Grid
    from pathfinding.finder.a_star import AStarFinder
    from pathfinding.core.diagonal_movement import DiagonalMovement
    _PATHFINDING_AVAILABLE = True
except ImportError:
    _PATHFINDING_AVAILABLE = False
    print("ERROR: python-pathfinding library not found! Install using: pip install python-pathfinding", file=sys.stderr)


def calculate_cspace(occupancy_grid, map_resolution, robot_radius_m, obstacle_threshold=50):
    """(辅助函数) 计算配置空间"""
    if not _SCIPY_AVAILABLE or occupancy_grid is None or map_resolution <= 0:
        rospy.logwarn("Cannot calculate C-Space: SciPy not available, grid missing, or invalid resolution.")
        return None

    if robot_radius_m <= 0:
        rospy.logwarn("Robot radius is zero or negative, using original occupancy grid as C-Space.")
        # 将障碍物和未知区域标记为 True (障碍)
        return (occupancy_grid >= obstacle_threshold) | (occupancy_grid == -1)

    robot_radius_cells = int(math.ceil(robot_radius_m / map_resolution))
    rospy.loginfo(f"Calculating C-Space with radius {robot_radius_m}m ({robot_radius_cells} cells)...")

    obstacle_mask = (occupancy_grid >= obstacle_threshold) | (occupancy_grid == -1)
    print(f"obstacle_mask:\n{obstacle_mask}")
    struct = generate_binary_structure(2, 2) # 8-connectivity
    print(f"struct:\n{struct}")

    try:
        start_time = time.time()
        cspace_grid = binary_dilation(obstacle_mask, structure=struct, iterations=robot_radius_cells)
        end_time = time.time()
        rospy.loginfo(f"cspace_grid {cspace_grid}")
        rospy.loginfo(f"C-Space calculation took {end_time - start_time:.3f} seconds.")
        return cspace_grid # 返回布尔型 NumPy 数组 (True=Obstacle)
    except Exception as e:
        rospy.logerr(f"Error during C-Space calculation (binary_dilation): {e}")
        return None

test_map_10x10 = np.zeros((10, 10), dtype=np.int8)

# 设置边界为未知区域 (-1)
test_map_10x10[0, :] = -1  # 第一行
test_map_10x10[-1, :] = -1 # 最后一行 (索引-1)
test_map_10x10[:, 0] = -1  # 第一列
test_map_10x10[:, -1] = -1 # 最后一列 (索引-1)

# 在中心区域设置一个 3x3 的障碍物 (100)
# 注意 Python 切片是左闭右开，所以 4:7 包含索引 4, 5, 6
obstacle_row_start, obstacle_row_end = 4, 7
obstacle_col_start, obstacle_col_end = 4, 7
test_map_10x10[obstacle_row_start:obstacle_row_end, obstacle_col_start:obstacle_col_end] = 100
if __name__ == "__main__":
    # 初始化 ROS 节点
    rospy.init_node('test_roi', anonymous=True)
    occupancy_grid = test_map_10x10
    map_resolution = 0.5 # 您可以为这个 10x10 地图设置一个合适的分辨率
    robot_radius_m = 1.0 # 根据需要设置机器人半径
    map_resolution = 1.0 # 每个栅格的实际大小 (m)
    print("Occupancy Grid:\n", occupancy_grid)
    # 调用函数
    grid_matrix = np.where(occupancy_grid, 1, 0).astype(np.int32)
    # cspace = calculate_cspace(occupancy_grid, map_resolution, robot_radius_m)
    # print("C-Space:\n", cspace)