import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

# 步骤1：从TXT文件读取数据
def load_tlspoints_from_txt(file_path):
    """从TXT文件读取三维点数据，假设格式为 'x y z'"""
    points = np.loadtxt(file_path, delimiter=' ')  # 假设空格分隔
    if points.shape[1] != 3:
        raise ValueError("TXT文件格式应为每行 'x y z'，当前列数不符")
    return points
def load_skepoints_from_txt(file_path):
    """从TXT文件读取三维点数据，假设格式为 'x y z'"""
    points = np.loadtxt(file_path, usecols=(0,1,2),skiprows=1)  # 假设空格分隔
    if points.shape[1] != 3:
        raise ValueError("TXT文件格式应为每行 'x y z'，当前列数不符")
    return points
# 替换为你的文件路径
tls_file_path = r"E:\Waterdropletshrinkage\Ours\Skeletonpointgeneration\Data\L1_data\Tree_1.txt"  # TLS点云TXT文件路径
skeleton_file_path = r"E:\Waterdropletshrinkage\Ours\Skeletonpointgeneration\code_refine\reconstruction\Test_data\Tree_1.txt"  # 骨架点TXT文件路径

tls_points = load_tlspoints_from_txt(tls_file_path)
skeleton_points = load_skepoints_from_txt(skeleton_file_path)

print(f"加载了 {len(tls_points)} 个TLS点和 {len(skeleton_points)} 个骨架点")

# 步骤2：为每个TLS点分配最近的骨架点
skeleton_tree = KDTree(skeleton_points)
distances, nearest_skeleton_indices = skeleton_tree.query(tls_points)

# 输出分配结果（可选）
for tls_idx, (tls_point, skeleton_idx, dist) in enumerate(zip(tls_points, nearest_skeleton_indices, distances)):
    print(f"TLS点 {tls_idx}: {tls_point} -> 最近骨架点 {skeleton_idx}: {skeleton_points[skeleton_idx]}, 距离: {dist:.4f}")

# （可选）构建邻域集合
neighborhoods = [[] for _ in range(len(skeleton_points))]
for tls_idx, skeleton_idx in enumerate(nearest_skeleton_indices):
    neighborhoods[skeleton_idx].append(tls_points[tls_idx])
neighborhoods = [np.array(n) if n else np.empty((0, 3)) for n in neighborhoods]

# 步骤3：使用Open3D可视化
# 创建TLS点云对象
tls_pcd = o3d.geometry.PointCloud()
tls_pcd.points = o3d.utility.Vector3dVector(tls_points)
tls_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色表示TLS点

# 创建骨架点云对象
skeleton_pcd = o3d.geometry.PointCloud()
skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_points)
skeleton_pcd.paint_uniform_color([1, 0, 0])  # 红色表示骨架点

# 创建线段表示TLS点到最近骨架点的连接
lines = []
for tls_idx, skeleton_idx in enumerate(nearest_skeleton_indices):
    lines.append([tls_idx, skeleton_idx + len(tls_points)])  # 线段连接TLS点和骨架点
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(np.vstack((tls_points, skeleton_points)))  # 合并所有点
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(lines))])  # 蓝色线段

# 可视化
o3d.visualization.draw_geometries([tls_pcd, skeleton_pcd, line_set],
                                  window_name="TLS to Skeleton Assignment",
                                  width=800, height=600)

# （可选）输出邻域信息
print("\n每个骨架点的邻域TLS点：")
for i, neighborhood in enumerate(neighborhoods):
    print(f"骨架点 {i}: {skeleton_points[i]} 的邻域 ({len(neighborhood)} 个点):")
    print(neighborhood)