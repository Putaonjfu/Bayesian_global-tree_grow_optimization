import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import os
import open3d as o3d  # 引入 Open3D


def distance_point_to_segment(p, a, b):
    """
    利用向量的点乘判断向量夹角
    """
    ab = b - a
    len_sq = np.dot(ab, ab)
    if len_sq == 0.0:#ab重合
        return np.linalg.norm(p - a)#返回p与a之间的距离
    t = np.dot(p - a, ab) / len_sq
    if 0 <= t <= 1:# 向量在直线之间，先计算p在ab间的投影，再计算距离
        projection = a + t * ab
        return np.linalg.norm(p - projection)
    return min(np.linalg.norm(p - a), np.linalg.norm(p - b))# 否则返回到a与b的最短距离


# (注意: 之前添加的 distance_point_to_line 函数已被移除，
# 因为新的逻辑是“投影筛选”，而不是计算到无限长直线的距离)

# ******************************************************
# 2. 核心算法：贝叶斯拓扑恢复 (已按您的建议修改)
# ******************************************************

def bayesian_topology_reconstruction(skeleton_points, point_cloud, k, alpha, sigma):
    """
    基于贝叶斯框架，从树骨架和原始点云中恢复拓扑结构。
    """
    num_skeleton_pts = skeleton_points.shape[0]

    # --- 步骤 1: 构建候选图 (Candidate Graph) ---
    skeleton_kdtree = KDTree(skeleton_points)
    candidate_graph = nx.Graph()
    candidate_graph.add_nodes_from(range(num_skeleton_pts))

    for i in range(num_skeleton_pts):# 创建一个跟骨架点数量一致的候选图，并构建将K近邻边连接关系放入图中形成无向图。
        distances, indices = skeleton_kdtree.query(skeleton_points[i], k=k + 1)
        for j_idx in indices[1:]:
            if not candidate_graph.has_edge(i, j_idx):
                candidate_graph.add_edge(i, j_idx)

    print(f"候选图构建完成：{candidate_graph.number_of_nodes()} 个节点, {candidate_graph.number_of_edges()} 条边。")

    # --- 步骤 2: 将原始点云关联到候选边 ---
    edge_associations = {edge: [] for edge in candidate_graph.edges()}# 为每条边建立一个字典，建是候选边，值是一个列表用于后续储存对应的最近邻tls点索引
    cloud_kdtree = KDTree(skeleton_points)
    _, nearest_2_indices = cloud_kdtree.query(point_cloud, k=2) #为每个TLS点云找到最近的两个骨架点

    for i in range(point_cloud.shape[0]):
        u, v = sorted(nearest_2_indices[i])#获取这两个骨架点的索引
        if candidate_graph.has_edge(u, v):# 如果存在边
            edge_associations[(u, v)].append(point_cloud[i])# tls的最邻近骨架边就是uv了，加入候选中

    print("点云到边的关联计算完成。")

    # --- 步骤 3: 计算每条候选边的贝叶斯权重 (【已按您的要求重写】) ---
    for u, v in candidate_graph.edges():#遍历每条边
        p1 = skeleton_points[u]
        p2 = skeleton_points[v]
        edge_length = np.linalg.norm(p1 - p2)# 两个边之间的距离
        log_prior = -alpha * edge_length # 先验概率

        associated_points = np.array(edge_associations.get((u, v), []))

        # --- 【修改点】: 实施“投影筛选”逻辑 ---

        valid_distances_sq = []
        ab = p2 - p1# 计算边向量
        len_sq = np.dot(ab, ab)# 若计算边长

        if len_sq == 0.0:
            # 边长为0，骨架点重合了，所有点都是到该点的距离
            for pt in associated_points:
                valid_distances_sq.append(np.sum((pt - p1) ** 2))
        else:# 存在骨架边
            for pt in associated_points:# 遍历与这条边匹配的那些最近的tls点
                ap = pt - p1
                # 计算投影系数 t
                t = np.dot(ap, ab) / len_sq

                # 【核心筛选】: 只保留投影在线段内部的点
                if 0 <= t <= 1:
                    # 计算该点到直线的垂直距离的平方
                    projection = p1 + t * ab
                    distance_sq = np.sum((pt - projection) ** 2)
                    valid_distances_sq.append(distance_sq)# 将距离的平方放进数组

        # 似然函数现在基于 "有效点" 的数量和距离
        num_valid_points = len(valid_distances_sq)

        if num_valid_points == 0:
            # 如果没有点支持这条边 (或者所有点都在"帽子"区域)，则似然为-inf
            log_likelihood = -np.inf
        else:
            ssd = sum(valid_distances_sq)

            # 使用 num_valid_points 而不是 num_points
            log_likelihood = num_valid_points * np.log(1 / (sigma * np.sqrt(2 * np.pi))) - ssd / (2 * sigma ** 2)
        # --- 【修改结束】 ---

        weight = -(log_likelihood + log_prior)
        candidate_graph[u][v]['weight'] = weight

    print("所有候选边的权重计算完成。")

    # --- 步骤 4: 计算最小生成树 (MST) ---
    mst = nx.minimum_spanning_tree(candidate_graph, weight='weight')
    print(f"最小生成树计算完成，恢复的拓扑包含 {mst.number_of_edges()} 条边。")
    return mst


# ******************************************************
# 3. 数据生成与可视化 (代码无变化)
# ******************************************************

def generate_sample_data(num_branches=4, points_per_branch=50, noise=0.1):
    """生成一个用于演示的Y形树状点云和骨架"""
    trunk_skel = np.array([np.array([0, 0, z]) for z in np.linspace(0, 5, 20)])
    skeleton = [trunk_skel]
    angles = np.linspace(0, 2 * np.pi, num_branches + 1)[:-1]
    for i in range(num_branches):
        angle = angles[i]
        branch_skel = np.array([
            np.array([r * np.cos(angle), r * np.sin(angle), 5 + r * 0.5])
            for r in np.linspace(0.1, 3, 15)
        ])
        skeleton.append(branch_skel)
    skeleton = np.vstack(skeleton)
    point_cloud = []
    for _ in range(points_per_branch * skeleton.shape[0]):
        pt = skeleton[np.random.randint(0, skeleton.shape[0])]
        point_cloud.append(pt + np.random.randn(3) * noise)
    return skeleton, np.array(point_cloud)


def plot_results_o3d(skeleton, point_cloud, mst_graph, root_node_idx):
    """
    使用 Open3D 可视化输入和输出。
    """
    # 1. 创建原始点云对象 (灰色)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色

    # 2. 创建骨架点云对象 (蓝色)
    skel_pcd = o3d.geometry.PointCloud()
    skel_pcd.points = o3d.utility.Vector3dVector(skeleton)
    skel_pcd.paint_uniform_color([0.1, 0.1, 0.8])  # 蓝色

    # 3. 创建表示MST的LineSet对象 (红色)
    line_indices = list(mst_graph.edges())
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(skeleton),
        lines=o3d.utility.Vector2iVector(line_indices),
    )
    line_set.paint_uniform_color([1.0, 0.0, 0.0])  # 红色

    # 4. 创建根节点的几何体 (一个大的红色球体)
    root_pos = skeleton[root_node_idx]
    # 根据点云的整体尺寸来决定球体的半径
    bbox_diag = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    root_sphere_radius = bbox_diag * 0.01  # 半径为包围盒对角线的1%

    root_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=root_sphere_radius)
    root_sphere.translate(root_pos)
    root_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色

    # 5. 可视化所有几何体
    print("\n正在打开Open3D可视化窗口... 在窗口中按 'q' 键可关闭。")
    o3d.visualization.draw_geometries(
        [pcd, skel_pcd, line_set, root_sphere],
        window_name="贝叶斯拓扑恢复结果 (Open3D)",
        width=1600,
        height=900
    )


# ******************************************************
# 4. 数据加载函数 (代码无变化)
# ******************************************************
def load_data_from_files(point_cloud_path, skeleton_path):
    """
    从TXT文件加载点云和骨架。
    - 点云文件直接读取。
    - 骨架文件跳过第一行读取。
    """
    print("--- 开始从文件加载数据 ---")
    point_cloud, skeleton_points = None, None

    if not os.path.exists(point_cloud_path):
        print(f"❌ 错误: 点云文件未找到 -> {point_cloud_path}")
        return None, None
    if not os.path.exists(skeleton_path):
        print(f"❌ 错误: 骨架文件未找到 -> {skeleton_path}")
        return None, None

    try:
        point_cloud = np.loadtxt(point_cloud_path)
        print(f"✅ 成功加载点云 ({point_cloud.shape[0]} 个点): {point_cloud_path}")
    except Exception as e:
        print(f"❌ 加载点云文件时发生错误: {e}")
        return None, None

    try:
        # 仅读取前三列 (x, y, z)
        skeleton_points = np.loadtxt(skeleton_path, skiprows=1, usecols=(0, 1, 2))
        print(f"✅ 成功加载骨架 ({skeleton_points.shape[0]} 个点，已跳过第一行): {skeleton_path}")
    except Exception as e:
        print(f"❌ 加载骨架文件时发生错误: {e}")
        return None, None

    return skeleton_points, point_cloud


# ******************************************************
# 5. 主程序入口 (代码无变化)
# ******************************************************

if __name__ == '__main__':
    # --- 1. 准备数据 ---
    # 【请在这里修改为您自己的文件路径】
    SKELETON_FILE = r'E:\PHD\Waterdropletshrinkage\Ours\result_knn_l1_ske\结果部分(最佳参数)\我们的结果\未优化的结果\Tree_1.txt'
    POINT_CLOUD_FILE = r'E:\PHD\Waterdropletshrinkage\Ours\result_knn_l1_ske\Data\Tree_1.txt'

    # --- 自动生成示例文件用于测试 ---
    if not os.path.exists(SKELETON_FILE) or not os.path.exists(POINT_CLOUD_FILE):
        print("警告: 未找到指定的数据文件，正在生成示例文件用于演示...")
        # 定义示例文件的路径
        current_dir = os.getcwd()
        SKELETON_FILE = os.path.join(current_dir, 'skeleton_demo.txt')
        POINT_CLOUD_FILE = os.path.join(current_dir, 'point_cloud_demo.txt')

        skel_data, pc_data = generate_sample_data(num_branches=3, points_per_branch=80, noise=0.2)
        np.savetxt(POINT_CLOUD_FILE, pc_data, fmt='%.6f')
        np.savetxt(SKELETON_FILE, skel_data, fmt='%.6f', header='x y z', comments='')
        print(f"已创建示例文件 '{SKELETON_FILE}' 和 '{POINT_CLOUD_FILE}'")
    # --- 示例文件生成结束 ---

    # 从文件加载数据
    skeleton_points, point_cloud = load_data_from_files(
        point_cloud_path=POINT_CLOUD_FILE,
        skeleton_path=SKELETON_FILE
    )

    if skeleton_points is None or point_cloud is None:
        print("\n数据加载失败，程序已终止。")
    else:
        # --- 2. 设置超参数 ---
        K_NEIGHBORS = 5  # 候选图中每个骨架点连接的邻居数
        ALPHA = 2.0  # 长度惩罚系数 (prior)
        SIGMA = 0.25  # 点云分布的标准差 (likelihood)

        # --- 3. 运行拓扑恢复算法 ---
        print("\n开始拓扑恢复...")
        reconstructed_tree = bayesian_topology_reconstruction(
            skeleton_points,
            point_cloud,
            k=K_NEIGHBORS,
            alpha=ALPHA,
            sigma=SIGMA
        )

        # --- 4. 找到根节点并使用 Open3D 可视化 ---
        root_node_index = np.argmin(skeleton_points[:, 2])

        print("\n准备可视化结果...")
        plot_results_o3d(skeleton_points, point_cloud, reconstructed_tree, root_node_index)