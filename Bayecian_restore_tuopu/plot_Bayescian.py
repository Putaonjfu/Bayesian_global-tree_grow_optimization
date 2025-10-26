import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import os
import open3d as o3d  # 引入 Open3D
import matplotlib.cm as cm  # 引入 matplotlib 色彩映射


# 计算一个点 p 到线段 (a, b) 的最短距离
def distance_point_to_segment(p, a, b):
    """
    利用向量的点乘判断向量夹角
    """
    ab = b - a
    len_sq = np.dot(ab, ab)
    if len_sq == 0.0:  # ab重合
        return np.linalg.norm(p - a)  # 返回p与a之间的距离
    t = np.dot(p - a, ab) / len_sq
    if 0 <= t <= 1:  # 向量在直线之间，先计算p在ab间的投影，再计算距离
        projection = a + t * ab
        return np.linalg.norm(p - projection)
    return min(np.linalg.norm(p - a), np.linalg.norm(p - b))  # 否则返回到a与b的最短距离


# ******************************************************
# 2. 核心算法：贝叶斯拓扑恢复
# ******************************************************

def bayesian_topology_reconstruction(skeleton_points, point_cloud, k, alpha, sigma, a, b, kappa, mu_theta):
    """
    基于贝叶斯框架，从树骨架和原始点云中恢复拓扑结构。
    """
    num_skeleton_pts = skeleton_points.shape[0]

    # --- 步骤 1: 构建候选图 (Candidate Graph) ---
    skeleton_kdtree = KDTree(skeleton_points)
    candidate_graph = nx.Graph()
    candidate_graph.add_nodes_from(range(num_skeleton_pts))

    for i in range(num_skeleton_pts):
        distances, indices = skeleton_kdtree.query(skeleton_points[i], k=k + 1)
        for j_idx in indices[1:]:
            if not candidate_graph.has_edge(i, j_idx):
                candidate_graph.add_edge(i, j_idx)

    print(f"候选图构建完成：{candidate_graph.number_of_nodes()} 个节点, {candidate_graph.number_of_edges()} 条边。")

    # --- 步骤 2: 将原始点云关联到候选边 ---
    edge_associations = {edge: [] for edge in candidate_graph.edges()}
    cloud_kdtree = KDTree(skeleton_points)
    _, nearest_2_indices = cloud_kdtree.query(point_cloud, k=2)

    for i in range(point_cloud.shape[0]):
        u, v = sorted(nearest_2_indices[i])
        if candidate_graph.has_edge(u, v):
            edge_associations[(u, v)].append(point_cloud[i])

    print("点云到边的关联计算完成。")

    # --- 步骤 3: 计算每条候选边的贝叶斯权重 ---
    for u, v in candidate_graph.edges():
        p1 = skeleton_points[u]
        p2 = skeleton_points[v]
        edge_length = np.linalg.norm(p1 - p2)

        if edge_length == 0:
            edge_length = 1e-9

        log_prior = -alpha * edge_length
        gamma_prior = (a - 1) * np.log(edge_length) - edge_length / b

        norm_p1 = np.linalg.norm(p1)
        norm_p2 = np.linalg.norm(p2)
        if norm_p1 == 0 or norm_p2 == 0:
            angle_diff = 0
        else:
            dot_product = np.clip(np.dot(p1, p2) / (norm_p1 * norm_p2), -1.0, 1.0)
            angle_diff = np.arccos(dot_product)

        von_mises_prior = kappa * np.cos(angle_diff - mu_theta)

        associated_points = np.array(edge_associations.get((u, v), []))

        valid_distances_sq = []
        ab = p2 - p1
        len_sq = np.dot(ab, ab)

        if len_sq == 0.0:
            for pt in associated_points:
                valid_distances_sq.append(np.sum((pt - p1) ** 2))
        else:
            for pt in associated_points:
                ap = pt - p1
                t = np.dot(ap, ab) / len_sq
                if 0 <= t <= 1:
                    projection = p1 + t * ab
                    distance_sq = np.sum((pt - projection) ** 2)
                    valid_distances_sq.append(distance_sq)

        num_valid_points = len(valid_distances_sq)

        if num_valid_points == 0:
            log_likelihood = -np.inf
        else:
            ssd = sum(valid_distances_sq)
            log_likelihood = num_valid_points * np.log(1 / (sigma * np.sqrt(2 * np.pi))) - ssd / (2 * sigma ** 2)

        weight = -(log_likelihood + log_prior + gamma_prior + von_mises_prior)
        candidate_graph[u][v]['weight'] = weight

    print("所有候选边的权重计算完成。")

    # --- 步骤 4: 计算最小生成树 (MST) ---
    mst = nx.minimum_spanning_tree(candidate_graph, weight='weight')
    print(f"最小生成树计算完成，恢复的拓扑包含 {mst.number_of_edges()} 条边。")

    return mst, candidate_graph


# ******************************************************
# 3. 数据生成与可视化
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


# --- 【新】可视化函数 1 ---
def plot_candidate_graph_o3d(skeleton, point_cloud, candidate_graph, root_node_idx, root_sphere_radius):
    """
    【新】可视化1: 初始候选图 (黑点, 绿线)
    """
    # 1. 【已注释掉】创建点云
    # pcd = o3d.geometry.PointCloud()

    # 2. 创建骨架点 (黑色)
    skel_pcd = o3d.geometry.PointCloud()
    skel_pcd.points = o3d.utility.Vector3dVector(skeleton)
    skel_pcd.paint_uniform_color([0.0, 0.0, 0.0])  # 黑色

    # 3. 创建表示 *所有* 候选边的 LineSet (绿色)
    line_indices = list(candidate_graph.edges())
    line_set_all = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(skeleton),
        lines=o3d.utility.Vector2iVector(line_indices),
    )
    line_set_all.paint_uniform_color([0.0, 1.0, 0.0])  # 【新】绿色

    # 4. 创建根节点的几何体 (红色)
    root_pos = skeleton[root_node_idx]
    root_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=root_sphere_radius)
    root_sphere.translate(root_pos)
    root_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色

    # 5. 可视化
    print("\n正在打开Open3D可视化窗口 (1/3: 初始候选图)... 在窗口中按 'q' 键可关闭。")
    print("  (绿色 = 所有候选边)")
    print("  (提示: 在窗口中按 '+' 键可以使线条变粗)")

    o3d.visualization.draw_geometries(
        [skel_pcd, line_set_all, root_sphere],
        window_name="1. 初始候选图 (全绿)",
        width=1600,
        height=900
    )


# --- 可视化函数 2 (原 plot_probability_graph_o3d) ---
def plot_probability_graph_o3d(skeleton, point_cloud, candidate_graph, mst_graph, root_node_idx, root_sphere_radius):
    """
    可视化2: 高亮图 (黑点, 红线, 蓝线, 红色根节点)
    """
    # 1. 【已注释掉】创建点云
    # pcd = o3d.geometry.PointCloud()

    # 创建骨架点 (黑色)
    skel_pcd = o3d.geometry.PointCloud()
    skel_pcd.points = o3d.utility.Vector3dVector(skeleton)
    skel_pcd.paint_uniform_color([0.0, 0.0, 0.0])  # 黑色

    # 2. 将边分为 "MST" 和 "其他"
    mst_edges = set(tuple(sorted(e)) for e in mst_graph.edges())
    mst_line_indices = []
    other_line_indices = []

    for u, v in candidate_graph.edges():
        edge = tuple(sorted((u, v)))
        if edge in mst_edges:
            mst_line_indices.append([u, v])
        else:
            other_line_indices.append([u, v])

    # 3. 创建两个独立的 LineSet

    # "其他" 候选边 (蓝色)
    line_set_other = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(skeleton),
        lines=o3d.utility.Vector2iVector(other_line_indices),
    )
    line_set_other.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色

    # "MST" 边 (红色)
    line_set_mst = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(skeleton),
        lines=o3d.utility.Vector2iVector(mst_line_indices),
    )
    line_set_mst.paint_uniform_color([1.0, 0.0, 0.0])  # 红色

    # 4. 创建根节点的几何体 (红色)
    root_pos = skeleton[root_node_idx]
    root_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=root_sphere_radius)
    root_sphere.translate(root_pos)
    root_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色

    # 5. 可视化
    print("\n正在打开Open3D可视化窗口 (2/3: MST 高亮图)... 在窗口中按 'q' 键可关闭。")
    print("  (红色 = 最终MST选择, 蓝色 = 其他候选边)")
    print("  (提示: 在窗口中按 '+' 键可以使线条变粗)")

    o3d.visualization.draw_geometries(
        [skel_pcd, line_set_other, line_set_mst, root_sphere],
        window_name="2. MST 高亮可视化 (红=选中, 蓝=其他)",
        width=1600,
        height=900
    )


# --- 可视化函数 3 (原 plot_results_o3d) ---
def plot_results_o3d(skeleton, point_cloud, mst_graph, root_node_idx, root_sphere_radius, output_path=None):
    """
    可视化3: 最终MST结果 (黑点, 红线)
    """
    # 1. 【已注释掉】创建原始点云对象
    # pcd = o3d.geometry.PointCloud()

    # 2. 创建骨架点云对象 (黑色)
    skel_pcd = o3d.geometry.PointCloud()
    skel_pcd.points = o3d.utility.Vector3dVector(skeleton)
    skel_pcd.paint_uniform_color([0.0, 0.0, 0.0])  # 黑色

    # 3. 创建表示MST的LineSet对象 (红色)
    line_indices = list(mst_graph.edges())
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(skeleton),
        lines=o3d.utility.Vector2iVector(line_indices),
    )
    line_set.paint_uniform_color([1.0, 0.0, 0.0])  # 红色

    # 4. 创建根节点的几何体 (红色)
    root_pos = skeleton[root_node_idx]
    root_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=root_sphere_radius)
    root_sphere.translate(root_pos)
    root_sphere.paint_uniform_color([1.0, 0.0, 0.0])

    # 5. 保存拓扑结构
    if output_path:
        try:
            o3d.io.write_line_set(output_path, line_set)
            print(f"✅ 成功将拓扑结构 (MST) 保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存拓扑结构时发生错误: {e}")

    # 6. 可视化所有几何体
    print("\n正在打开Open3D可视化窗口 (3/3: 最终MST结果)... 在窗口中按 'q' 键可关闭。")
    print("  (提示: 在窗口中按 '+' 键可以使线条变粗)")
    o3d.visualization.draw_geometries(
        [skel_pcd, line_set, root_sphere],
        window_name="3. 最终拓扑恢复结果 (MST)",
        width=1600,
        height=900
    )


# ******************************************************
# 4. 数据加载函数
# ******************************************************

def load_data_from_files(point_cloud_path, skeleton_path):
    """
    从TXT文件加载点云和骨架。
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
        skeleton_points = np.loadtxt(skeleton_path, skiprows=1, usecols=(0, 1, 2))
        print(f"✅ 成功加载骨架 ({skeleton_points.shape[0]} 个点，已跳过第一行): {skeleton_path}")
    except Exception as e:
        print(f"❌ 加载骨架文件时发生错误: {e}")
        return None, None

    return skeleton_points, point_cloud


# ******************************************************
# 5. 主程序入口 (【已修改】)
# ******************************************************

if __name__ == '__main__':
    # --- 1. 准备数据 ---
    SKELETON_FILE = r'E:\PHD\Waterdropletshrinkage\Ours\result_knn_l1_ske\结果部分(最佳参数)\我们的结果\未优化的结果\Tree_1.txt'
    POINT_CLOUD_FILE = r'E:\PHD\Waterdropletshrinkage\Ours\result_knn_l1_ske\Data\Tree_1.txt'
    MST_OUTPUT_FILE = r'reconstructed_topology.ply'

    # --- 自动生成示例文件用于测试 ---
    if not os.path.exists(SKELETON_FILE) or not os.path.exists(POINT_CLOUD_FILE):
        print("警告: 未找到指定的数据文件，正在生成示例文件用于演示...")
        current_dir = os.getcwd()
        SKELETON_FILE = os.path.join(current_dir, 'skeleton_demo.txt')
        POINT_CLOUD_FILE = os.path.join(current_dir, 'point_cloud_demo.txt')
        MST_OUTPUT_FILE = os.path.join(current_dir, 'reconstructed_topology_demo.ply')

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
        K_NEIGHBORS = 5
        ALPHA = 2.0
        SIGMA = 0.25
        a = 1.2
        b = 0.044
        kappa = 2.16
        mu_theta = np.pi / 4

        # --- 3. 运行拓扑恢复算法 ---
        print("\n开始拓扑恢复...")
        reconstructed_tree, full_candidate_graph = bayesian_topology_reconstruction(
            skeleton_points,
            point_cloud,
            k=K_NEIGHBORS,
            alpha=ALPHA,
            sigma=SIGMA,
            a=a,
            b=b,
            kappa=kappa,
            mu_theta=mu_theta
        )

        # --- 4. 找到根节点并【统一计算】球体大小 ---
        root_node_index = np.argmin(skeleton_points[:, 2])

        skel_bbox_diag = np.linalg.norm(skeleton_points.max(axis=0) - skeleton_points.min(axis=0))
        if skel_bbox_diag == 0: skel_bbox_diag = 1.0
        root_sphere_radius = skel_bbox_diag * 0.01  # 统一设为骨架包围盒的1%

        print("\n准备可视化流程 (共 3 个窗口)...")

        # 【新】可视化 1 - 初始候选图 (绿色)
        plot_candidate_graph_o3d(
            skeleton_points,
            point_cloud,  # 传入但未使用
            full_candidate_graph,
            root_node_index,
            root_sphere_radius
        )

        # 可视化 2 - MST高亮图 (红/蓝)
        plot_probability_graph_o3d(
            skeleton_points,
            point_cloud,
            full_candidate_graph,
            reconstructed_tree,
            root_node_index,
            root_sphere_radius
        )

        # 可视化 3 - 最终的 MST 结果 (红色) (并保存)
        plot_results_o3d(
            skeleton_points,
            point_cloud,
            reconstructed_tree,
            root_node_index,
            root_sphere_radius,
            output_path=MST_OUTPUT_FILE
        )