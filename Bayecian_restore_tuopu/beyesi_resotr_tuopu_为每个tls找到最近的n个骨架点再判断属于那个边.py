import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import os
import open3d as o3d  # 引入 Open3D
import matplotlib.cm as cm  # 引入 matplotlib 色彩映射
from itertools import combinations  # 引入 itertools 用于高效遍历邻居对


# (distance_point_to_segment 函数无变化)
def distance_point_to_segment(p, a, b):
    ab = b - a
    len_sq = np.dot(ab, ab)
    if len_sq == 0.0:
        return np.linalg.norm(p - a)
    t = np.dot(p - a, ab) / len_sq
    if 0 <= t <= 1:
        projection = a + t * ab
        return np.linalg.norm(p - projection)
    return min(np.linalg.norm(p - a), np.linalg.norm(p - b))


# ******************************************************
# 2. 核心算法：贝叶斯拓扑恢复 (【已按 "分段先验" 重写】)
# ******************************************************

def bayesian_topology_reconstruction(skeleton_points, point_cloud, k_graph, k_assoc, alpha, sigma, a, b, kappa,
                                     mu_theta):
    """
    基于贝叶斯框架，从树骨架和原始点云中恢复拓扑结构。

    【已重构】:
    - 实现了基于高度的 "树干/树冠" 分段先验 [cite: 22, 134, 135]。
    - 实现了 "方法E" (K邻居-多重分配) 的似然计算。
    """
    num_skeleton_pts = skeleton_points.shape[0]

    # --- 步骤 1: 构建候选图 (Candidate Graph) ---
    skeleton_kdtree = KDTree(skeleton_points)
    candidate_graph = nx.Graph()
    candidate_graph.add_nodes_from(range(num_skeleton_pts))

    for i in range(num_skeleton_pts):
        distances, indices = skeleton_kdtree.query(skeleton_points[i], k=k_graph + 1)
        for j_idx in indices[1:]:
            if not candidate_graph.has_edge(i, j_idx):
                candidate_graph.add_edge(i, j_idx)

    print(
        f"候选图构建完成 (k_graph={k_graph})：{candidate_graph.number_of_nodes()} 个节点, {candidate_graph.number_of_edges()} 条边。")

    # --- 【新】步骤 1.5: 定义分段阈值 ---
    # 这是实现PDF中 "分段先验" [cite: 22] 的关键
    root_node_z = skeleton_points.min(axis=0)[2]
    max_z = skeleton_points.max(axis=0)[2]
    # 假设树干占总高度的底部 30% (这是一个可调参数)
    trunk_height_threshold = root_node_z + (max_z - root_node_z) * 0.3
    print(f"检测到树干/树冠分界高度 (Z): {trunk_height_threshold:.2f}")

    # 为树干定义更强的先验
    TRUNK_MU_THETA = np.pi  # 180度，强制直线
    TRUNK_KAPPA = 10.0  # 高集中度，强制执行 [cite: 136]

    # --- 步骤 2: 计算贝叶斯权重 (合并了原步骤2和3) ---
    edge_associations_sq_dists = {edge: [] for edge in candidate_graph.edges()}

    print(f"开始计算似然函数 (k_assoc={k_assoc})... (这将需要一些时间)")
    num_cloud_points = point_cloud.shape[0]

    # --- 【"方法E" 核心逻辑】 ---
    for i in range(num_cloud_points):
        pt = point_cloud[i]
        distances, indices = skeleton_kdtree.query(pt, k=k_assoc)

        for u, v in combinations(indices, 2):
            u_idx, v_idx = sorted((u, v))

            if not candidate_graph.has_edge(u_idx, v_idx):
                continue

            p1 = skeleton_points[u_idx]
            p2 = skeleton_points[v_idx]

            ab = p2 - p1
            ap = pt - p1
            len_sq = np.dot(ab, ab)

            if len_sq == 0.0:
                t = 0.0
            else:
                t = np.dot(ap, ab) / len_sq

            if 0 <= t <= 1:
                projection = p1 + t * ab
                distance_sq = np.sum((pt - projection) ** 2)
                edge_associations_sq_dists[(u_idx, v_idx)].append(distance_sq)

        if (i + 1) % 50000 == 0:
            print(f"  ...已处理 {i + 1} / {num_cloud_points} 个点")

    print("所有点的似然贡献计算完成。")
    print("开始计算最终的边权重 (包含分段先验)...")

    # --- 步骤 3: 汇总权重 (已应用分段先验) ---
    for u, v in candidate_graph.edges():
        p1 = skeleton_points[u]
        p2 = skeleton_points[v]
        edge_length = np.linalg.norm(p1 - p2)

        if edge_length == 0:
            edge_length = 1e-9

        # --- A. 计算Gamma长度先验 ---
        log_prior = -alpha * edge_length
        gamma_prior = (a - 1) * np.log(edge_length) - edge_length / b

        # --- B. 【新】计算分段角度先验 ---
        # (注意: 这里的 angle_diff 是基于原点的位置向量，
        # 这是您代码中的原始实现，而不是PDF中的 "边-边" 角度。我们在此保留它。)
        norm_p1 = np.linalg.norm(p1)
        norm_p2 = np.linalg.norm(p2)
        if norm_p1 == 0 or norm_p2 == 0:
            angle_diff = 0
        else:
            dot_product = np.clip(np.dot(p1, p2) / (norm_p1 * norm_p2), -1.0, 1.0)
            angle_diff = np.arccos(dot_product)

        # 【核心修改】: 根据边的平均高度应用不同的先验
        edge_z_mid = (p1[2] + p2[2]) / 2

        if edge_z_mid < trunk_height_threshold:
            # 这是树干: 应用 180 度先验
            current_mu_theta = TRUNK_MU_THETA
            current_kappa = TRUNK_KAPPA
        else:
            # 这是树冠: 应用用户指定的 45 度先验
            current_mu_theta = mu_theta
            current_kappa = kappa

        von_mises_prior = current_kappa * np.cos(angle_diff - current_mu_theta)

        # --- C. 计算似然 (Likelihood) ---
        valid_distances_sq = edge_associations_sq_dists.get((u, v), [])
        num_valid_points = len(valid_distances_sq)

        if num_valid_points == 0:
            log_likelihood = -np.inf
        else:
            ssd = sum(valid_distances_sq)
            # 【重要】: 这里的 sigma 是全局的。
            # 真正的PDF实现会在这里也使用分段的 sigma_trunk 和 sigma_crown [cite: 347, 349]
            # 但我们目前在代码中只使用一个全局 sigma
            log_likelihood = num_valid_points * np.log(1 / (sigma * np.sqrt(2 * np.pi))) - ssd / (2 * sigma ** 2)

        # --- D. 计算总权重 ---
        weight = -(log_likelihood + log_prior + gamma_prior + von_mises_prior)
        candidate_graph[u][v]['weight'] = weight

    print("所有候选边的权重计算完成。")

    # --- 步骤 4: 计算最小生成树 (MST) ---
    mst = nx.minimum_spanning_tree(candidate_graph, weight='weight')
    print(f"最小生成树计算完成，恢复的拓扑包含 {mst.number_of_edges()} 条边。")

    return mst, candidate_graph


# ******************************************************
# 3. 数据生成与可视化 (无变化)
# ******************************************************

def generate_sample_data(num_branches=4, points_per_branch=50, noise=0.1):
    # ... (此函数代码无变化) ...
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


# --- 可视化函数 1 ---
def plot_candidate_graph_o3d(skeleton, point_cloud, candidate_graph, root_node_idx, root_sphere_radius):
    # ... (此函数代码无变化) ...
    skel_pcd = o3d.geometry.PointCloud()
    skel_pcd.points = o3d.utility.Vector3dVector(skeleton)
    skel_pcd.paint_uniform_color([0.0, 0.0, 0.0])  # 黑色
    line_indices = list(candidate_graph.edges())
    line_set_all = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(skeleton),
        lines=o3d.utility.Vector2iVector(line_indices),
    )
    line_set_all.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
    root_pos = skeleton[root_node_idx]
    root_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=root_sphere_radius)
    root_sphere.translate(root_pos)
    root_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    print("\n正在打开Open3D可视化窗口 (1/3: 初始候选图)... 在窗口中按 'q' 键可关闭。")
    print("  (绿色 = 所有候选边)")
    print("  (提示: 在窗口中按 '+' 键可以使线条变粗)")
    o3d.visualization.draw_geometries(
        [skel_pcd, line_set_all, root_sphere],
        window_name="1. 初始候选图 (全绿)",
        width=1600,
        height=900
    )


# --- 可视化函数 2 ---
def plot_probability_graph_o3d(skeleton, point_cloud, candidate_graph, mst_graph, root_node_idx, root_sphere_radius):
    # ... (此函数代码无变化) ...
    skel_pcd = o3d.geometry.PointCloud()
    skel_pcd.points = o3d.utility.Vector3dVector(skeleton)
    skel_pcd.paint_uniform_color([0.0, 0.0, 0.0])  # 黑色
    mst_edges = set(tuple(sorted(e)) for e in mst_graph.edges())
    mst_line_indices = []
    other_line_indices = []
    for u, v in candidate_graph.edges():
        edge = tuple(sorted((u, v)))
        if edge in mst_edges:
            mst_line_indices.append([u, v])
        else:
            other_line_indices.append([u, v])
    line_set_other = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(skeleton),
        lines=o3d.utility.Vector2iVector(other_line_indices),
    )
    line_set_other.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
    line_set_mst = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(skeleton),
        lines=o3d.utility.Vector2iVector(mst_line_indices),
    )
    line_set_mst.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    root_pos = skeleton[root_node_idx]
    root_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=root_sphere_radius)
    root_sphere.translate(root_pos)
    root_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    print("\n正在打开Open3D可视化窗口 (2/3: MST 高亮图)... 在窗口中按 'q' 键可关闭。")
    print("  (红色 = 最终MST选择, 蓝色 = 其他候选边)")
    print("  (提示: 在窗口中按 '+' 键可以使线条变粗)")
    o3d.visualization.draw_geometries(
        [skel_pcd, line_set_other, line_set_mst, root_sphere],
        window_name="2. MST 高亮可视化 (红=选中, 蓝=其他)",
        width=1600,
        height=900
    )


# --- 可视化函数 3 ---
def plot_results_o3d(skeleton, point_cloud, mst_graph, root_node_idx, root_sphere_radius, output_path=None):
    # ... (此函数代码无变化) ...
    skel_pcd = o3d.geometry.PointCloud()
    skel_pcd.points = o3d.utility.Vector3dVector(skeleton)
    skel_pcd.paint_uniform_color([0.0, 0.0, 0.0])  # 黑色
    line_indices = list(mst_graph.edges())
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(skeleton),
        lines=o3d.utility.Vector2iVector(line_indices),
    )
    line_set.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    root_pos = skeleton[root_node_idx]
    root_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=root_sphere_radius)
    root_sphere.translate(root_pos)
    root_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    if output_path:
        try:
            o3d.io.write_line_set(output_path, line_set)
            print(f"✅ 成功将拓扑结构 (MST) 保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存拓扑结构时发生错误: {e}")
    print("\n正在打开Open3D可视化窗口 (3/3: 最终MST结果)... 在窗口中按 'q' 键可关闭。")
    print("  (提示: 在窗口中按 '+' 键可以使线条变粗)")
    o3d.visualization.draw_geometries(
        [skel_pcd, line_set, root_sphere],
        window_name="3. 最终拓扑恢复结果 (MST)",
        width=1600,
        height=900
    )


# ******************************************************
# 4. 数据加载函数 (无变化)
# ******************************************************

def load_data_from_files(point_cloud_path, skeleton_path):
    # ... (此函数代码无变化) ...
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

    # --- 自动生成示例文件 (无变化) ---
    if not os.path.exists(SKELETON_FILE) or not os.path.exists(POINT_CLOUD_FILE):
        print("警告: 未找到指定的数据文件，正在生成示例文件用于演示...")
        # ... (代码无变化) ...
        current_dir = os.getcwd()
        SKELETON_FILE = os.path.join(current_dir, 'skeleton_demo.txt')
        POINT_CLOUD_FILE = os.path.join(current_dir, 'point_cloud_demo.txt')
        MST_OUTPUT_FILE = os.path.join(current_dir, 'reconstructed_topology_demo.ply')
        skel_data, pc_data = generate_sample_data(num_branches=3, points_per_branch=80, noise=0.2)
        np.savetxt(POINT_CLOUD_FILE, pc_data, fmt='%.6f')
        np.savetxt(SKELETON_FILE, skel_data, fmt='%.6f', header='x y z', comments='')
        print(f"已创建示例文件 '{SKELETON_FILE}' 和 '{POINT_CLOUD_FILE}'")
    # --- 示例文件生成结束 ---

    skeleton_points, point_cloud = load_data_from_files(
        point_cloud_path=POINT_CLOUD_FILE,
        skeleton_path=SKELETON_FILE
    )

    if skeleton_points is None or point_cloud is None:
        print("\n数据加载失败，程序已终止。")
    else:
        # --- 2. 设置超参数 ---

        # 【重要】: 解决过度拟合的建议参数
        # (我们现在实现了分段角度先验，但 "削弱似然" 和 "加强长度先验" 仍然至关重要)

        K_GRAPH_NEIGHBORS = 4  # (原为 5) 稍稍减少候选边
        K_ASSOCIATION_NEIGHBORS = 8  # (原为 10) 稍稍减少局部搜索范围

        ALPHA = 2.0  # (原为 2.0) 保持不变或略增

        # 【关键】: 显著增大 SIGMA, 削弱似然项, 避免拟合噪声 [cite: 351]
        SIGMA = 1.0  # (原为 0.25)

        # 【关键】: 增大 'a', 惩罚极短的边, 鼓励更平滑的结构 [cite: 120, 124]
        a = 2.0  # (原为 1.2)
        b = 0.1  # (原为 0.044)

        # 这些是 "树冠" 的先验
        kappa = 2.16
        mu_theta = np.pi / 4

        # --- 3. 运行拓扑恢复算法 ---
        print("\n开始拓扑恢复...")
        reconstructed_tree, full_candidate_graph = bayesian_topology_reconstruction(
            skeleton_points,
            point_cloud,
            k_graph=K_GRAPH_NEIGHBORS,
            k_assoc=K_ASSOCIATION_NEIGHBORS,
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
        root_sphere_radius = skel_bbox_diag * 0.01

        print("\n准备可视化流程 (共 3 个窗口)...")

        # --- 5. 按顺序显示三个可视化窗口 ---

        # 可视化 1 - 初始候选图 (绿色)
        plot_candidate_graph_o3d(
            skeleton_points,
            point_cloud,
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