import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import os
import open3d as o3d
from itertools import combinations
from networkx.exception import NetworkXUnbounded, NetworkXNoPath

# ******************************************************
# 2. 核心算法：贝叶斯拓扑恢复 (【已修改】)
# ******************************************************

def bayesian_topology_reconstruction(skeleton_points, point_cloud, k_graph, k_assoc, alpha, sigma):
    """
    基于贝叶斯框架，从树骨架和原始点云中恢复拓扑结构。

    【已修改】:
    - 步骤4 使用 Bellman-Ford 的 "最短路径树" 替换 Dijkstra。
    - 函数现在会返回 (tree, root_node_index)。
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

    # --- 步骤 2: 将原始点云关联到候选边 (逻辑不变) ---
    print(f"开始将点云关联到候选边 (k_assoc={k_assoc})...")
    edge_associations = {edge: [] for edge in candidate_graph.edges()}
    num_cloud_points = point_cloud.shape[0]
    for i in range(num_cloud_points):
        p = point_cloud[i]
        distances, indices = skeleton_kdtree.query(p, k=k_assoc)
        min_valid_dist_sq = np.inf
        best_edge = None
        for u, v in combinations(indices, 2):
            u_idx, v_idx = sorted((u, v))
            if not candidate_graph.has_edge(u_idx, v_idx):
                continue
            p1 = skeleton_points[u_idx]
            p2 = skeleton_points[v_idx]
            ab = p2 - p1
            ap = p - p1
            len_sq = np.dot(ab, ab)
            if len_sq == 0.0:
                distance_sq = np.dot(ap, ap)
            else:
                t = np.dot(ap, ab) / len_sq
                if 0 <= t <= 1:
                    projection = p1 + t * ab
                    distance_sq = np.sum((p - projection) ** 2)
                else:
                    distance_sq = np.inf
            if distance_sq < min_valid_dist_sq:
                min_valid_dist_sq = distance_sq
                best_edge = (u_idx, v_idx)
        if best_edge is not None:
            edge_associations[best_edge].append(p)
        if (i + 1) % 20000 == 0:
            print(f"  ...已处理 {i + 1} / {num_cloud_points} 个点")
    print("点云到边的关联计算完成。")

    # --- 步骤 3: 计算每条候选边的贝叶斯权重 (逻辑不变) ---
    print("开始计算所有候选边的贝叶斯权重...")
    has_negative_weights = False  # 【新】加一个标记
    for u, v in candidate_graph.edges():
        p1 = skeleton_points[u]
        p2 = skeleton_points[v]
        edge_length = np.linalg.norm(p1 - p2)
        log_prior = -alpha * edge_length
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

        weight = -(log_likelihood + log_prior)

        # 检查权重是否为负
        if weight < 0:
            has_negative_weights = True

        candidate_graph[u][v]['weight'] = weight

    print("所有候选边的权重计算完成。")
    if has_negative_weights:
        print("信息: 检测到负权重。使用 Bellman-Ford 算法是正确的。")
    else:
        print("信息: 未检测到负权重。")

    # --- 【新】步骤 3.5: 查找根节点 ---
    print("正在查找根节点 (Z 坐标最低的点)...")
    root_node_index = np.argmin(skeleton_points[:, 2])
    print(f"根节点索引为: {root_node_index}")

    # --- 步骤 4: 【已修改】计算最短路径树 (SPT) (Bellman-Ford) ---
    print(f"开始以 {root_node_index} 为源点，计算最短路径树 (Bellman-Ford)...")

    # 检查候选图是否连通
    if not nx.is_connected(candidate_graph):
        print("警告: 候选图不连通。算法只能找到与根节点连通的部分。")
        try:
            root_component_nodes = nx.node_connected_component(candidate_graph, root_node_index)
            sub_graph = candidate_graph.subgraph(root_component_nodes)
            print(f"将在包含 {len(root_component_nodes)} 个节点的子图上运行。")
        except Exception as e:
            print(f"错误：无法找到根节点 {root_node_index} 的连通分量: {e}")
            return nx.Graph(), root_node_index
    else:
        print("候选图已连通。")
        sub_graph = candidate_graph

    # 2. 运行 Bellman-Ford 算法
    try:
        # 【修改】: 使用 nx.single_source_bellman_ford
        print("  正在使用 nx.single_source_bellman_ford ...")

        # 它返回 (距离字典, 前驱节点字典)
        # 注意：这里我们只关心 'pred' (前驱节点)
        # 我们需要将 sub_graph 转换为 DiGraph (有向图)，因为 Bellman-Ford 通常在有向图上定义
        # 但这里的权重是对称的，所以我们可以简单地创建一个有向版本
        G_directed = sub_graph.to_directed()

        # pred 是 {node: predecessor}
        pred, dist = nx.single_source_bellman_ford(G_directed, source=root_node_index, weight='weight')

        # 【修改】: 从返回的 "前驱节点字典" 中手动构建树图
        spt_undirected = nx.Graph()
        spt_undirected.add_nodes_from(sub_graph.nodes())

        for node, predecessor in pred.items():
            if predecessor is not None:
                # 添加从节点到其前驱节点的边
                spt_undirected.add_edge(node, predecessor)

        print(f"最短路径树计算完成，恢复的拓扑包含 {spt_undirected.number_of_edges()} 条边。")

        # 返回 SPT 树和根节点索引
        return spt_undirected, root_node_index

    except NetworkXUnbounded:
        # 【新】捕获 Bellman-Ford 的特定错误
        print("\n" + "=" * 50)
        print("❌ 致命错误: Bellman-Ford 算法失败。")
        print("原因: 图中存在“负权重环路”。")
        print("这意味着算法无法找到一个“最短”路径，因为它可以无限次地绕着这个环来降低成本。")
        print("这可能说明您的贝叶斯权重模型与最短路径算法不兼容。")
        print("强烈建议您退回使用 MST (最小生成树) 算法。")
        print("=" * 50 + "\n")
        return nx.Graph(), root_node_index
    except NetworkXNoPath:
        # (这个错误理论上不应该发生，因为我们是在连通分量上运行的)
        print(f"错误: 根节点 {root_node_index} 与图中的其他部分没有路径。")
        return nx.Graph(), root_node_index
    except Exception as e:
        # 捕获其他未知错误
        print(f"计算最短路径时发生未知错误: {e}")
        return nx.Graph(), root_node_index


# ******************************************************
# 3. 数据生成与可视化 (代码无变化)
# ******************************************************

def generate_sample_data(num_branches=4, points_per_branch=50, noise=0.1):
    # (此函数代码无变化)
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
    # (此函数代码无变化, 'mst_graph' 变量名仍可通用)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    skel_pcd = o3d.geometry.PointCloud()
    skel_pcd.points = o3d.utility.Vector3dVector(skeleton)
    skel_pcd.paint_uniform_color([0.1, 0.1, 0.8])
    line_indices = list(mst_graph.edges())
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(skeleton),
        lines=o3d.utility.Vector2iVector(line_indices),
    )
    line_set.paint_uniform_color([1.0, 0.0, 0.0])
    root_pos = skeleton[root_node_idx]
    bbox_diag = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    root_sphere_radius = bbox_diag * 0.01
    root_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=root_sphere_radius)
    root_sphere.translate(root_pos)
    root_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    print("\n正在打开Open3D可视化窗口... 在窗口中按 'q' 键可关闭。")
    o3d.visualization.draw_geometries(
        [pcd, skel_pcd, line_set, root_sphere],
        window_name="Dijkstra 最短路径树恢复结果 (Open3D)",  # 窗口标题已修改
        width=1600,
        height=900
    )


# ******************************************************
# 4. 数据加载函数 (代码无变化)
# ******************************************************
def load_data_from_files(point_cloud_path, skeleton_path):
    # (此函数代码无变化)
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

    if not os.path.exists(SKELETON_FILE) or not os.path.exists(POINT_CLOUD_FILE):
        print("警告: 未找到指定的数据文件，正在生成示例文件用于演示...")
        current_dir = os.getcwd()
        SKELETON_FILE = os.path.join(current_dir, 'skeleton_demo.txt')
        POINT_CLOUD_FILE = os.path.join(current_dir, 'point_cloud_demo.txt')
        skel_data, pc_data = generate_sample_data(num_branches=3, points_per_branch=80, noise=0.2)
        np.savetxt(POINT_CLOUD_FILE, pc_data, fmt='%.6f')
        np.savetxt(SKELETON_FILE, skel_data, fmt='%.6f', header='x y z', comments='')
        print(f"已创建示例文件 '{SKELETON_FILE}' 和 '{POINT_CLOUD_FILE}'")

    skeleton_points, point_cloud = load_data_from_files(
        point_cloud_path=POINT_CLOUD_FILE,
        skeleton_path=SKELETON_FILE
    )

    if skeleton_points is None or point_cloud is None:
        print("\n数据加载失败，程序已终止。")
    else:
        # --- 2. 设置超参数 ---
        K_GRAPH_NEIGHBORS = 5
        K_ASSOCIATION_NEIGHBORS = 10
        ALPHA = 2.0
        SIGMA = 0.25

        print("\n--- 超参数设置 ---")
        print(f"K (候选图):     {K_GRAPH_NEIGHBORS}")
        print(f"K (点云关联):  {K_ASSOCIATION_NEIGHBORS}")
        print(f"Alpha (长度惩罚): {ALPHA}")
        print(f"Sigma (噪声标准差): {SIGMA}")
        print("--------------------")

        # --- 3. 运行拓扑恢复算法 ---
        # 【已修改】: 现在函数会返回两个值
        print("\n开始拓扑恢复 (使用 Dijkstra)...")
        reconstructed_tree, root_node_index = bayesian_topology_reconstruction(
            skeleton_points,
            point_cloud,
            k_graph=K_GRAPH_NEIGHBORS,
            k_assoc=K_ASSOCIATION_NEIGHBORS,
            alpha=ALPHA,
            sigma=SIGMA
        )

        # --- 4. 使用 Open3D 可视化 ---
        # 【已修改】: 不再需要在这里计算 root_node_index
        print("\n准备可视化结果...")
        plot_results_o3d(
            skeleton_points,
            point_cloud,
            reconstructed_tree,
            root_node_index
        )