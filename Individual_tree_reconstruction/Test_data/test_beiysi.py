import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import os
import open3d as o3d
import warnings
from collections import defaultdict
from tqdm import tqdm
import heapq  # 优先队列库
import math

# 忽略计算中可能出现的无效值警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
# ***********************************************************************************
#
#                      基于贝叶斯推断的树木拓扑重建算法
#
# 核心思想:
# 本算法的核心思想是采用一个基于贝叶斯推断的、高性能的“贪心生长”策略，用于从给定的
# 骨架点集和原始点云中，稳健地恢复树状拓扑结构。
#
# 与传统的最小生成树（MST）等方法不同，本算法的创新之处在于，它并非依赖单一的几何
# 权重（如边长），而是将决策依据建立在一个综合了“数据证据”和“先验知识”的后验概
# 率模型上。这使得算法能够智能地避免在稀疏或噪声区域产生“抄近道”等拓扑错误。
#
# 算法框架:
# 算法通过一个类似于经典Prim算法的框架，从一个根节点开始逐步构建最终的树。在生长的
# 每一步，算法都会从众多连接着现有树内和树外的候选边中，选择一条能使拓扑“最优”的
# 边进行添加。“最优”的评判标准（即边的权重）由以下三个关键部分动态计算得出：
#
# 1.  数据似然度 (Data Likelihood):
#     此项是算法感知数据的核心。基于“最近边”归属原则，它精确地量化了每条候选边
#     与局部点云数据的拟合程度。具体通过计算归属点云在边上投影的方差来实现，方
#     差越小，说明边越处在点云的中心，其似然度就越高。这个机制是避免在点云稀疏
#     区域产生错误连接的关键。
#
# 2.  长度先验 (Length Prior):
#     一个简单的几何约束，它对较长的边施加惩罚，鼓励算法建立局部连接，这符合自
#     然结构的一般规律。该项的强度由超参数`alpha`控制。
#
# 3.  角度先验 (Angle Prior):
#     这是保证拓扑平滑性的核心。在生长的每一步，算法都会评估候选边与当前生长方
#     向的夹角，并严厉惩罚形成锐角的、不自然的“拐点”。这使得算法倾向于沿着连
#     贯、平滑的路径延伸，有效抑制了跨分支的“跳线”。该项的强度由超参数`beta`
#     控制。
#
# 工作流程:
# 整个流程首先对所有候选边预计算其基础权重（似然度+长度先验），然后利用一个优先
# 队列（Min-Heap）来高效管理这些候选边。在从队列中提取最优边时，动态地加入角
# 度先验的考量，最终以较高的效率（近似O(E log V)）逐步构建出全局后验概率最高
# 的树拓扑。
#
# ***********************************************************************************

# ******************************************************
# 1. 几何辅助函数 (无变化)
# ******************************************************

def project_point_onto_line(p, a, b):
    ap = p - a
    ab = b - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-9: return a
    t = np.dot(ap, ab) / ab_len_sq
    projection = a + t * ab
    return projection


def distance_point_to_segment(p, a, b):
    ab = b - a
    len_sq = np.dot(ab, ab)
    if len_sq == 0.0: return np.linalg.norm(p - a)
    t = np.dot(p - a, ab) / len_sq
    if 0 <= t <= 1:
        projection = a + t * ab
        return np.linalg.norm(p - projection)
    return min(np.linalg.norm(p - a), np.linalg.norm(p - b))


# ******************************************************
# 2. 核心算法：带角度先验的高性能贪心生长
# ******************************************************

def greedy_reconstruction_with_angle_prior(
        skeleton_points,
        point_cloud,
        root_node,
        k,
        alpha,
        sigma,
        beta  # 新增：角度先验的强度系数
):
    """
    使用集成了角度先验的高性能贪心生长算法。
    """
    print("--- 开始运行“带角度先验的贪心生长”算法 ---")
    num_pts = len(skeleton_points)

    # --- 步骤 1: 构建候选图并计算局部似然度和长度先验 ---
    skeleton_kdtree = KDTree(skeleton_points)
    candidate_graph = nx.Graph()
    incident_edges = defaultdict(list)

    print("1. 构建候选图...")
    for i in tqdm(range(num_pts), desc="构建候选图"):
        _, indices = skeleton_kdtree.query(skeleton_points[i], k=k + 1)
        for j_idx in indices[1:]:
            u, v = tuple(sorted((i, j_idx)))
            if not candidate_graph.has_edge(u, v):
                candidate_graph.add_edge(u, v)
                incident_edges[u].append((u, v))
                incident_edges[v].append((u, v))
    print(f"   候选图构建完成: {candidate_graph.number_of_nodes()} 个节点, {candidate_graph.number_of_edges()} 条边。")

    print("2. 预计算边的局部似然度和长度先验...")
    edge_associations = defaultdict(list)
    pcd_kdtree = KDTree(point_cloud)

    for p in tqdm(point_cloud, desc="关联点云到最近边"):
        _, K_indices = skeleton_kdtree.query(p, k=min(10, num_pts))
        min_dist, best_edge = np.inf, None
        local_candidate_edges = {edge for idx in K_indices for edge in incident_edges[idx]}
        if not local_candidate_edges: continue
        for u, v in local_candidate_edges:
            dist = distance_point_to_segment(p, skeleton_points[u], skeleton_points[v])
            if dist < min_dist:
                min_dist, best_edge = dist, (u, v)
        if best_edge:
            edge_associations[best_edge].append(p)

    for u, v in tqdm(candidate_graph.edges(), desc="计算基础权重"):
        p1, p2 = skeleton_points[u], skeleton_points[v]
        edge_length = np.linalg.norm(p1 - p2)
        log_prior_len = -alpha * edge_length

        associated_points = np.array(edge_associations.get((u, v), []))
        if len(associated_points) < 2:
            log_likelihood = -np.inf
        else:
            projections = np.array([project_point_onto_line(pt, p1, p2) for pt in associated_points])
            direction_vec = (p2 - p1) / edge_length if edge_length > 1e-9 else np.array([1, 0, 0])
            coords_on_line = np.dot(projections - p1, direction_vec)
            variance = np.var(coords_on_line)
            log_likelihood = -variance / (2 * sigma ** 2) - len(associated_points) * np.log(sigma * np.sqrt(2 * np.pi))

        # 存储不含角度先验的基础权重
        base_weight = -(log_likelihood + log_prior_len)
        candidate_graph[u][v]['base_weight'] = base_weight

    # --- 步骤 3: 使用Prim算法框架和动态权重进行贪心生长 ---
    print("3. 使用Prim框架和动态角度先验进行生长...")
    final_tree = nx.Graph()
    final_tree.add_node(root_node)

    # parent_map 用于在生长过程中追踪每个节点的“父节点”，以计算角度
    parent_map = {root_node: None}

    # 优先队列，存储 (总权重, 节点u, 节点v)
    pq = []

    # 将根节点的所有边加入优先队列
    for neighbor in candidate_graph.neighbors(root_node):
        base_weight = candidate_graph[root_node][neighbor]['base_weight']
        # 根节点没有父节点，角度先验为0
        total_weight = base_weight
        heapq.heappush(pq, (total_weight, root_node, neighbor))

    pbar = tqdm(total=num_pts - 1, desc="执行贪心生长")
    while pq and final_tree.number_of_nodes() < num_pts:
        weight, u, v = heapq.heappop(pq)

        if v in final_tree:
            continue

        final_tree.add_edge(u, v)
        parent_map[v] = u  # 记录v是由u带入树中的
        pbar.update(1)

        # 将新加入的节点v的所有边，计算其包含角度先验的总权重后，加入优先队列
        # 这是角度先验的核心实现
        parent_of_v = u
        vec_pv = skeleton_points[parent_of_v] - skeleton_points[v]
        norm_pv = np.linalg.norm(vec_pv)

        for neighbor in candidate_graph.neighbors(v):
            if neighbor not in final_tree:
                # 基础权重（似然+长度）
                base_weight = candidate_graph[v][neighbor]['base_weight']

                # 计算角度先验
                log_prior_angle = 0
                if norm_pv > 1e-9:
                    vec_vn = skeleton_points[neighbor] - skeleton_points[v]
                    norm_vn = np.linalg.norm(vec_vn)
                    if norm_vn > 1e-9:
                        # cos_theta 越接近-1，代表越平滑（180度）
                        cos_theta = np.dot(vec_pv, vec_vn) / (norm_pv * norm_vn)
                        # (1 + cos_theta) 作为惩罚项，cos_theta=-1时惩罚为0，cos_theta=1时惩罚最大
                        angle_penalty = beta * (1 + cos_theta)
                        log_prior_angle = -angle_penalty  # 先验是log形式，所以惩罚是负值

                # 总权重 = -(似然 + 长度先验 + 角度先验)
                total_weight = base_weight - log_prior_angle
                heapq.heappush(pq, (total_weight, v, neighbor))

    pbar.close()
    print("贪心生长完成！")
    return final_tree


# ******************************************************
# 4. 数据生成与可视化 (无变化)
# ******************************************************
def generate_sample_data(num_branches=4, points_per_branch=50, noise=0.1):
    trunk_skel = np.array([np.array([0, 0, z]) for z in np.linspace(0, 5, 20)])
    skeleton = [trunk_skel]
    angles = np.linspace(0, 2 * np.pi, num_branches + 1)[:-1]
    for i in range(num_branches):
        angle = angles[i]
        branch_skel = np.array(
            [np.array([r * np.cos(angle), r * np.sin(angle), 5 + r * 0.5]) for r in np.linspace(0.1, 3, 15)])
        skeleton.append(branch_skel)
    skeleton = np.vstack(skeleton)
    point_cloud = []
    for _ in range(points_per_branch * skeleton.shape[0]):
        pt = skeleton[np.random.randint(0, skeleton.shape[0])]
        point_cloud.append(pt + np.random.randn(3) * noise)
    return skeleton, np.array(point_cloud)


def plot_results_o3d(skeleton, point_cloud, final_graph, root_node_idx):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    skel_pcd = o3d.geometry.PointCloud()
    skel_pcd.points = o3d.utility.Vector3dVector(skeleton)
    skel_pcd.paint_uniform_color([0.1, 0.1, 0.8])
    line_indices = list(final_graph.edges())
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(skeleton),
                                    lines=o3d.utility.Vector2iVector(line_indices))
    line_set.paint_uniform_color([1.0, 0.0, 0.0])
    root_pos = skeleton[root_node_idx]
    bbox_diag = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    root_sphere_radius = bbox_diag * 0.015
    root_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=root_sphere_radius)
    root_sphere.translate(root_pos)
    root_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    print("\n正在打开Open3D可视化窗口... 在窗口中按 'q' 键可关闭。")
    o3d.visualization.draw_geometries([pcd, skel_pcd, line_set, root_sphere],
                                      window_name="带角度先验的贪心生长拓扑恢复", width=1600, height=900)


# ******************************************************
# 5. 数据加载函数 (无变化)
# ******************************************************
def load_data_from_files(point_cloud_path, skeleton_path):
    print("--- 开始从文件加载数据 ---")
    point_cloud, skeleton_points = None, None
    if not os.path.exists(point_cloud_path) or not os.path.exists(skeleton_path):
        return None, None
    try:
        point_cloud = np.loadtxt(point_cloud_path)
        print(f"✅ 成功加载点云 ({point_cloud.shape[0]} 个点): {point_cloud_path}")
    except Exception as e:
        print(f"❌ 加载点云文件时发生错误: {e}")
        return None, None
    try:
        skeleton_points = np.loadtxt(skeleton_path, skiprows=1, usecols=(0, 1, 2))
        print(f"✅ 成功加载骨架 ({skeleton_points.shape[0]} 个点): {skeleton_path}")
    except Exception as e:
        print(f"❌ 加载骨架文件时发生错误: {e}")
        return None, None
    return skeleton_points, point_cloud


# ******************************************************
# 6. 主程序入口
# ******************************************************
if __name__ == '__main__':
    SKELETON_FILE = r'E:\Waterdropletshrinkage\Ours\result_knn_l1_ske\结果部分(最佳参数)\我们的结果\未优化的结果\Tree_1.txt'
    POINT_CLOUD_FILE = r'E:\Waterdropletshrinkage\Ours\result_knn_l1_ske\Data\Tree_1.txt'

    if not os.path.exists(SKELETON_FILE) or not os.path.exists(POINT_CLOUD_FILE):
        print("警告: 未找到指定的数据文件，正在生成示例文件用于演示...")
        current_dir = os.getcwd()
        SKELETON_FILE = os.path.join(current_dir, 'skeleton_demo.txt')
        POINT_CLOUD_FILE = os.path.join(current_dir, 'point_cloud_demo.txt')
        skel_data, pc_data = generate_sample_data(num_branches=3, points_per_branch=80, noise=0.2)
        np.savetxt(POINT_CLOUD_FILE, pc_data, fmt='%.6f')
        np.savetxt(SKELETON_FILE, skel_data, fmt='%.6f', header='x y z', comments='')
        print(f"已创建示例文件 '{SKELETON_FILE}' 和 '{POINT_CLOUD_FILE}'")

    skeleton_points, point_cloud = load_data_from_files(point_cloud_path=POINT_CLOUD_FILE, skeleton_path=SKELETON_FILE)

    if skeleton_points is None or point_cloud is None:
        print("\n数据加载失败，程序已终止。")
    else:
        # --- 设置超参数 ---
        K_NEIGHBORS = 8  # 候选图中每个骨架点连接的邻居数 (可适当增加以确保正确路径存在)
        ALPHA = 1.0  # 长度惩罚系数 (prior)
        SIGMA = 0.2  # 投影点分布的标准差 (likelihood)
        BETA = 5.0  # 角度先验强度系数 (关键！可调)
        # BETA越大，算法越倾向于平滑生长，越能避免“抄近道”

        # --- 运行拓扑恢复算法 ---
        root_node_index = np.argmin(skeleton_points[:, 2])

        reconstructed_tree = greedy_reconstruction_with_angle_prior(
            skeleton_points,
            point_cloud,
            root_node=root_node_index,
            k=K_NEIGHBORS,
            alpha=ALPHA,
            sigma=SIGMA,
            beta=BETA
        )

        # --- 可视化 ---
        print("\n准备可视化结果...")
        plot_results_o3d(skeleton_points, point_cloud, reconstructed_tree, root_node_index)