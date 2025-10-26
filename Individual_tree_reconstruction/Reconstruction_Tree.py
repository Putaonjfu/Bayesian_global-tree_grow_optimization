import numpy as np
import open3d as o3d  # 用于读取PLY和点云操作
from scipy.optimize import least_squares, minimize
from scipy.spatial import KDTree
import networkx as nx  # 用于图结构分析


# --- 1. 数据加载模块 ---

def read_skeleton_from_ply(ply_path):
    """
    从PLY文件中读取骨架的顶点和边。
    文件格式需符合用户提供的截图。
    """
    try:
        # Open3D的LineSet可以很好地处理这种格式
        line_set = o3d.io.read_line_set(ply_path)
        vertices = np.asarray(line_set.points)
        edges = np.asarray(line_set.lines)
        print(f"成功从 {ply_path} 读取骨架：")
        print(f"  - 顶点数量: {len(vertices)}")
        print(f"  - 边数量: {len(edges)}")
        return vertices, edges
    except Exception as e:
        print(f"错误：无法读取骨架文件 {ply_path}。请确保文件格式正确。")
        print(e)
        return None, None


# --- 新的、更强大的点云读取函数 ---
def read_point_cloud_robust(cloud_path):
    """
    使用Numpy从文本文件中稳健地读取点云。
    这个函数更强大，能处理不规范的文本文件。
    """
    print(f"正在尝试用Numpy读取点云文件: {cloud_path}")
    try:
        # np.loadtxt非常强大：
        # - usecols=(0, 1, 2) 明确告诉它只使用前三列
        # - comments='#' or '//' 可以忽略注释行
        # - ndmin=2 确保即使只有一行数据也返回二维数组
        points = np.loadtxt(cloud_path, usecols=(0, 1, 2), ndmin=2)

        if points.shape[0] == 0:
            raise ValueError("文件中没有可解析的坐标行。")

        print(f"成功从 {cloud_path} 读取点云，点数: {len(points)}")
        return points
    except Exception as e:
        print(f"错误：使用Numpy读取点云文件 {cloud_path} 失败。")
        print("  - 请重点检查文件内容是否存在格式问题（如非数字、错误的分隔符等）。")
        print(f"  - 详细错误: {e}")
        return None


# --- 2. 几何与数据关联模块 ---

# --- 2. 几何与数据关联模块 (已优化) ---
def associate_points_to_edges(skeleton_vertices, skeleton_edges, point_cloud):
    """
    【已优化】将密集点云中的每个点关联到最近的骨架边上。
    """
    print("正在将点云关联到骨架边 (已优化)...")

    # --- 关键优化点 1: 预先建立“顶点 -> 边”的索引地图 ---
    num_vertices = len(skeleton_vertices)
    vertex_to_edge_map = {i: [] for i in range(num_vertices)}
    for edge_idx, (v1, v2) in enumerate(skeleton_edges):
        vertex_to_edge_map[v1].append(edge_idx)
        vertex_to_edge_map[v2].append(edge_idx)
    print("  - 顶点到边的索引已建立。")

    edge_to_points = {i: [] for i in range(len(skeleton_edges))}
    vertex_tree = KDTree(skeleton_vertices)
    point_count = len(point_cloud)

    # 主循环遍历所有点云点
    for pt_idx, point in enumerate(point_cloud):
        if pt_idx > 0 and pt_idx % (point_count // 10 or 1) == 0:
            progress = int(pt_idx / point_count * 100)
            print(f"  - 关联进度: {progress}%")

        # 找到最近的10个骨架顶点作为候选
        _, nearest_vertex_indices = vertex_tree.query(point, k=10)

        min_dist_sq = float('inf')
        best_edge_idx = -1

        # --- 关键优化点 2: 使用索引地图快速查找候选边 ---
        candidate_edges = set()
        for v_idx in nearest_vertex_indices:
            # 直接从地图中获取边，无需遍历！
            candidate_edges.update(vertex_to_edge_map[v_idx])

        # 在候选边中找到距离点最近的边 (这部分计算不变)
        for edge_idx in candidate_edges:
            p1_idx, p2_idx = skeleton_edges[edge_idx]
            p1, p2 = skeleton_vertices[p1_idx], skeleton_vertices[p2_idx]

            line_vec = p2 - p1
            point_vec = point - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue

            t = np.dot(point_vec, line_vec) / line_len_sq

            if 0 <= t <= 1:  # 筛选：确保投影点在线段内
                projection = p1 + t * line_vec
                dist_sq = np.sum((point - projection) ** 2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_edge_idx = edge_idx

        if best_edge_idx != -1:
            edge_to_points[best_edge_idx].append(point)

    print("点云关联和筛选完成。")
    return {k: np.array(v) for k, v in edge_to_points.items() if v}


# --- 3. 局部圆柱拟合模块 ---

def fit_cylinder(points):
    """
    对一组点进行圆柱拟合，返回半径和轴线。
    这是一个简化的实现，使用主成分分析（PCA）来确定轴线方向。
    """
    if len(points) < 10:
        return None, None, None  # 点太少，无法可靠拟合

    # 使用PCA确定轴线方向
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mean, covariance = pcd.compute_mean_and_covariance()
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # 轴线方向是最大特征值对应的特征向量
    axis_direction = eigenvectors[:, np.argmax(eigenvalues)]

    # 计算每个点到轴线的距离，以估算半径
    distances = np.linalg.norm(np.cross(points - mean, axis_direction), axis=1)
    radius = np.mean(distances)

    return radius, mean, axis_direction


# --- 4. 全局优化模块 ---

# --- 4. 全局优化模块 ---
def build_graph_and_identify_nodes(vertices, edges):
    """
    【已更新】使用NetworkX构建图，并根据Z值最低点确定根节点，然后识别不同类型的节点。
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    if len(vertices) == 0:
        return G, {}, [], []

    # --- 核心改动：根据Z值最低点确定根节点 ---
    root_node = np.argmin(vertices[:, 2])
    print(f"根据最低Z值，已确定根节点为: 节点 {root_node}")

    # --- 从新的根节点开始进行BFS，建立父子关系 ---
    parents = {root_node: None}
    queue = [root_node]
    visited = {root_node}

    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                parents[v] = u
                queue.append(v)

    # 识别不同类型的节点 (这部分逻辑不变)
    fork_nodes, smooth_nodes, leaf_nodes = [], [], []
    for i in range(len(vertices)):
        # 根节点不属于任何类型
        if i == root_node:
            continue

        degree = G.degree(i)
        # 父节点存在且度大于2的是分叉点
        if parents.get(i) is not None and degree > 2:
            fork_nodes.append(i)
        # 度为2的是普通树干点 (根节点可能度为1或2，但不算)
        elif degree == 2:
            smooth_nodes.append(i)
        # 度为1且不是根节点的是叶节点
        elif degree == 1:
            leaf_nodes.append(i)

    return G, parents, fork_nodes, smooth_nodes


def run_global_optimization(vertices, G, parents, fork_nodes, smooth_nodes, true_radii):
    """
    执行我们最终设计的全局优化方程。
    """
    print("开始全局优化...")

    num_nodes = len(vertices)
    # 初始半径猜测：全部设为1.0或使用真值的平均值
    initial_radii = np.ones(num_nodes) * np.mean(list(true_radii.values())) if true_radii else np.ones(num_nodes)

    # 定义权重
    w_data = 1
    w_smooth = 0.1
    w_pipe = 0.1

    # 定义我们的目标函数 (Eq. 1)
    def objective_function(r):
        # Eq. 2: 数据保真项
        e_data = 0
        for i, R_i in true_radii.items():
            e_data += (r[i] - R_i) ** 2

        # Eq. 3: 平滑正则项
        e_smooth = 0
        for i in smooth_nodes:
            p = parents[i]
            # 找到子节点
            children = [n for n in G.neighbors(i) if n != p]
            if p is not None and len(children) == 1:
                c = children[0]
                e_smooth += (r[i] - (r[p] + r[c]) / 2) ** 2

        # Eq. 4: 拓扑约束项
        e_pipe = 0
        for i in fork_nodes:
            p = parents[i]
            if p is not None:
                children = [n for n in G.neighbors(i) if n != p]
                sum_children_r_pow = sum(r[c] ** 2.49 for c in children)
                e_pipe += (r[i] ** 2.49 - sum_children_r_pow) ** 2

        return w_data * e_data + w_smooth * e_smooth + w_pipe * e_pipe

    # 执行最小化
    # bounds可以防止半径为负
    bounds = [(0.001, None) for _ in range(num_nodes)]
    result = minimize(objective_function, initial_radii, method='L-BFGS-B', bounds=bounds)

    if result.success:
        print("全局优化成功！")
        return result.x
    else:
        print("警告：全局优化可能未收敛。")
        return result.x


# --- 5. 主流程 ---

def main():
    # --- 用户需要配置的参数 ---
    # 1. 骨架文件路径
    skeleton_ply_file = r"E:\PHD\Waterdropletshrinkage\Ours\Skeletonpointgeneration\code_refine\reconstruction\Test_data\Tree_1_major_iter_2.ply"  # <--- 请将这里替换成您的PLY文件名

    # 2. 密集点云文件路径
    point_cloud_file =r"E:\PHD\Waterdropletshrinkage\Ours\Skeletonpointgeneration\Data\L1_data\Tree_1.txt"  # <--- 请将这里替换成您的点云文件名

    # --- 脚本执行 ---
    # 步骤1：加载数据
    vertices, edges = read_skeleton_from_ply(skeleton_ply_file)
    point_cloud = read_point_cloud_robust(point_cloud_file)

    if vertices is None or point_cloud is None:
        print("数据加载失败，程序退出。")
        # 为方便演示，若文件不存在，则创建虚拟数据
        if vertices is None:
            print("创建虚拟骨架数据用于演示...")
            vertices = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 2], [0, -1, 2]])
            edges = np.array([[0, 1], [1, 2], [2, 3], [2, 4]])
        if point_cloud is None:
            print("创建虚拟点云数据用于演示...")
            # 围绕骨架生成一些点
            points = []
            for p1_idx, p2_idx in edges:
                p1, p2 = vertices[p1_idx], vertices[p2_idx]
                for _ in range(200):
                    t = np.random.rand()
                    center = p1 + t * (p2 - p1)
                    radius = 0.2 * (1 - center[2] / 3)  # 半径随高度减小
                    angle = np.random.rand() * 2 * np.pi
                    offset = np.array([np.cos(angle), np.sin(angle), 0]) * radius
                    points.append(center + offset + np.random.randn(3) * 0.01)
            point_cloud = np.array(points)

    # 步骤2 & 3: 关联点云并进行筛选
    edge_points_map = associate_points_to_edges(vertices, edges, point_cloud)

    # 步骤4: 局部拟合，得到"真值"
    print("正在进行局部圆柱拟合以获取'真值'...")
    true_radii = {}
    # 我们将真值半径赋给边的两个端点中的子节点
    G_temp, parents_temp, _, _ = build_graph_and_identify_nodes(vertices, edges)

    for edge_idx, points in edge_points_map.items():
        radius, _, _ = fit_cylinder(points)
        if radius is not None:
            p1_idx, p2_idx = edges[edge_idx]
            # 将半径赋给更远离根节点的那个点
            child_idx = p2_idx if parents_temp.get(p2_idx) == p1_idx else p1_idx
            # 如果已有值，可取平均或覆盖
            true_radii[child_idx] = radius
            print(f"  - 边 {edge_idx} (节点 {p1_idx}-{p2_idx}) -> 拟合半径: {radius:.4f} (赋给节点 {child_idx})")

    if not true_radii:
        print("警告：没有任何边可以成功拟合出'真值'半径。全局优化的数据驱动项将不起作用。")

    # 步骤5: 全局优化
    G, parents, fork_nodes, smooth_nodes = build_graph_and_identify_nodes(vertices, edges)
    final_radii = run_global_optimization(vertices, G, parents, fork_nodes, smooth_nodes, true_radii)

    # 步骤6: 输出结果
    print("\n--- 重建结果 ---")
    for i in range(len(vertices)):
        print(f"节点 {i}: 坐标 {np.round(vertices[i], 3)}, 最终半径: {final_radii[i]:.4f}")

    # 可选：将结果可视化
    # (这里需要更复杂的代码来基于半径创建广义圆柱体网格，Open3D本身不直接支持)
    print("\n提示：要进行三维可视化，需要编写额外的代码将骨架和半径转换成网格模型。")


if __name__ == '__main__':
    main()