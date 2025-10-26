import numpy as np
from scipy.spatial import distance
from scipy.optimize import least_squares


# 读取骨架数据
def read_skeleton(file_path):
    vertices = []
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts and parts[0] == 'l':
                i, j = int(parts[1]) - 1, int(parts[2]) - 1
                edges.append([i, j])
    return vertices, edges


# 读取点云数据
def read_point_cloud(file_path):
    point_cloud = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 3:
                print(f"错误：行数据不足3个值: {line.strip()}")
                continue
            try:
                point_cloud.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError as e:
                print(f"错误：行无法转换为浮点数: {line.strip()} ({e})")
                continue
    if not point_cloud:
        raise ValueError("点云文件中没有有效数据")
    return np.array(point_cloud)


# 构建树结构
def build_tree(edges, n_vertices):
    adj = [[] for _ in range(n_vertices)]
    for i, j in edges:
        if i >= n_vertices or j >= n_vertices or i < 0 or j < 0:
            print(f"错误：边 [{i}, {j}] 超出顶点范围 (0 到 {n_vertices - 1})")
            raise ValueError("边的索引超出顶点数量")
        adj[i].append(j)
        adj[j].append(i)
    return adj


# 计算子树长度和权重
def compute_subtree_lengths(adj, vertices, root=0):
    n = len(vertices)
    lengths = [0] * n
    weights = [0] * n
    visited = [False] * n
    parents = [-1] * n

    def dfs(node, parent):
        visited[node] = True
        parents[node] = parent
        subtree_len = 0
        for child in adj[node]:
            if not visited[child]:
                edge_len = distance.euclidean(vertices[node], vertices[child])
                subtree_len += edge_len + dfs(child, node)
        lengths[node] = subtree_len
        return subtree_len

    dfs(root, -1)

    for node in range(n):
        if len(adj[node]) == 1 and node != root:
            parent = parents[node]
            lengths[node] = lengths[parent]
            weights[node] = distance.euclidean(vertices[node], vertices[parent])
        else:
            weights[node] = lengths[node]

    weights[root] = sum(lengths)
    return lengths, weights, parents


# 拟合树底部半径
def fit_root_radius(point_cloud):
    print('拟合底部1%的点云')
    points = point_cloud[point_cloud[:, 2].argsort()]
    print(f"点云总数: {points.shape[0]}")

    num_bottom_points = max(points.shape[0] // 100, 3)
    bottom_points = points[:num_bottom_points, :]
    print(f"底部1%点数 (过滤前): {bottom_points.shape[0]}")

    min_x, max_x = np.quantile(bottom_points[:, 0], [0.05, 0.95])
    min_y, max_y = np.quantile(bottom_points[:, 1], [0.05, 0.95])
    min_z, max_z = np.quantile(bottom_points[:, 2], [0.05, 0.95])
    print(f"底部点分位数范围: X [{min_x}, {max_x}], Y [{min_y}, {max_y}], Z [{min_z}, {max_z}]")

    select_mask = (
            (bottom_points[:, 0] >= min_x) & (bottom_points[:, 0] <= max_x) &
            (bottom_points[:, 1] >= min_y) & (bottom_points[:, 1] <= max_y) &
            (bottom_points[:, 2] >= min_z) & (bottom_points[:, 2] <= max_z)
    )
    bottom_points = bottom_points[select_mask]
    print(f"过滤后点数: {bottom_points.shape[0]}")

    if bottom_points.shape[0] < 3:
        print("警告：过滤后点数不足3个，无法拟合圆，使用默认半径 1.0")
        return 1.0

    def circle_residuals(k, points):
        return (points[:, 0] - k[0]) ** 2 + (points[:, 1] - k[1]) ** 2 - k[2] ** 2

    initial_parameters = np.array([0.0, 0.0, 1.0])
    result = least_squares(circle_residuals, initial_parameters, args=(bottom_points,))

    radius = abs(result.x[2])
    print('拟合树底部半径完成')
    return radius


# 计算半径（恢复之前公式）
def compute_radii(r0, adj, lengths, parents, root=0):
    n = len(adj)
    radii = [0.0] * n
    visited = [False] * n

    def get_children(node):
        return [child for child in adj[node] if child != parents[node]]

    def dfs_radius(node):
        visited[node] = True
        children = get_children(node)

        if children:
            if len(children) == 1:
                child = children[0]
                radii[child] = radii[node] * (lengths[child] / lengths[node]) ** 1.5
            else:
                total_child_length = sum(lengths[c] for c in children)
                for child in children:
                    radii[child] = radii[node] * (lengths[child] / total_child_length) ** (1 / 2.49)

        for child in children:
            if not visited[child]:
                dfs_radius(child)

    radii[root] = r0
    dfs_radius(root)
    return radii


# 保存结果到TXT文件
def save_results(filename, vertices, lengths, radii):
    with open(filename, 'w') as f:
        for v, l, r in zip(vertices, lengths, radii):
            f.write(f"{v[0]} {v[1]} {v[2]} {l} {r}\n")


# 主程序
def process_tree(skeleton_file, point_cloud_file, output_file):
    vertices, edges = read_skeleton(skeleton_file)
    point_cloud = read_point_cloud(point_cloud_file)

    n_vertices = len(vertices)
    print(f"顶点数量: {n_vertices}")
    print(f"边数量: {len(edges)}")
    max_edge_index = max(max(i, j) for i, j in edges)
    print(f"边的最大索引: {max_edge_index}")

    adj = build_tree(edges, n_vertices)
    lengths, weights, parents = compute_subtree_lengths(adj, vertices)
    r0 = fit_root_radius(point_cloud)
    radii = compute_radii(r0, adj, lengths, parents)
    save_results(output_file, vertices, lengths, radii)

    return r0, radii, vertices, lengths


# 文件路径
skeleton_file = "E:\\图拓扑优化与局部特征增强\\Result_tupo\\ours\\DMST\\Tree_1_processed_202503051024.obj"
point_cloud_file = "E:\\图拓扑优化与局部特征增强\\Ours\\骨架点产生\\Data\\L1_data\\Tree_1.txt"
output_file = "tree1_with_lengths_radii.txt"

# 运行
try:
    r0, radii, vertices, lengths = process_tree(skeleton_file, point_cloud_file, output_file)
    print("根节点半径:", r0)
    print(f"结果已保存到: {output_file}")
    print("前5个节点的结果:")
    for i in range(min(5, len(vertices))):
        print(f"节点 {i}: 坐标 {vertices[i]}, 子树长度 {lengths[i]}, 半径 {radii[i]}")
except Exception as e:
    print(f"发生错误: {e}")