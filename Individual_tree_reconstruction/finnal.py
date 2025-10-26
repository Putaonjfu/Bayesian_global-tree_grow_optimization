import numpy as np
import open3d as o3d
from scipy.optimize import minimize,least_squares
from scipy.spatial import KDTree
import networkx as nx




def read_skeleton_from_ply(ply_path):
    try:
        line_set = o3d.io.read_line_set(ply_path)
        if not line_set.has_points() or not line_set.has_lines():
            raise ValueError("PLY文件为空或不包含顶点/边。")
        vertices = np.asarray(line_set.points)
        edges = np.asarray(line_set.lines)
        print(f"成功从 {ply_path} 读取骨架：")
        print(f"  - 顶点数量: {len(vertices)}")
        print(f"  - 边数量: {len(edges)}")
        return vertices, edges
    except Exception as e:
        print(f"错误：无法读取骨架文件 {ply_path}。")
        print(e)
        return None, None


def read_point_cloud_with_numpy(cloud_path):
    print(f"正在使用Numpy直接读取点云文件: {cloud_path}")
    try:
        points = np.loadtxt(cloud_path, usecols=(0, 1, 2), ndmin=2)
        if points.shape[0] == 0:
            raise ValueError("文件中没有可解析的坐标行。")
        print(f"成功从 {cloud_path} 读取点云，点数: {len(points)}")
        return points
    except Exception as e:
        print(f"错误：使用Numpy读取点云文件 {cloud_path} 失败。")
        print(f"  - 详细错误: {e}")
        return None


# 将TLS点分配给最近的骨架边
def associate_points_to_edges(skeleton_vertices, skeleton_edges, point_cloud):
    print("正在将点云关联到骨架边 (已优化)...")
    num_vertices = len(skeleton_vertices)
    vertex_to_edge_map = {i: [] for i in range(num_vertices)}
    for edge_idx, (v1, v2) in enumerate(skeleton_edges):# 遍历每个边取出所有的顶点并分别加到列表中
        vertex_to_edge_map[v1].append(edge_idx)# 将每个顶点对应的边的索引添加到V1与V2中
        vertex_to_edge_map[v2].append(edge_idx)# 将每个顶点对应的边的索引添加到V1与V2中
    print("  - 顶点到边的索引已建立。")
    edge_to_points = {i: [] for i in range(len(skeleton_edges))}
    vertex_tree = KDTree(skeleton_vertices)
    point_count = len(point_cloud)
    for pt_idx, point in enumerate(point_cloud):
        if pt_idx > 0 and pt_idx % (point_count // 10 or 1) == 0:
            progress = int(pt_idx / point_count * 100)
            print(f"  - 关联进度: {progress}%")
        _, nearest_vertex_indices = vertex_tree.query(point, k=10)# 找到当前tls点最近的10个骨架点
        min_dist_sq = float('inf')
        best_edge_idx = -1
        candidate_edges = set()
        for v_idx in nearest_vertex_indices:# 遍历这10个骨架点
            candidate_edges.update(vertex_to_edge_map[v_idx])# 将这10个骨架点对应的边取出来
        for edge_idx in candidate_edges:#遍历这10个骨架点所对应的边
            p1_idx, p2_idx = skeleton_edges[edge_idx]# 边顶点坐标索引
            p1, p2 = skeleton_vertices[p1_idx], skeleton_vertices[p2_idx]#坐标
            line_vec = p2 - p1
            point_vec = point - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue
            t = np.dot(point_vec, line_vec) / line_len_sq
            if 0 <= t <= 1:# 如果这个tls点投影在这条边中间
                projection = p1 + t * line_vec
                dist_sq = np.sum((point - projection) ** 2)#计算垂直距离
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_edge_idx = edge_idx
        if best_edge_idx != -1:
            edge_to_points[best_edge_idx].append(point)
    print("点云关联和筛选完成。")
    return {k: np.array(v) for k, v in edge_to_points.items() if v}

# 计算一组点到一条边的平均距离
def calculate_average_distance_radius(points, edge_p1, edge_p2):
    if len(points) < 5: return None #若分配的tls点少于5，则无半径
    line_vec = edge_p2 - edge_p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0: return None #若两个边的点重复，则无半径
    point_vecs = points - edge_p1
    distances = np.linalg.norm(np.cross(point_vecs, line_vec), axis=1) / np.sqrt(line_len_sq)
    return np.mean(distances)# 计算这些点到这边的平均距离作为半径

# 找到Z值最低的tls点，然后在网上0.1m的点作为候选点，将其投影到xoy上拟合一个圆，其半径作为参考树半径
def calculate_root_radius_2d_fit(point_cloud):
    print("正在为根节点进行特殊的2D圆形拟合...")
    if point_cloud is None or len(point_cloud) == 0: return None
    z_min = np.min(point_cloud[:, 2])
    z_max_slice = z_min + 0.1
    bottom_points = point_cloud[(point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] < z_max_slice)]
    if len(bottom_points) < 10:
        print("警告：底部0.1米范围内的点过少，无法进行根节点半径拟合。")
        return None
    points_2d = bottom_points[:, :2]

    def circle_residuals(params, x, y):
        xc, yc, r = params
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    mean_x, mean_y = np.mean(points_2d, axis=0)
    initial_guess = [mean_x, mean_y, np.std(points_2d)]
    result = least_squares(circle_residuals, initial_guess, args=(points_2d[:, 0], points_2d[:, 1]))
    if result.success:
        fitted_radius = abs(result.x[2])
        print(f"  - 根节点半径拟合成功，半径: {fitted_radius:.4f}")
        return fitted_radius
    else:
        print("警告：根节点2D圆形拟合失败。")
        return None


# 根据骨架拓扑，利用BFS 获取度为2的节点（位于分支），度为1的节点（叶子及树根），以及度大于2的节点（分叉点）并用于后续半径优化
def build_graph_and_identify_nodes(vertices, edges):
    G = nx.Graph() #创建一个图结构
    G.add_edges_from(edges) #添加边
    if len(vertices) == 0: return G, -1, {}, [], []
    root_node = np.argmin(vertices[:, 2]) #根节点为Z值最低的点
    print(f"根据最低Z值，已确定根节点为: 节点 {root_node}")
    parents = {root_node: None} # 根节点的父节点为空
    queue = [root_node]
    visited = {root_node}
    head = 0
    # 利用BFS 获取度为2的节点（位于分支），度为1的节点（叶子及树根），以及度大于2的节点（分叉点）
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                parents[v] = u
                queue.append(v)
    fork_nodes, smooth_nodes, leaf_nodes = [], [], []
    for i in range(len(vertices)):
        if i == root_node: continue
        degree = G.degree(i)
        if parents.get(i) is not None and degree > 2:
            fork_nodes.append(i)
        elif degree == 2:
            smooth_nodes.append(i)
        elif degree == 1:
            leaf_nodes.append(i)
    return G, root_node, parents, fork_nodes, smooth_nodes

#对每个节点的半径全局优化
def run_global_optimization(vertices, G, parents, fork_nodes, smooth_nodes, true_radii):
    print("开始全局优化...")
    num_nodes = len(vertices)
    initial_radii = np.ones(num_nodes) * np.mean(list(true_radii.values())) if true_radii else np.ones(num_nodes) * 0.1 # 让所有真值的半径的平均值作为所有半径的初始猜测值
    w_data, w_smooth, w_pipe = 0.1, 0.1, 0.1 # 数据保真项、平滑项、拓扑约束项的权重

    def objective_function(r):
        e_data = sum((r[i] - R_i) ** 2 for i, R_i in true_radii.items()) # 数据保真项
        e_smooth = 0
        for i in smooth_nodes: # 找到度为2的节点
            p = parents.get(i) # 找到他的父节点
            if p is not None:
                children = [n for n in G.neighbors(i) if n != p] # 找到他的子节点
                if len(children) == 1:
                    c = children[0]
                    e_smooth += (r[i] - (r[p] + r[c]) / 2) ** 2 # 节点i的半径ri与其父节点的半径与子节点的半径的平均值之间的平方差
        e_pipe = 0 # 拓扑约束项
        for i in fork_nodes: # 找到度为3的节点
            p = parents.get(i) # 找到他的父节点
            if p is not None:
                children = [n for n in G.neighbors(i) if n != p]# 找到他的所有子节点
                if children:
                    sum_children_r_pow = sum(r[c] ** 2.49 for c in children) # 所有子节点半径的2.49次方之和
                    e_pipe += (r[i] ** 2.49 - sum_children_r_pow) ** 2 #父节点的半径的2.49放与子节点的2.49次之和的差值
        return w_data * e_data + w_smooth * e_smooth + w_pipe * e_pipe # 返回所有度为1、2以及大于3的节点的能量项的加权和

    bounds = [(0.0001, None) for _ in range(num_nodes)] # 为每个半径设置一下极小的值防止为0
    result = minimize(objective_function, initial_radii, method='L-BFGS-B', bounds=bounds) # 全局最小化求解所有节点的半径。
    if result.success:
        print("全局优化成功！")
    else:
        print(f"警告：全局优化可能未完全收敛。消息: {result.message}")
    return result.x

# 找到与输入向量垂直的向量
def find_orthogonal_vector(vec):
    vec = vec / np.linalg.norm(vec)
    if abs(vec[0]) > 0.9:
        ortho = np.cross(vec, [0, 1, 0])
    else:
        ortho = np.cross(vec, [1, 0, 0])
    return ortho / np.linalg.norm(ortho)

# 为一条树枝（一系列点与半径创建圆管网格），具体而言，给每个点利用对应的半径在与方向向量的方向上建立一个n个顶点的圆面，然后再根据拓扑，在相邻两个面之间构建三角面片，形成整个单条树枝的模型。
def create_strict_tube_mesh(points, radii, slices=20):
    if len(points) < 2 or len(points) != len(radii): return o3d.geometry.TriangleMesh()
    all_vertices, all_triangles = [], [] # 储存最终的网格和三角面片
    cross_sections_verts, perp = [], None # 每个截面的顶点，用于绘制截面的垂直向量
    for i in range(len(points)): # 遍历这个分支的点
        r, s = radii[i], points[i] # 对应半径与顶点坐标
        if i < len(points) - 1: # 如果这个点不是最后一个点
            axis = points[i + 1] - s # 则他的方向向量指向下一个点
        else:
            axis = s - points[i - 1] # 路径最后一个点的方向向量跟上一个点一致
        axis_norm = np.linalg.norm(axis) #归一化方向向量
        if axis_norm < 1e-6: # 如果这两个点重合
            if i > 0 and cross_sections_verts: cross_sections_verts.append(cross_sections_verts[-1] - points[i - 1] + s)
            continue
        axis /= axis_norm
        if perp is None: # 对于路径的第一个点
            perp = find_orthogonal_vector(axis) # 找到与方向向量垂直的那个向量
        else: # 其他路径的节点，利用格拉姆-施密特正交化，取上一个截面的perp向量，并移除它在当前axis方向上的分量，这使得新的perp向量既垂直于当前的axis，又尽可能地“接近”上一个perp向量。这可以防止圆管在树枝弯曲时发生不必要的“扭曲”。
            perp = perp - np.dot(perp, axis) * axis
            perp_norm = np.linalg.norm(perp)
            if perp_norm < 1e-6:
                perp = find_orthogonal_vector(axis)
            else:
                perp /= perp_norm
        current_cross_section = []
        for j in range(slices):
            angle = 2.0 * np.pi * j / slices
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            rotated_vec = R @ (perp * r)
            current_cross_section.append(s + rotated_vec)
        cross_sections_verts.append(np.array(current_cross_section)) # 一个列表，里面又包含N个列表，每个子列表包含20个3D顶点坐标。
    vert_offset, cross_sections_indices = 0, [] # 将所有顶点合并到一个大列表，并记录每个截面的索引范围
    for cs_verts in cross_sections_verts:
        all_vertices.extend(cs_verts)
        # 记录这20个顶点在大列表中的全局索引
        indices = np.arange(vert_offset, vert_offset + len(cs_verts))
        cross_sections_indices.append(indices)
        vert_offset += len(cs_verts) # all_vertices 是一个包含（N * 20）个顶点的一维列表。


        # 在相邻的两个截面之间创建三角形
    for i in range(len(cross_sections_indices) - 1):
        cs_curr, cs_next = cross_sections_indices[i], cross_sections_indices[i + 1]
        for j in range(slices):
            v1, v2 = cs_curr[j], cs_curr[(j + 1) % slices]
            v3, v4 = cs_next[j], cs_next[(j + 1) % slices]
            all_triangles.append([v1, v3, v2])
            all_triangles.append([v2, v3, v4])
    # 创建并返回网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(all_vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(all_triangles))
    return mesh

# 找到所有树枝的“起点”（根节点和所有分叉点），从每个“起点”出发，沿着它的“子节点”向下追踪，直到遇到下一个分叉点或叶节点，从而提取出一条完整的、不间断的树枝路径。
# 把这条路径（点坐标 + 优化后的半径）交给 create_strict_tube_mesh 函数，让它生成这条树枝的网格。
# 收集所有单独的树枝网格。将所有网格合并成一个。

def reconstruct_mesh_from_skeleton(vertices, G, parents, final_radii, root_node):
    print("正在从优化后的骨架和半径重建最终网格...")
    branch_starts = [root_node] + [node for node, deg in G.degree() if deg > 2] # 找到根节点以及度大于2的节点（他肯定是分支的起始点）
    all_meshes = []
    #找到所有的分支，以及半径并重建添加格网
    for start_node in branch_starts:
        for neighbor in G.neighbors(start_node):
            if parents.get(neighbor) == start_node:
                current_path_indices = [start_node, neighbor]
                curr = neighbor
                while G.degree(curr) == 2:
                    next_node = [n for n in G.neighbors(curr) if n != parents.get(curr)][0]
                    current_path_indices.append(next_node)
                    curr = next_node
                path_points = vertices[current_path_indices]
                path_radii = final_radii[current_path_indices]
                if len(path_points) > 1:
                    print(f"  - 正在重建分支: {current_path_indices}")
                    branch_mesh = create_strict_tube_mesh(path_points, path_radii)
                    all_meshes.append(branch_mesh)
    if not all_meshes:
        print("警告：没有生成任何分支网格。")
        return o3d.geometry.TriangleMesh()
    print("合并所有分支网格...")
    final_mesh = all_meshes[0]
    for i in range(1, len(all_meshes)): final_mesh += all_meshes[i]
    final_mesh.compute_vertex_normals()
    return final_mesh


# --- 5. 主流程 ---
def main():
    # --- 用户需要配置的参数 ---
    skeleton_ply_file = r"E:\PHD\Waterdropletshrinkage\Ours\Skeletonpointgeneration\code_refine\reconstruction\Test_data\Tree_1_major_iter_2.ply"  # <--- 请将这里替换成您的PLY文件名

    # 2. 密集点云文件路径
    point_cloud_file = r"E:\PHD\Waterdropletshrinkage\Ours\Skeletonpointgeneration\Data\L1_data\Tree_1.txt"  # <--- 请将这里替换成您的点云文件名

    # 步骤1：加载数据
    vertices, edges = read_skeleton_from_ply(skeleton_ply_file)
    point_cloud = read_point_cloud_with_numpy(point_cloud_file)

    if vertices is None or point_cloud is None:
        print("数据加载失败，程序退出。")
        return

    # 步骤2: 确定骨架拓扑和根节点
    G, root_node, parents, fork_nodes, smooth_nodes = build_graph_and_identify_nodes(vertices, edges)

    # 步骤3: 关联点云到边
    edge_points_map = associate_points_to_edges(vertices, edges, point_cloud)

    # 步骤4: 计算所有"真值"半径
    print("正在进行局部半径计算以获取'真值'...")
    true_radii = {}

    # 4.1 对根节点进行特殊处理
    root_radius = calculate_root_radius_2d_fit(point_cloud)
    if root_radius is not None:
        true_radii[root_node] = root_radius

    # 4.2 对其他所有边，计算平均距离半径
    for edge_idx, points in edge_points_map.items():
        p1_idx, p2_idx = edges[edge_idx]
        radius = calculate_average_distance_radius(points, vertices[p1_idx], vertices[p2_idx])

        if radius is not None:
            child_idx = p2_idx if parents.get(p2_idx) == p1_idx else p1_idx
            if child_idx not in true_radii:
                true_radii[child_idx] = radius

    if not true_radii:
        print("警告：没有任何“真值”半径被成功计算。")

    # 步骤5: 全局优化
    final_radii = run_global_optimization(vertices, G, parents, fork_nodes, smooth_nodes, true_radii)

    # 步骤6: 输出与可视化 (已更新)
    print("\n--- 重建结果 ---")
    for i in range(len(vertices)):
        print(f"节点 {i}: 坐标 {np.round(vertices[i], 3)}, 最终半径: {final_radii[i]:.4f}")

    final_tree_mesh = reconstruct_mesh_from_skeleton(vertices, G, parents, final_radii, root_node)

    if final_tree_mesh.has_triangles():
        final_tree_mesh.paint_uniform_color([0.6, 0.4, 0.2])  # 棕色模型
        print("显示最终重建的三维模型与原始点云...")

        # --- 新增：创建点云可视化对象 ---
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_vis.paint_uniform_color([0.8, 0.8, 0.8])  # 灰色点云

        # --- 将模型和点云一起显示 ---
        o3d.visualization.draw_geometries(
            [final_tree_mesh, pcd_vis],
            window_name="最终重建模型与原始点云"
        )
    else:
        print("未能生成有效的三维模型。")


if __name__ == '__main__':
    main()