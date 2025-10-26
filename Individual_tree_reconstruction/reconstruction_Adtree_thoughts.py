import open3d as o3d
import numpy as np


# ==============================================================================

# ==============================================================================
def find_orthogonal_vector(vec):
    """
    找到一个与给定向量正交的任意向量。
    """
    vec = vec / np.linalg.norm(vec)
    if abs(vec[0]) > 0.9:
        ortho = np.cross(vec, [0, 1, 0])
    else:
        ortho = np.cross(vec, [1, 0, 0])
    return ortho / np.linalg.norm(ortho)


def create_strict_tube_mesh(points, radii, slices=16, cap_ends=True):
    """
    严格按照骨架分段方向重建管状网格，与C++代码逻辑保持一致。
    精度优先，不进行平滑处理。
    """
    if len(points) < 2 or len(points) != len(radii):
        return o3d.geometry.TriangleMesh()

    vertices = []
    triangles = []
    cross_sections = []
    perp = None

    for i in range(len(points) - 1):
        s = points[i]
        t = points[i + 1]
        r = radii[i]

        axis = t - s
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            continue
        axis = axis / axis_norm

        if perp is None:
            perp = find_orthogonal_vector(axis)
        else:
            perp = perp - np.dot(perp, axis) * axis
            perp = perp / np.linalg.norm(perp)

        p0 = s + perp * r

        current_cross_section_indices = []
        for j in range(slices):
            angle = 2.0 * np.pi * j / slices
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            rotated_p = s + R @ (p0 - s)
            vertices.append(rotated_p)
            current_cross_section_indices.append(len(vertices) - 1)
        cross_sections.append(current_cross_section_indices)

    if len(cross_sections) >= 2:
        for i in range(len(cross_sections) - 1):
            cs_curr = cross_sections[i]
            cs_next = cross_sections[i + 1]
            for j in range(slices):
                v1 = cs_curr[j]
                v2 = cs_curr[(j + 1) % slices]
                v3 = cs_next[j]
                v4 = cs_next[(j + 1) % slices]
                triangles.append([v1, v2, v3])
                triangles.append([v2, v4, v3])

    if cap_ends and len(cross_sections) > 0:
        start_center_idx = len(vertices)
        vertices.append(points[0])
        cs_start = cross_sections[0]
        for j in range(slices):
            v1 = cs_start[j]
            v2 = cs_start[(j + 1) % slices]
            triangles.append([start_center_idx, v2, v1])

        end_center_idx = len(vertices)
        vertices.append(points[-1])
        cs_end = cross_sections[-1]
        for j in range(slices):
            v1 = cs_end[j]
            v2 = cs_end[(j + 1) % slices]
            triangles.append([end_center_idx, v1, v2])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()
    return mesh


# ==============================================================================
# 主程序：生成示例数据并执行重建
# ==============================================================================
if __name__ == "__main__":

    # 1. 定义三个分支的数据（主干、左分支、右分支）
    # --------------------------------------------------------------------------
    print("正在生成骨架示例数据...")

    # --- 主干 (Trunk) ---
    # 从 (0,0,0) 延伸到 (0,5,0)，共6个点
    trunk_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 5.0, 0.0]  # 分叉点
    ])
    # 半径从根部的0.8平滑减小到分叉点的0.5
    trunk_radii = np.linspace(0.8, 0.5, len(trunk_points))

    # --- 左分支 (Left Branch) ---
    # 从分叉点 (0,5,0) 延伸到 (-3, 8, 0)，共4个点
    branch1_points = np.array([
        [0.0, 5.0, 0.0],  # 分叉点
        [-1.0, 6.0, 0.0],
        [-2.0, 7.0, 0.0],
        [-3.0, 8.0, 0.0]  # 末端点
    ])
    # 半径从分叉点的0.5平滑减小到末端的0.2
    branch1_radii = np.linspace(0.5, 0.2, len(branch1_points))

    # --- 右分支 (Right Branch) ---
    # 从分叉点 (0,5,0) 延伸到 (4, 7, 0)，共5个点
    branch2_points = np.array([
        [0.0, 5.0, 0.0],  # 分叉点
        [1.0, 5.5, 0.0],
        [2.0, 6.0, 0.0],
        [3.0, 6.5, 0.0],
        [4.0, 7.0, 0.0]  # 末端点
    ])
    # 半径从分叉点的0.5平滑减小到末端的0.2
    branch2_radii = np.linspace(0.5, 0.2, len(branch2_points))

    # 2. 为每个分支独立生成网格
    # --------------------------------------------------------------------------
    print("正在为每个分支重建网格...")
    trunk_mesh = create_strict_tube_mesh(trunk_points, trunk_radii, slices=20)
    branch1_mesh = create_strict_tube_mesh(branch1_points, branch1_radii, slices=20)
    branch2_mesh = create_strict_tube_mesh(branch2_points, branch2_radii, slices=20)

    # 3. 合并网格并进行可视化
    # --------------------------------------------------------------------------
    print("正在合并网格并准备可视化...")
    # 使用 "+" 运算符可以方便地合并多个网格
    tree_mesh = trunk_mesh + branch1_mesh + branch2_mesh
    tree_mesh.paint_uniform_color([0.6, 0.4, 0.2])  # 给树一个棕色

    # (可选) 创建一个LineSet来可视化原始骨架，用于对比
    # 我们需要合并点并创建边的索引
    skeleton_points = np.vstack([trunk_points, branch1_points[1:], branch2_points[1:]])  # [1:]避免重复分叉点
    skeleton_lines = []
    # 主干的边
    for i in range(len(trunk_points) - 1):
        skeleton_lines.append([i, i + 1])
    # 左分支的边
    bifurcation_idx = len(trunk_points) - 1
    branch1_start_idx = len(trunk_points)
    skeleton_lines.append([bifurcation_idx, branch1_start_idx])
    for i in range(len(branch1_points) - 2):
        skeleton_lines.append([branch1_start_idx + i, branch1_start_idx + i + 1])
    # 右分支的边
    branch2_start_idx = len(trunk_points) + len(branch1_points) - 1
    skeleton_lines.append([bifurcation_idx, branch2_start_idx])
    for i in range(len(branch2_points) - 2):
        skeleton_lines.append([branch2_start_idx + i, branch2_start_idx + i + 1])

    skeleton_vis = o3d.geometry.LineSet()
    skeleton_vis.points = o3d.utility.Vector3dVector(skeleton_points)
    skeleton_vis.lines = o3d.utility.Vector2iVector(skeleton_lines)
    skeleton_vis.paint_uniform_color([1, 0, 0])  # 红色骨架

    print("显示重建结果。红色为原始骨架，棕色为重建模型。")
    o3d.visualization.draw_geometries(
        [tree_mesh, skeleton_vis],
        window_name="树状骨架重建示例"
    )