"""系统发育树构建模块。"""

from __future__ import annotations

import copy
from io import StringIO
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio.Phylo import Consensus
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree
from scipy.spatial.distance import pdist, squareform


def _ensure_string_list(species_names: Sequence[object]) -> List[str]:
    """确保物种名是 Python 字符串列表。"""
    if isinstance(species_names, np.ndarray):
        species_names = species_names.tolist()
    elif not isinstance(species_names, list):
        try:
            species_names = list(species_names)
        except TypeError as exc:
            raise TypeError(
                f"无法将 species_names 转换为列表: {type(species_names)}"
            ) from exc
    return [str(name) for name in species_names]


def _extract_genus(species_name: str) -> str:
    """从 ``Genus_species`` 或 ``Genus species`` 中提取属名。"""
    return str(species_name).replace("_", " ").split()[0]


def _validate_distance_matrix(
    distance_matrix: np.ndarray, species_names: Sequence[str]
) -> np.ndarray:
    """检查并标准化距离矩阵。"""
    names = _ensure_string_list(species_names)
    matrix = np.asarray(distance_matrix, dtype=float)
    n = len(names)

    if n < 2:
        raise ValueError(f"构建系统发育树至少需要2个物种，当前只有 {n} 个")
    if matrix.shape != (n, n):
        raise ValueError(
            f"距离矩阵维度 ({matrix.shape}) 与物种数量 ({n}) 不匹配"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("距离矩阵不能包含 NaN 或无穷大")
    if np.any(matrix < 0):
        raise ValueError("距离矩阵不能包含负距离")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("距离矩阵必须是对称矩阵")

    # 避免输入中的极小浮点误差影响 squareform；不改变实际距离。
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def calculate_distance_matrix(features, metric="euclidean"):
    """计算距离矩阵。

    Args:
        features: 特征矩阵 (n_samples, n_features)
        metric: ``euclidean``、``cosine`` 或 ``manhattan``
    """
    metric_map = {
        "euclidean": "euclidean",
        "cosine": "cosine",
        "manhattan": "cityblock",
    }
    if metric not in metric_map:
        raise ValueError(f"不支持的距离度量: {metric}")
    return squareform(pdist(features, metric=metric_map[metric]))


def _average_genus_distance(
    distance_matrix: np.ndarray, genus_indices: Dict[str, List[int]]
) -> np.ndarray:
    """计算属级距离：两个属之间所有物种对的距离平均值。"""
    genera = list(genus_indices)
    n_genera = len(genera)
    genus_matrix = np.zeros((n_genera, n_genera), dtype=float)

    for i in range(n_genera):
        for j in range(i):
            cross_distances = distance_matrix[np.ix_(genus_indices[genera[i]], genus_indices[genera[j]])]
            genus_matrix[i, j] = genus_matrix[j, i] = float(np.mean(cross_distances))
    return genus_matrix


def _cluster_node_to_newick(
    node: ClusterNode,
    names: Sequence[str],
    parent_height: float,
) -> str:
    """将 SciPy ClusterNode 转成带 branch length 的 Newick 片段。

    SciPy linkage 的 ``node.dist`` 是聚类高度对应的距离，
    UPGMA/average-linkage 的树高为 ``node.dist / 2``。
    """
    node_height = 0.0 if node.is_leaf() else max(0.0, float(node.dist) / 2.0)

    if node.is_leaf():
        label = _newick_label(names[node.id])
        branch = max(0.0, float(parent_height) - node_height)
        return f"{label}:{branch:.6f}"

    left = _cluster_node_to_newick(node.left, names, node_height)
    right = _cluster_node_to_newick(node.right, names, node_height)
    branch = max(0.0, float(parent_height) - node_height)
    return f"({left},{right}):{branch:.6f}"


def _newick_label(name: str) -> str:
    """为含空格或 Newick 特殊字符的叶名添加引号。"""
    if any(char in name for char in "(),:;[]\t "):
        return "'" + name.replace("'", "''") + "'"
    return name


def tree_to_newick(node: ClusterNode, species_names, parent_dist=0.0):
    """将 SciPy 聚类树转换为 Newick 字符串片段。

    ``parent_dist`` 保留旧接口名称；其值按树高（branch-length units）处理。
    通常直接传入根节点时使用默认值即可。
    """
    names = _ensure_string_list(species_names)
    return _cluster_node_to_newick(node, names, float(parent_dist))


def _build_flat_average_newick(
    distance_matrix: np.ndarray, species_names: Sequence[str]
) -> Tuple[str, np.ndarray]:
    """对全部物种直接执行 SciPy average linkage。"""
    names = _ensure_string_list(species_names)
    matrix = _validate_distance_matrix(distance_matrix, names)
    condensed_dist = squareform(matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method="average")
    root = to_tree(linkage_matrix, rd=False)
    return _cluster_node_to_newick(root, names, 0.0) + ";", linkage_matrix


def build_hierarchical_average_tree(
    distance_matrix: np.ndarray, species_names: Sequence[str]
) -> Tuple[str, Optional[np.ndarray], Dict[str, Optional[np.ndarray]], np.ndarray]:
    """先属内、再属间建 average-linkage 树，并拼接为最终 Newick 树。

    Returns:
        final_newick: 拼接后的最终树
        genus_linkage: 属间 linkage 矩阵；只有一个属时为 ``None``
        within_linkages: 每个属对应的属内 linkage 矩阵
        genus_distance_matrix: 属级距离矩阵
    """
    names = _ensure_string_list(species_names)
    matrix = _validate_distance_matrix(distance_matrix, names)

    genus_indices: Dict[str, List[int]] = {}
    for index, name in enumerate(names):
        genus_indices.setdefault(_extract_genus(name), []).append(index)
    genera = list(genus_indices)

    within_roots: Dict[str, Optional[ClusterNode]] = {}
    within_linkages: Dict[str, Optional[np.ndarray]] = {}
    within_names: Dict[str, List[str]] = {}

    # 第一层：每个属内部建树。
    for genus, indices in genus_indices.items():
        local_names = [names[index] for index in indices]
        within_names[genus] = local_names
        if len(indices) == 1:
            within_roots[genus] = None
            within_linkages[genus] = None
            continue

        local_matrix = matrix[np.ix_(indices, indices)]
        local_linkage = linkage(
            squareform(local_matrix, checks=False), method="average"
        )
        within_linkages[genus] = local_linkage
        within_roots[genus] = to_tree(local_linkage, rd=False)

    # 第二层：用属间平均物种距离构建属级树。
    genus_matrix = _average_genus_distance(matrix, genus_indices)
    if len(genera) == 1:
        genus_linkage = None
        genus_root = None
    else:
        genus_linkage = linkage(
            squareform(genus_matrix, checks=False), method="average"
        )
        genus_root = to_tree(genus_linkage, rd=False)

    def genus_subtree(genus: str, parent_height: float) -> str:
        local_root = within_roots[genus]
        local_names = within_names[genus]
        if local_root is None:
            return f"{_newick_label(local_names[0])}:{max(0.0, parent_height):.6f}"
        return _cluster_node_to_newick(local_root, local_names, parent_height)

    if genus_root is None:
        # 只有一个属时，属内树就是最终树；根到自身的 branch length 为 0。
        only_genus = genera[0]
        if within_roots[only_genus] is None:
            final_newick = f"{_newick_label(names[0])}:0.000000;"
        else:
            final_newick = _cluster_node_to_newick(
                within_roots[only_genus], within_names[only_genus], 0.0
            ) + ";"
        return final_newick, genus_linkage, within_linkages, genus_matrix

    def inter_node_to_newick(node: ClusterNode, parent_height: float) -> str:
        node_height = 0.0 if node.is_leaf() else max(0.0, float(node.dist) / 2.0)
        if node.is_leaf():
            genus = genera[node.id]
            return genus_subtree(genus, parent_height)

        left = inter_node_to_newick(node.left, node_height)
        right = inter_node_to_newick(node.right, node_height)
        branch = max(0.0, float(parent_height) - node_height)
        return f"({left},{right}):{branch:.6f}"

    final_newick = inter_node_to_newick(genus_root, 0.0) + ";"
    return final_newick, genus_linkage, within_linkages, genus_matrix


def build_upgma_tree(distance_matrix, species_names):
    """构建两层属约束的 average-linkage 树并返回 BioPython Tree 对象。

    函数名保留是为了兼容旧代码；这里不再调用 BioPython UPGMA。
    """
    newick, _, _, _ = build_hierarchical_average_tree(distance_matrix, species_names)
    return Phylo.read(StringIO(newick), "newick")


def build_upgma_scipy(distance_matrix, species_names):
    """兼容旧接口：返回两层 average-linkage 树的 Newick 和属间 linkage。"""
    newick, genus_linkage, _, _ = build_hierarchical_average_tree(
        distance_matrix, species_names
    )
    return newick, genus_linkage


def build_nj_tree(distance_matrix, species_names):
    """使用 BioPython Neighbor-Joining 方法构建树。"""
    names = _ensure_string_list(species_names)
    matrix = _validate_distance_matrix(distance_matrix, names)

    return _build_nj_tree_from_validated_matrix(matrix, names)


def _build_nj_tree_from_validated_matrix(
    matrix: np.ndarray, names: Sequence[str]
):
    """对已验证的距离矩阵执行一次 BioPython NJ。"""
    dist_list = [
        [float(matrix[i, j]) for j in range(i + 1)] for i in range(len(names))
    ]
    dm = DistanceMatrix(names=names, matrix=dist_list)
    return DistanceTreeConstructor().nj(dm)


def build_hierarchical_nj_tree(
    distance_matrix: np.ndarray, species_names: Sequence[str]
):
    """先属内、再属间执行 NJ，并将属内 NJ 子树拼接回属间 NJ 树。

    NJ 本身不接受 monophyly constraint 参数；这里通过把每个属先压缩成
    一个属级叶节点，再用属内树替换该叶节点，保证每个属在最终树中单系。
    属间距离使用两个属之间所有物种距离的平均值。
    """
    names = _ensure_string_list(species_names)
    matrix = _validate_distance_matrix(distance_matrix, names)

    genus_indices: Dict[str, List[int]] = {}
    for index, name in enumerate(names):
        genus_indices.setdefault(_extract_genus(name), []).append(index)
    genera = list(genus_indices)

    within_trees = {}
    for genus, indices in genus_indices.items():
        local_names = [names[index] for index in indices]
        if len(indices) == 1:
            within_trees[genus] = Tree(root=Clade(name=local_names[0]))
        else:
            local_matrix = matrix[np.ix_(indices, indices)]
            within_trees[genus] = _build_nj_tree_from_validated_matrix(
                local_matrix, local_names
            )

    genus_matrix = _average_genus_distance(matrix, genus_indices)
    if len(genera) == 1:
        # 只有一个属时，不需要额外的属间节点。
        return within_trees[genera[0]]

    genus_tree = _build_nj_tree_from_validated_matrix(genus_matrix, genera)

    def replace_genus_leaf(clade: Clade) -> Clade:
        if clade.is_terminal():
            genus = clade.name
            if genus not in within_trees:
                raise ValueError(f"属间树中出现未识别的属节点: {genus}")

            # 保留属间 NJ 给该属叶节点分配的 branch length，
            # 并将属内树根替换到同一位置。
            replacement = copy.deepcopy(within_trees[genus].root)
            replacement.branch_length = clade.branch_length
            return replacement

        clade.clades = [replace_genus_leaf(child) for child in clade.clades]
        return clade

    genus_tree.root = replace_genus_leaf(genus_tree.root)
    return genus_tree


def visualize_tree(tree, species_names, method="upgma", save_path=None):
    """可视化系统发育树。"""
    names = _ensure_string_list(species_names)
    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))

    if isinstance(tree, str):
        tree_obj = Phylo.read(StringIO(tree), "newick")
    else:
        tree_obj = tree

    Phylo.draw(tree_obj, axes=ax, do_show=False)
    method_key = str(method).lower().replace("-", "_")
    method_name = (
        "Average Linkage"
        if method_key in {"upgma", "average", "average_linkage"}
        else "Neighbor-Joining"
    )
    ax.set_title(
        f"Phylogenetic Tree ({method_name})", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", format="pdf")
        print(f"系统发育树已保存到: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def save_tree_newick(tree, species_names, filepath):
    """保存树为 Newick 格式。"""
    if isinstance(tree, str):
        newick_str = tree if tree.rstrip().endswith(";") else tree.rstrip() + ";"
    else:
        handle = StringIO()
        Phylo.write(tree, handle, "newick")
        newick_str = handle.getvalue()

    with open(filepath, "w", encoding="utf-8") as file_handle:
        file_handle.write(newick_str)
    print(f"Newick格式树已保存到: {filepath}")


def save_tree_nexus(tree, species_names, filepath):
    """保存树为 NEXUS 格式。"""
    tree_obj = (
        Phylo.read(StringIO(tree), "newick") if isinstance(tree, str) else tree
    )
    Phylo.write(tree_obj, filepath, "nexus")
    print(f"NEXUS格式树已安全保存到: {filepath}")


def save_distance_matrix_excel(distance_matrix, species_names, filepath):
    """保存距离矩阵为 Excel 文件。"""
    import pandas as pd

    names = _ensure_string_list(species_names)
    pd.DataFrame(distance_matrix, index=names, columns=names).to_excel(filepath)
    print(f"距离矩阵Excel文件已保存到: {filepath}")


def save_matrix_phylip(distance_matrix, species_names, filepath):
    """将距离矩阵保存为 FastME 支持的 PHYLIP 格式。"""
    names = _ensure_string_list(species_names)
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.shape != (len(names), len(names)):
        raise ValueError("距离矩阵维度与物种名数量不匹配")

    with open(filepath, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"    {len(names)}\n")
        for index, name in enumerate(names):
            distances = " ".join(f"{value:.6f}" for value in matrix[index])
            file_handle.write(f"{name} {distances}\n")
    print(f"PHYLIP 格式距离矩阵已保存至: {filepath}")


def bootstrap_consensus_tree(
    features, labels, species_names, method="upgma", n_bootstraps=100
):
    """对物种内样本进行 bootstrap，并生成 majority-rule 共识树。"""
    names = _ensure_string_list(species_names)
    labels = np.asarray(labels)
    features = np.asarray(features)
    unique_labels = np.unique(labels)
    if len(unique_labels) != len(names):
        raise ValueError("species_names 数量必须与 labels 中的物种数量一致")

    print(f"开始Bootstrap分析 ({n_bootstraps}次)...")
    trees = []
    method_key = str(method).lower().replace("-", "_")

    for _ in range(n_bootstraps):
        boot_species_features = []
        for label in unique_labels:
            species_features = features[labels == label]
            if len(species_features) == 0:
                raise ValueError(f"物种标签 {label} 没有对应特征")
            indices = np.random.choice(len(species_features), len(species_features), replace=True)
            boot_species_features.append(
                np.mean(species_features[indices], axis=0)
            )

        boot_distance = calculate_distance_matrix(np.asarray(boot_species_features))
        if method_key in {"upgma", "average", "average_linkage"}:
            tree = build_upgma_tree(boot_distance, names)
        elif method_key == "nj":
            tree = build_hierarchical_nj_tree(boot_distance, names)
        else:
            raise ValueError(f"不支持的建树方法: {method}")
        trees.append(tree)

    return Consensus.majority_consensus(trees, 0.5)


def build_phylogenetic_trees(
    features,
    species_names,
    methods=("upgma", "nj"),
    distance_metric="euclidean",
    output_dir="outputs/trees",
    raw_features=None,
    raw_labels=None,
    precomputed_distance_matrix=None,
):
    """构建、保存并返回 average-linkage/NJ 系统发育树。"""
    names = _ensure_string_list(species_names)
    if precomputed_distance_matrix is not None:
        distance_matrix = _validate_distance_matrix(
            precomputed_distance_matrix, names
        )
        print("  -> 使用外部传入的距离矩阵 ...")
    else:
        distance_matrix = calculate_distance_matrix(features, metric=distance_metric)
        print(f"计算原始距离矩阵 (度量: {distance_metric})...")

    os.makedirs(output_dir, exist_ok=True)
    save_distance_matrix_excel(
        distance_matrix,
        names,
        os.path.join(output_dir, "distance_matrix_original.xlsx"),
    )
    save_matrix_phylip(
        distance_matrix,
        names,
        os.path.join(output_dir, "distance_matrix_fastme.phy"),
    )

    # 属级矩阵仅用于 average-linkage 的第二层，不再生成惩罚距离矩阵。
    genus_names = []
    genus_indices: Dict[str, List[int]] = {}
    for index, name in enumerate(names):
        genus = _extract_genus(name)
        if genus not in genus_indices:
            genus_names.append(genus)
            genus_indices[genus] = []
        genus_indices[genus].append(index)
    genus_matrix = _average_genus_distance(distance_matrix, genus_indices)
    save_distance_matrix_excel(
        genus_matrix,
        genus_names,
        os.path.join(output_dir, "distance_matrix_genus.xlsx"),
    )

    trees = {}
    average_methods = {"upgma", "average", "average_linkage"}
    for method in methods:
        method_key = str(method).lower().replace("-", "_")
        print(f"\n使用 {str(method).upper()} 方法构建系统发育树...")
        try:
            if method_key in average_methods:
                tree = build_upgma_tree(distance_matrix, names)
            elif method_key == "nj":
                tree = build_hierarchical_nj_tree(distance_matrix, names)
            else:
                print(f"警告: 不支持的方法 {method}，跳过")
                continue

            trees[method] = tree
            visualize_tree(
                tree,
                names,
                method,
                os.path.join(output_dir, f"tree_{method}.pdf"),
            )
            save_tree_newick(
                tree, names, os.path.join(output_dir, f"tree_{method}.newick")
            )
            save_tree_nexus(
                tree, names, os.path.join(output_dir, f"tree_{method}.nexus")
            )

            if raw_features is not None and raw_labels is not None:
                print(f"正在进行 {str(method).upper()} Bootstrap验证...")
                consensus_tree = bootstrap_consensus_tree(
                    raw_features, raw_labels, names, method=method
                )
                save_tree_newick(
                    consensus_tree,
                    names,
                    os.path.join(output_dir, f"tree_{method}_consensus.newick"),
                )
                visualize_tree(
                    consensus_tree,
                    names,
                    f"{method} Consensus",
                    os.path.join(output_dir, f"tree_{method}_consensus.pdf"),
                )
            print(f"✓ {str(method).upper()} 树构建完成")
        except Exception as exc:
            print(f"错误: 构建 {method} 树时出错: {exc}")
            import traceback

            traceback.print_exc()

    return trees, distance_matrix
