"""
形态学特征表型树构建模块 (支持加权 Gower 距离与 BioPython 导出)
"""
import pandas as pd
import numpy as np
import os
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio import Phylo

def calculate_gower_distance_matrix(df, cat_features, num_features, weights=None):
    """
    计算基于加权 Gower 距离的对称距离矩阵
    
    Args:
        df: pandas DataFrame (必须包含特征列，且不要把物种名设在列中，应设为Index或稍后提取)
        cat_features: 定类/二元特征列名列表
        num_features: 定量/连续数值特征列名列表
        weights: 权重字典 {特征名: 权重数值}
    
    Returns:
        distance_matrix: numpy二维数组 (n_samples, n_samples)
    """
    if weights is None:
        weights = {}
        
    n_samples = len(df)
    gower_dist = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            sum_s = 0.0
            sum_weight = 0.0
            
            # 定类特征计算
            for col in cat_features:
                val_i, val_j = df.iloc[i][col], df.iloc[j][col]
                if pd.notna(val_i) and pd.notna(val_j):
                    w = weights.get(col, 1.0)
                    sum_weight += w
                    if val_i == val_j:
                        sum_s += w * 1.0 
                        
            # 定量数值计算
            for col in num_features:
                val_i, val_j = df.iloc[i][col], df.iloc[j][col]
                if pd.notna(val_i) and pd.notna(val_j):
                    w = weights.get(col, 1.0)
                    sum_weight += w
                    R_k = df[col].max() - df[col].min()
                    if R_k == 0:
                        sum_s += w * 1.0 
                    else:
                        sum_s += w * (1.0 - (abs(val_i - val_j) / R_k))
                        
            similarity = sum_s / sum_weight if sum_weight > 0 else 0
            distance = 1.0 - similarity
            gower_dist[i, j] = distance
            gower_dist[j, i] = distance

    return gower_dist

def build_tree_with_biopython(distance_matrix, species_names, method='upgma'):
    """
    使用 BioPython 接收距离矩阵并构建进化树
    """
    # 清洗物种名：BioPython 和 Newick 格式非常忌讳空格、括号和冒号
    clean_names = [str(name).replace(" ", "_").replace("(", "").replace(")", "").replace(":", "") 
                   for name in species_names]
    
    n = len(clean_names)
    
    # 将完整的 numpy 矩阵转换为 BioPython DistanceMatrix 要求的下三角列表格式
    dist_list = []
    for i in range(n):
        row = []
        for j in range(i + 1):
            row.append(float(distance_matrix[i, j]))
        dist_list.append(row)
    
    dm = DistanceMatrix(names=clean_names, matrix=dist_list)
    constructor = DistanceTreeConstructor()
    
    if method.lower() == 'upgma':
        tree = constructor.upgma(dm)
    elif method.lower() == 'nj':
        tree = constructor.nj(dm)
    else:
        raise ValueError(f"不支持的方法: {method}，请选择 'upgma' 或 'nj'")
        
    return tree

def generate_phenotypic_trees(csv_path, cat_cols, num_cols, custom_weights=None, output_dir="tree_outputs"):
    """
    主控制流程：读取数据 -> 清洗 -> 算距离 -> 建树 -> 导出 nwk 和 nex
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取与清洗数据
    df = pd.read_csv(csv_path, encoding='gbk')
    # 处理可能的非数值错误
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
        
    species_list = df['Species'].tolist()
    
    # 2. 计算 Gower 距离矩阵
    print("正在计算加权 Gower 距离矩阵...")
    dist_matrix = calculate_gower_distance_matrix(df, cat_cols, num_cols, weights=custom_weights)
    
    # 3. 构建并导出 UPGMA 树
    print("构建 UPGMA 树...")
    upgma_tree = build_tree_with_biopython(dist_matrix, species_list, method='upgma')
    Phylo.write(upgma_tree, os.path.join(output_dir, "UPGMA_weighted.nwk"), "newick")
    Phylo.write(upgma_tree, os.path.join(output_dir, "UPGMA_weighted.nex"), "nexus")
    
    # 4. 构建并导出 NJ (Neighbor-Joining) 树
    print("构建 NJ 树...")
    nj_tree = build_tree_with_biopython(dist_matrix, species_list, method='nj')
    Phylo.write(nj_tree, os.path.join(output_dir, "NJ_weighted.nwk"), "newick")
    Phylo.write(nj_tree, os.path.join(output_dir, "NJ_weighted.nex"), "nexus")
    
    print(f"✅ 所有树文件已成功保存至目录: {output_dir}")


# ==========================================
# 如何调用执行
# ==========================================
if __name__ == "__main__":
    
    # 您的分类学特征列名
    categorical_features = ['Hair_structures', 'Leaf_Phenology']
    numerical_features = ['Min_pedicel_length', 'Max_pedicel_length']
    
    # 设置您想要的权重
    weights_dict = {
        'Hair_structures(stellate=0,mixed=1,peltate=2)': 1.0,
        'Leaf_Phenology(Evergreen=0,Semi-evergreen=2,Deciduous=2)': 1.0,
        'Min_pedicel_length': 3.0,
        'Max_pedicel_length': 3.0
    }
    
    # 执行主函数
    generate_phenotypic_trees(
        csv_path='features.csv',
        cat_cols=categorical_features,
        num_cols=numerical_features,
        custom_weights=weights_dict,
        output_dir='./outputs/trees/baseline'
    )