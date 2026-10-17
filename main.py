"""
主程序：整合所有模块
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from utils import (set_seed, visualize_features, plot_distance_matrix, save_features,
                   plot_distance_distribution, plot_feature_correlation, 
                   plot_species_feature_comparison, plot_clustering_dendrogram,
                   plot_feature_statistics, detect_outliers, numpy_to_nexus_file)
from data_loader import load_dataset, create_dataloaders
from models import create_model
from trainer import train_model
from feature_extractor import extract_all_features, FeatureExtractor
from phylogeny import build_phylogenetic_trees, calculate_distance_matrix


def save_matrix_phylip(distance_matrix, species_names, filepath):
    """将距离矩阵保存为 FastME 支持的 PHYLIP 格式"""
    n = len(species_names)
    with open(filepath, 'w') as f:
        f.write(f"    {n}\n")
        for i in range(n):
            distances_str = " ".join([f"{d:.6f}" for d in distance_matrix[i]])
            f.write(f"{species_names[i]} {distances_str}\n")
    print(f"PHYLIP 格式距离矩阵已成功导出至: {filepath}")


def gower_distance_matrix(df, categorical_threshold=10):
    """
    真正的 Gower 距离，正确支持 NaN。
    - 分类性状：相同=0，不同=1，每个性状等权（不管类别数多少）
    - 连续性状：|x-y| / 该性状极差，每个性状等权
    - 缺失值：该性状权重为0，自动跳过
    - 最终距离 = sum(权重 × 距离) / sum(权重)
    - 两物种无任何共同非缺失性状时返回 NaN（不填0）
    - 类型判断：列名含 _size_/_length/_width/_height/_diameter 等关键词
      判定为连续测量性状，其余判定为分类性状
    """
    n = len(df)
    cols = list(df.columns)
    values = df.values.astype(float)

    # 连续测量性状的列名关键词（小写匹配）
    continuous_keywords = ['_size_', '_length', '_width', '_height',
                           '_diameter', '_thickness', '_area', '_volume']

    col_types, col_ranges = [], []
    for ci in range(len(cols)):
        col_name = cols[ci].lower()
        non_null = values[~np.isnan(values[:, ci]), ci]
        if len(non_null) == 0:
            col_types.append('continuous')
            col_ranges.append(1.0)
            continue

        # 优先按列名关键词判断：含尺寸关键词 → 连续
        is_continuous = any(kw in col_name for kw in continuous_keywords)

        if is_continuous:
            col_types.append('continuous')
            rng = non_null.max() - non_null.min()
            col_ranges.append(rng if rng > 0 else 1.0)
        else:
            # 不含尺寸关键词 → 分类性状（spine/habit/Duration/color/shape 等）
            col_types.append('categorical')
            col_ranges.append(None)

    # 打印类型判断结果，方便检查是否有误判
    print("  性状类型判断:")
    for ci, col in enumerate(cols):
        print(f"    {col:30s} -> {col_types[ci]}")

    # 计算距离矩阵
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            total_weight = 0.0
            total_dist = 0.0
            for k in range(len(cols)):
                vi, vj = values[i, k], values[j, k]
                if np.isnan(vi) or np.isnan(vj):
                    continue  # 缺失→权重0，跳过
                total_weight += 1.0  # 每个性状等权
                if col_types[k] == 'categorical':
                    if vi != vj:
                        total_dist += 1.0
                else:
                    total_dist += abs(vi - vj) / col_ranges[k]
            dist[i, j] = dist[j, i] = (
                total_dist / total_weight if total_weight > 0 else np.nan
            )
    return dist

def count_valid_characters(df):
    """
    计算每对物种的共同有效（非缺失）性状数矩阵。
    用于检查 Gower 距离中每对物种实际使用了多少个性状，
    避免某些物种对因共同性状过少而距离不可靠。
    """
    n = len(df)
    values = df.values.astype(float)
    count = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            valid = int(np.sum(~np.isnan(values[i]) & ~np.isnan(values[j])))
            count[i, j] = count[j, i] = valid
    return count

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='植物系统发育树构建')
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['train', 'extract', 'phylogeny', 'full'],
                        help='运行模式: train(仅训练), extract(仅提取特征), phylogeny(仅构建树), full(完整流程)')
    parser.add_argument('--image_type', type=str, default='specimen',
                        choices=['specimen', 'habitat'],
                        help='图像类型: specimen(标本) 或 habitat(生境)')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        choices=['resnet50', 'vit_b16', 'inception_resnet_v2'],
                        help='模型类型')
    parser.add_argument('--skip_training', action='store_true',
                        help='跳过训练，直接使用预训练模型')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    config.MODEL_TYPE = args.model_type
    # 每种图像类型单独保存，避免 specimen / habitat 互相覆盖
    config.OUTPUT_DIR = os.path.join("outputs", args.image_type)
    config.create_output_dirs()
    
    # 设置随机种子
    set_seed(config.SEED)
    
    # 确定数据路径
    if args.image_type == 'specimen':
        data_path = config.SPECIMEN_PATH
        if not os.path.exists(data_path):
            data_path = "数据集/标本图像/results_plantsam"
    else:
        data_path = config.HABITAT_PATH
    
    print("=" * 60)
    print("植物系统发育树构建系统")
    print("=" * 60)
    print(f"图像类型: {args.image_type}")
    print(f"数据路径: {data_path}")
    print(f"模型类型: {config.MODEL_TYPE}")
    print(f"特征维度: {config.FEATURE_DIM}")
    print(f"设备: {config.DEVICE}")
    print("=" * 60)
    
    # 加载数据集
    print("\n[1/5] 加载数据集...")
    image_paths, labels, species_names = load_dataset(data_path, args.image_type)
    
    if len(image_paths) == 0:
        print("错误: 没有找到图像数据！")
        return
    
    # 创建数据加载器
    print("\n[2/5] 创建数据加载器...")
    train_loader, val_loader, test_loader, all_eval_loader = create_dataloaders(
        image_paths, labels,
        batch_size=config.BATCH_SIZE,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        use_triplet=True,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS,
        model_type=config.MODEL_TYPE
    )
    
    # 训练模型
    model_path = os.path.join(config.OUTPUT_DIR, "models", f"{config.MODEL_TYPE}_best.pth")
    
    if args.mode in ['train', 'full'] and not args.skip_training:
        print("\n[3/5] 训练模型...")
        model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(config, train_loader, val_loader)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        loss_curve_path = os.path.join(config.OUTPUT_DIR, "figures", "training_loss_curve.pdf")
        plt.savefig(loss_curve_path, bbox_inches='tight', format='pdf')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(train_accuracies, label='Training Accuracy', linewidth=2)
        plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Training Accuracy Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        acc_curve_path = os.path.join(config.OUTPUT_DIR, "figures", "training_accuracy_curve.pdf")
        plt.savefig(acc_curve_path, bbox_inches='tight', format='pdf')
        plt.close()
    else:
        print("\n[3/5] 跳过训练")
    
    # 提取特征或加载特征
    if args.mode in ['extract', 'full']:
        print("\n[4/5] 提取特征...")
        if os.path.exists(model_path):
            print(f"发现本地权重文件，准备加载: {model_path}")
            features, labels, image_paths = extract_all_features(config, all_eval_loader, model_path)
        else:
            print("未发现本地权重文件，使用默认预训练模型。")
            features, labels, image_paths = extract_all_features(config, test_loader, None)
            
    elif args.mode == 'phylogeny':
        print("\n[4/5] 模式为 phylogeny：正在从本地加载已提取的特征文件...")
        from utils import load_features
        feature_path = os.path.join(
            config.OUTPUT_DIR, 
            "features", 
            f"{args.image_type}_{config.MODEL_TYPE}_features.npz"
        )
        if not os.path.exists(feature_path):
            print(f"错误: 找不到特征文件 {feature_path}！请先运行 --mode extract")
            return
            
        features, labels, loaded_names = load_features(feature_path)
        
        if loaded_names is not None:
            species_names = loaded_names
            
        image_paths = [] 
        print(f"成功加载特征：共有 {len(features)} 个样本")
    if args.mode in ['extract', 'full', 'phylogeny']:
        # 计算物种级特征
        from utils import calculate_species_features
        species_features, species_names_clean = calculate_species_features(
            features, labels, all_species_names=species_names, aggregation='mean'
        )
        
        print(f"提取了 {len(species_features)} 个物种的 DL 特征")
        
        # ==================== 全自动性状融合版（无需手动配置性状） ====================
        import pandas as pd
        from scipy.spatial.distance import pdist, squareform
        print("\n[Trait Fusion] 正在自动读取性状表并执行全性状等权融合...")
        trait_path = "features.csv"  # 你的性状表路径，第一列必须是物种名
        if os.path.exists(trait_path):
            # 1. 自动读取性状表，兼容编码
            if trait_path.endswith('.xlsx') or trait_path.endswith('.xls'):
                trait_df = pd.read_excel(trait_path)
            else:
                try:
                    trait_df = pd.read_csv(trait_path, encoding='utf-8')
                except UnicodeDecodeError:
                    trait_df = pd.read_csv(trait_path, encoding='gbk')
            
            # 2. 自动设置第一列为物种名索引，剩余所有列自动作为性状
            species_col = trait_df.columns[0]
            trait_df.set_index(species_col, inplace=True)
            trait_df.index.name = "Species"

            # === 性状数据清理：空白/异常字符串转 NaN，强制所有列转数字 ===
            trait_df = trait_df.replace(['', ' ', 'NA', 'na', '-', '?', 'nan'], np.nan)
            for col in trait_df.columns:
                trait_df[col] = pd.to_numeric(trait_df[col], errors='coerce')
            
            trait_count = trait_df.shape[1]
            print(f"✓ 自动识别到 {trait_count} 个形态性状，涉及 {trait_df.shape[0]} 个物种")
            print(f"  性状列表: {list(trait_df.columns)}")
            # 3. 整体融合权重（仅需调这两个数，默认 DL:性状 = 1:1 等权）
            WEIGHT_DL = 0.5    
            WEIGHT_TRAIT = 0.5 
            trait_suffix = "all_traits"  # 输出文件后缀
            # 4. 深度学习特征与性状表按物种名自动对齐
            dl_df = pd.DataFrame(species_features, index=species_names_clean)
            merged_df = dl_df.join(trait_df, how='inner')
            species_names_clean = merged_df.index.tolist()
            print(f"✓ 物种名对齐后剩余 {len(species_names_clean)} 个共有物种")
            
            # 5. 拆分 DL 特征矩阵 和 形态性状矩阵
            dl_matrix = merged_df.iloc[:, :species_features.shape[1]].values
            trait_matrix = merged_df.iloc[:, species_features.shape[1]:]
            # ==================== 🛠️ 针对形态空值（NaN）的严谨修复版 ====================
            # 6. 生成归一化融合特征表（原样保留 NaN，不使用易崩溃的 MinMaxScaler）
            #    注意：trait_norm_df 仅用于输出 CSV，不参与距离计算
            trait_min = trait_matrix.min()
            trait_max = trait_matrix.max()
            trait_range = (trait_max - trait_min).replace(0, 1)  # 防止分母为 0
            
            # 使用 Pandas 自带算子进行最大极差缩放，NaN 会被原样保留
            trait_norm_df = (trait_matrix - trait_min) / trait_range
            
            # DL特征命名
            dl_feature_cols = [f"DL_Feature_{i}" for i in range(dl_matrix.shape[1])]
            dl_norm_df = pd.DataFrame(dl_matrix, columns=dl_feature_cols, index=species_names_clean)
            
            # 横向拼接成完整融合特征表（此时形态列中包含合法的 NaN）
            fused_features_df = pd.concat([dl_norm_df, trait_norm_df], axis=1)
            fused_features_df.index.name = 'Species'
            # 保存融合特征表
            features_out_dir = os.path.join(config.OUTPUT_DIR, "features")
            os.makedirs(features_out_dir, exist_ok=True)
            fused_features_path = os.path.join(features_out_dir, f"fused_features_{trait_suffix}.csv")
            fused_features_df.to_csv(fused_features_path)
            print(f"✓ 融合特征表已成功保存（保留原始空值）至: {fused_features_path}")
            # 7. 分别计算距离矩阵
            print("正在计算深度学习特征欧氏距离矩阵...")
            dist_dl = squareform(pdist(dl_matrix, metric='euclidean'))
            dist_dl_norm = dist_dl / np.max(dist_dl)  # DL 归一化到 0~1
            print("正在计算形态性状 Gower 距离矩阵（真正Gower，支持NaN）...")
            dist_trait = gower_distance_matrix(trait_matrix)

            # 距离检查
            print(f"  DL距离范围: [{np.min(dist_dl_norm[dist_dl_norm>0]):.4f}, {np.max(dist_dl_norm):.4f}]")
            print(f"  Trait距离范围: [{np.nanmin(dist_trait[dist_trait>0]):.4f}, {np.nanmax(dist_trait):.4f}]")
            print(f"  Trait距离唯一值数量: {len(np.unique(np.round(dist_trait[np.isfinite(dist_trait)], 3)))}")
            if np.isnan(dist_trait).any():
                nan_pairs = np.argwhere(np.triu(np.isnan(dist_trait), k=1))
                raise ValueError(f"形态距离存在 {len(nan_pairs)} 对 NaN（无共同性状），请检查数据。")

            # 形态距离也归一化到 0~1，保证权重可比
            dist_trait_norm = dist_trait / np.nanmax(dist_trait)
            # 计算并保存有效性状计数矩阵（检查每对物种用了多少共同性状）
            valid_count = count_valid_characters(trait_matrix)
            valid_count_path = os.path.join(
                config.OUTPUT_DIR, "features", "trait_valid_character_count.csv"
            )
            pd.DataFrame(valid_count, index=species_names_clean,
                         columns=species_names_clean).to_csv(valid_count_path)
            triu_vc = valid_count[np.triu_indices_from(valid_count, k=1)]
            print(f"  共同性状数: 最小={triu_vc.min()}, 最大={triu_vc.max()}, "
                  f"均值={triu_vc.mean():.1f}, 中位数={np.median(triu_vc):.0f}")
            print(f"  共同性状<10的物种对: {(triu_vc < 10).sum()}/{len(triu_vc)}")
            print(f"  有效性状计数矩阵已保存: {valid_count_path}")
            # 8. 加权融合最终距离矩阵（两者都归一化后，0.5:0.5 真正等权）
            matrix_fused = (WEIGHT_DL * dist_dl_norm) + (WEIGHT_TRAIT * dist_trait_norm)
            if np.isnan(matrix_fused).any():
                raise ValueError("融合距离存在 NaN，请检查距离矩阵。")
            print(f"✓ 距离矩阵融合完成 | DL权重:{WEIGHT_DL} | 性状权重:{WEIGHT_TRAIT}")
            # 9. 导出 PHYLIP 格式距离矩阵
            save_matrix_phylip(matrix_fused, species_names_clean, f"fused_matrix_{trait_suffix}.phy")
            
            # 🌟 修复1：找出留下来的 34 个物种对应的正确标签，防止 35 vs 34 崩溃
            aligned_labels = [labels[list(species_names).index(name)] for name in species_names_clean]
            
            # --- 实验A：纯深度学习特征树（基线1） ---
            dl_only_dir = os.path.join(config.OUTPUT_DIR, "trees", args.image_type, "dl_only")
            print("\n[5.1/5] 构建纯深度学习特征树...")
            # ===== Bootstrap =====
            # labels 是图像级整数编码，species_names[label] = 物种名
            # 只保留与性状表共有物种，按 species_names_clean 顺序重新编码
            species_to_idx = {name: i for i, name in enumerate(species_names_clean)}
            valid_mask = np.array([
                (lbl < len(species_names)) and (species_names[lbl] in species_to_idx)
                for lbl in labels
            ])
            boot_features = features[valid_mask]          # 图像级特征 (N_images, feature_dim)
            boot_labels = np.array([                       # 图像级标签 (N_images,)，0~33
                species_to_idx[species_names[lbl]] for lbl in labels[valid_mask]
            ])
            print(f"  Bootstrap输入: {len(boot_features)} 张图像, "
                  f"{len(np.unique(boot_labels))} 个物种, "
                  f"每物种平均 {len(boot_features)/len(species_names_clean):.1f} 张")

            build_phylogenetic_trees(
                dl_matrix,
                species_names_clean,
                methods=config.PHYLOGENY_METHODS,
                distance_metric='euclidean',
                output_dir=dl_only_dir,
                raw_features=boot_features,    # ← 修复：图像级特征，不再是 dl_matrix
                raw_labels=boot_labels,        # ← 修复：图像级标签，不再是 aligned_labels
                precomputed_distance_matrix=dist_dl_norm
            )
            dl_norm_df.to_csv(os.path.join(dl_only_dir, "dl_features.csv"))

            # --- 实验B：纯形态性状树（基线2）---
            trait_baseline_dir = f"trait_baseline_{trait_suffix}"
            trait_baseline_output_path = os.path.join(config.OUTPUT_DIR, "trees", args.image_type, trait_baseline_dir)
            print(f"\n[5.2/5] 构建纯形态性状基线树...")

            # 基于原始含 NaN 的 trait_matrix 计算真正的 Gower 距离
            print("  基于原始含缺失值数据计算 Gower 距离（缺失性状自动跳过）...")
            dist_trait_raw = gower_distance_matrix(trait_matrix)

            # 安全网：仅当两物种真的无任何共同非缺失性状时才会出现 NaN
            nan_mask = np.isnan(dist_trait_raw)
            if nan_mask.any():
                nan_pairs = np.argwhere(np.triu(nan_mask, k=1))
                print("  ❌ 以下物种对无任何共同非缺失性状：")
                for i, j in nan_pairs:
                    print(f"     - {species_names_clean[i]}  vs  {species_names_clean[j]}")
                raise ValueError(f"共 {len(nan_pairs)} 对物种无共同非缺失性状，请过滤或补数据。")

            # 建树：不传 raw_features → 跳过 bootstrap（形态性状每物种仅1个向量，不适用）
            build_phylogenetic_trees(
                trait_matrix.values,
                species_names_clean,
                methods=config.PHYLOGENY_METHODS,
                distance_metric='euclidean',
                output_dir=trait_baseline_output_path,
                precomputed_distance_matrix=dist_trait_raw
            )
            trait_norm_df.to_csv(os.path.join(trait_baseline_output_path, "normalized_traits.csv"))
            # --- 实验C：全证据融合树 ---
            fused_tree_dir = f"fused_tree_{trait_suffix}"
            fused_tree_output_path = os.path.join(config.OUTPUT_DIR, "trees", args.image_type, fused_tree_dir)
            print(f"\n[5.3/5] 构建全证据融合树...")
            build_phylogenetic_trees(
                dl_matrix,               
                species_names_clean,
                methods=config.PHYLOGENY_METHODS,
                distance_metric='euclidean',  
                output_dir=fused_tree_output_path,
                precomputed_distance_matrix=matrix_fused 
            )
            fused_features_df.to_csv(os.path.join(fused_tree_output_path, "fused_features.csv"))
            print(f"✓ 所有树文件及配套特征文件生成完成")
            print("\n✅ 全性状自动融合流程完成！")
        else:
            print(f"警告：未找到性状表文件 '{trait_path}'，将仅使用深度学习特征建树。")
            merged_df = pd.DataFrame(species_features, index=species_names_clean)
            dl_matrix = species_features
        # ==================== 全自动性状融合版结束 ====================
        # ==================== 统一特征导出区 (在 phylogeny 模式下也能导出) ====================
        print("\n[特征导出] 正在保存分析所需的矩阵文件 (CSV & NEXUS) ...")
        
        if 'merged_df' in locals():
            traits_df = merged_df.copy()
            dl_col_count = species_features.shape[1]
            new_cols = [f"DL_Feature_{i+1}" for i in range(dl_col_count)] + list(traits_df.columns[dl_col_count:])
            traits_df.columns = new_cols
            
            # 处理 Nexus 导出用的纯数字矩阵
            export_df_for_nexus = traits_df.copy()
            for col in export_df_for_nexus.columns:
                if export_df_for_nexus[col].dtype == 'object' or export_df_for_nexus[col].dtype.name == 'category':
                    labels, uniques = pd.factorize(export_df_for_nexus[col])
                    export_df_for_nexus[col] = labels
            
            # 🌟 核心修正：系统发育软件（如 IQ-TREE）无法识别特殊的 float('nan')
            export_features_array = export_df_for_nexus.fillna(-999).values.astype(float)
        else:
            trait_columns = [f"DL_Feature_{i+1}" for i in range(species_features.shape[1])]
            traits_df = pd.DataFrame(species_features, index=species_names_clean, columns=trait_columns)
            export_features_array = species_features
        # 1. 导出融合特征的 CSV (方便后续 R 语言分析)
        traits_csv_path = os.path.join(
            config.OUTPUT_DIR, 
            "features", 
            f"{args.image_type}_{config.MODEL_TYPE}_fused_traits.csv"
        )
        traits_df.to_csv(traits_csv_path)
        print(f"✓ 包含列名的特征矩阵已保存至: {traits_csv_path}")
        
        # 2. 导出融合特征的 Nexus
        traits_nex_path = os.path.join(
            config.OUTPUT_DIR, 
            "features", 
            f"{args.image_type}_{config.MODEL_TYPE}_fused_traits.nex"
        )
        safe_species_names = [str(name).replace(' ', '_') for name in species_names_clean]
        numpy_to_nexus_file(export_features_array, safe_species_names, traits_nex_path)
        print(f"✓ 纯数值 Nexus 文件已保存至: {traits_nex_path}")
        if args.mode in ['extract', 'full']:
            detect_outliers(features, labels, image_paths, species_names, config.OUTPUT_DIR)
            
            if config.SAVE_FEATURES:
                feature_path = os.path.join(
                    config.OUTPUT_DIR, 
                    "features", 
                    f"{args.image_type}_{config.MODEL_TYPE}_features.npz"
                )
                save_features(features, labels, species_names, feature_path)
        # ===================================================================================
        print("\n可视化特征分布与生成实验图表...")
        tsne_path = os.path.join(
            config.OUTPUT_DIR, 
            "figures", 
            f"{args.image_type}_{config.MODEL_TYPE}_tsne.pdf"
        )
        
        if len(features) > 0:
            visualize_features(features, labels, species_names, tsne_path)
        
        if args.mode in ['phylogeny', 'full']:
            
            # 🌟 修复核心：动态提取画图专用的 34 物种对齐特征矩阵
            if 'matrix_fused' in locals():
                dist_matrix_for_plot = matrix_fused
                features_for_plot = np.nan_to_num(fused_features_df.values, nan=0.0)
                print("使用融合后的矩阵生成最终评估图表...")
            else:
                dist_matrix_for_plot = calculate_distance_matrix(species_features, metric='euclidean')
                features_for_plot = np.nan_to_num(species_features, nan=0.0)
            
            dist_matrix_path = os.path.join(
                config.OUTPUT_DIR,
                "figures",
                f"{args.image_type}_{config.MODEL_TYPE}_distance_matrix.pdf"
            )
            plot_distance_matrix(dist_matrix_for_plot, species_names_clean, dist_matrix_path)
            
            print("\n生成实验结果图...")
            
            dist_dist_path = os.path.join(config.OUTPUT_DIR, "figures", f"{args.image_type}_{config.MODEL_TYPE}_distance_distribution.pdf")
            plot_distance_distribution(dist_matrix_for_plot, dist_dist_path)
            
            corr_matrix_path = os.path.join(config.OUTPUT_DIR, "figures", f"{args.image_type}_{config.MODEL_TYPE}_feature_correlation.pdf")
            plot_feature_correlation(features_for_plot, species_names_clean, corr_matrix_path)
            
            feature_comp_path = os.path.join(config.OUTPUT_DIR, "figures", f"{args.image_type}_{config.MODEL_TYPE}_species_feature_comparison.pdf")
            plot_species_feature_comparison(features_for_plot, species_names_clean, feature_comp_path)
            
            dendrogram_path = os.path.join(config.OUTPUT_DIR, "figures", f"{args.image_type}_{config.MODEL_TYPE}_clustering_dendrogram.pdf")
            plot_clustering_dendrogram(dist_matrix_for_plot, species_names_clean, dendrogram_path)
            
            feature_stats_path = os.path.join(config.OUTPUT_DIR, "figures", f"{args.image_type}_{config.MODEL_TYPE}_feature_statistics.pdf")
            plot_feature_statistics(features_for_plot, species_names_clean, feature_stats_path)
            
            print("\n" + "=" * 60)
            print("🚀 全部流程圆满完成！")
            print("=" * 60)
            print(f"特征文件: {os.path.join(config.OUTPUT_DIR, 'features')}")
            print(f"系统发育树: {os.path.join(config.OUTPUT_DIR, 'trees', args.image_type)}")
            print(f"可视化结果: {os.path.join(config.OUTPUT_DIR, 'figures')}")
            print("=" * 60)
        else:
            print("\n跳过系统发育树绘图环节")
    else:
        print("\n跳过特征提取和系统发育树构建")

if __name__ == '__main__':
    main()
