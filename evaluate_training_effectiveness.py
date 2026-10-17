# -*- coding: utf-8 -*-
"""
训练有效性量化评估（不依赖 t-SNE / bootstrap）
新增：训练前(ImageNet 未微调) vs 训练后(Triplet 微调) 对比 —— 证明"训练带来了提升"。
  运行前置：先在服务器跑 extract_baseline_features.py 生成 *_PRETRAIN_features.npz
  若没有 PRETRAIN 文件，则只评估训练后并与随机基线对比。
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score,
                             roc_auc_score)
from sklearn.preprocessing import normalize

RNG = 42
np.random.seed(RNG)

def balanced_sample(X, y, max_per_class=40, seed=RNG):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        if len(ci) > max_per_class:
            ci = rng.choice(ci, max_per_class, replace=False)
        idx.extend(ci)
    idx = np.array(idx)
    return X[idx], y[idx]

def ap_from_ranking(is_relevant):
    rel = np.asarray(is_relevant, dtype=float)
    if rel.sum() == 0:
        return 0.0
    cum = np.cumsum(rel)
    prec = cum / np.arange(1, len(rel) + 1)
    return float((prec * rel).sum() / rel.sum())

def compute_metrics(X, y, species, verbose_dist=False):
    classes = np.unique(y)
    n_class = len(classes); n_img = len(X)
    R = {"images": n_img, "species": n_class,
         "random_baseline_%": round(100.0 / n_class, 2)}

    # 全量距离（特征 L2 归一化）
    sim = np.clip(X @ X.T, -1, 1)
    dist = np.sqrt(np.clip(2.0 - 2.0 * sim, 0, 4))
    np.fill_diagonal(dist, np.inf)
    order = np.argsort(dist, axis=1)
    y_sorted = y[order]

    per_r1, APs, rec1, rec2, rec4, rec8 = {}, [], [], [], [], []
    for i in range(n_img):
        same = (y_sorted[i] == y[i]).astype(int)
        n_rel = (y == y[i]).sum() - 1
        APs.append(ap_from_ranking(same))
        rec1.append(same[0])
        rec2.append(same[:2].sum() / min(n_rel, 2))
        rec4.append(same[:4].sum() / min(n_rel, 4))
        rec8.append(same[:8].sum() / min(n_rel, 8))
        per_r1.setdefault(y[i], []).append(same[0])

    R["mAP_%"] = round(100 * np.mean(APs), 2)
    R["Recall@1_%"] = round(100 * np.mean(rec1), 2)
    R["Recall@2_%"] = round(100 * np.mean(rec2), 2)
    R["Recall@4_%"] = round(100 * np.mean(rec4), 2)
    R["Recall@8_%"] = round(100 * np.mean(rec8), 2)
    R["Recall@1_macro_%"] = round(100 * np.mean([np.mean(v) for v in per_r1.values()]), 2)

    knn1, knn3, knn5, lin, cent = [], [], [], [], []
    for t in range(5):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=RNG + t)
        for k, lst in [(1, knn1), (3, knn3), (5, knn5)]:
            clf = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
            lst.append(balanced_accuracy_score(yte, clf.predict(Xte)))
        lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        lin.append(balanced_accuracy_score(yte, lr.predict(Xte)))
        cen = normalize(np.vstack([Xtr[ytr == c].mean(0) for c in classes]))
        pred = classes[np.argmin(1 - Xte @ cen.T, axis=1)]
        cent.append(balanced_accuracy_score(yte, pred))

    R["kNN1_bal_acc_%"] = round(100 * np.mean(knn1), 2)
    R["kNN3_bal_acc_%"] = round(100 * np.mean(knn3), 2)
    R["kNN5_bal_acc_%"] = round(100 * np.mean(knn5), 2)
    R["linear_probe_%"] = round(100 * np.mean(lin), 2)
    R["nearest_centroid_%"] = round(100 * np.mean(cent), 2)
    R["split_std_%"] = round(100 * np.mean([np.std(knn1), np.std(knn3)]), 2)

    Xs, ys = balanced_sample(X, y, 40)
    ds = np.sqrt(np.clip(2 - 2 * np.clip(Xs @ Xs.T, -1, 1), 0, 4))
    iu = np.triu_indices(len(ys), 1)
    dv = ds[iu]; samep = (ys[iu[0]] == ys[iu[1]]).astype(int)
    within = dv[samep == 1]; between = dv[samep == 0]
    R["distance_ROC_AUC"] = round(float(roc_auc_score(samep, -dv)), 4)
    R["within_dist_mean"] = round(float(within.mean()), 4)
    R["between_dist_mean"] = round(float(between.mean()), 4)
    R["separation_ratio"] = round(float(between.mean() / within.mean()), 3)
    R["within_overlap_%"] = round(100 * float((within > np.median(between)).mean()), 2)

    Xc, yc = balanced_sample(X, y, 30)
    R["silhouette"] = round(float(silhouette_score(Xc, yc)), 4)
    R["calinski_harabasz"] = round(float(calinski_harabasz_score(Xc, yc)), 1)
    R["davies_bouldin"] = round(float(davies_bouldin_score(Xc, yc)), 4)

    aux = None
    if verbose_dist:
        rows = []
        for c in classes:
            nm = species[c] if c < len(species) else str(c)
            cm = X[y == c].mean(0)
            comp = float(np.mean(np.linalg.norm(X[y == c] - cm, axis=1)))
            rows.append({"species": nm, "images": int((y == c).sum()),
                         "Recall@1_%": round(100 * np.mean(per_r1[c]), 1),
                         "intra_compactness": round(comp, 4)})
        aux = (pd.DataFrame(rows).sort_values("Recall@1_%"), within, between)
    return R, aux

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["features"].astype(np.float64); y = d["labels"].astype(int)
    species = [str(s) for s in d["species_names"]]
    return X, y, species

if __name__ == "__main__":
    summary = {}
    for name in ["specimen", "habitat"]:
        fdir = os.path.join("outputs", name, "features")
        post_p = os.path.join(fdir, f"{name}_resnet50_features.npz")
        pre_p = os.path.join(fdir, f"{name}_resnet50_PRETRAIN_features.npz")
        X, y, species = load_npz(post_p)
        Rpost, aux = compute_metrics(X, y, species, verbose_dist=True)
        per_df, within, between = aux

        compare = pd.DataFrame({"finetuned": Rpost})
        if os.path.exists(pre_p):
            Xp, yp, sp = load_npz(pre_p)
            Rpre, _ = compute_metrics(Xp, yp, sp)
            compare.insert(0, "pretrained", Rpre)
            compare["delta"] = compare["finetuned"] - compare["pretrained"]
            print("="*70, "\n", name, "：pretrained vs finetuned")
        else:
            print("="*70, "\n", name, "：(未找到 PRETRAIN 对照，仅训练后)")
        print(compare.to_string())
        print("\nWorst-8 species:")
        print(per_df.head(8).to_string(index=False))

        compare.to_csv(os.path.join(fdir, f"{name}_before_after_compare.csv"),
                       encoding="utf-8-sig")
        per_df.to_csv(os.path.join(fdir, f"{name}_per_species_validation.csv"),
                      index=False)

        # 分布图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(between, bins=60, density=True, alpha=0.6, color="#4a7a9a",
                label="Between species")
        ax.hist(within, bins=60, density=True, alpha=0.7, color="#4a8a6b",
                label="Within species")
        ax.set_xlabel("Euclidean distance"); ax.set_ylabel("Density")
        ax.set_title(f"{name}: within/between distance (AUC={Rpost['distance_ROC_AUC']})",
                     fontweight="bold")
        ax.legend(); fig.tight_layout()
        fp = os.path.join("outputs", name, "figures",
                          f"{name}_within_between_distance.pdf")
        fig.savefig(fp, format="pdf"); plt.close(fig)

        summary[name] = {"pretrained": Rpre if os.path.exists(pre_p) else None,
                         "finetuned": Rpost}
        print("\nfigure:", fp)

    with open("training_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nSaved JSON & per-dataset compare CSVs.")
