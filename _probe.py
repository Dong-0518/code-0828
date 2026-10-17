# -*- coding: utf-8 -*-
import os, sys
import numpy as np

print("python:", sys.version.split()[0])
try:
    import sklearn; print("sklearn:", sklearn.__version__)
except Exception as e:
    print("sklearn NOT available:", e)

for name in ["specimen", "habitat"]:
    p = os.path.join("outputs", name, "features", f"{name}_resnet50_features.npz")
    if not os.path.exists(p):
        print(f"[missing] {p}"); continue
    d = np.load(p, allow_pickle=True)
    print("="*60)
    print(name, "keys:", list(d.keys()))
    X = d["features"]; y = d["labels"]
    print("features shape:", X.shape, "dtype:", X.dtype)
    print("labels shape:", y.shape)
    u, c = np.unique(y, return_counts=True)
    print("num species:", len(u))
    print("images per species: min=%d max=%d mean=%.1f" % (c.min(), c.max(), c.mean()))
    norms = np.linalg.norm(X, axis=1)
    print("feature norm: min=%.4f max=%.4f mean=%.4f" % (norms.min(), norms.max(), norms.mean()))
    if "species_names" in d:
        sn = d["species_names"]
        print("species_names sample:", list(sn[:5]))
