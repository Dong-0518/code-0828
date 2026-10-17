# -*- coding: utf-8 -*-
"""
提取【训练前 = ImageNet 预训练、未做 Triplet 微调】的特征，
作为对照基线，用于证明"训练带来了提升"。

在服务器上、项目根目录运行：
    python extract_baseline_features.py --image_type specimen
    python extract_baseline_features.py --image_type habitat
输出：
    outputs/<image_type>/features/<image_type>_resnet50_PRETRAIN_features.npz
结构与训练后的 <image_type>_resnet50_features.npz 完全一致。
"""
import os, argparse
import numpy as np
from config import Config
from utils import set_seed
from data_loader import load_dataset, create_dataloaders
from models import create_model
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_type", type=str, default="specimen",
                        choices=["specimen", "habitat"])
    parser.add_argument("--model_type", type=str, default="resnet50")
    args = parser.parse_args()

    config = Config()
    config.MODEL_TYPE = args.model_type
    config.OUTPUT_DIR = os.path.join("outputs", args.image_type)
    set_seed(config.SEED)

    data_path = config.SPECIMEN_PATH if args.image_type == "specimen" else config.HABITAT_PATH
    if not os.path.exists(data_path):
        data_path = ("数据集/标本图像/results_plantsam"
                     if args.image_type == "specimen" else "数据集/生境图像")
    print("data path:", data_path)

    image_paths, labels, species_names = load_dataset(data_path, args.image_type)
    _, _, _, all_eval_loader = create_dataloaders(
        image_paths, labels,
        batch_size=config.BATCH_SIZE,
        train_ratio=config.TRAIN_RATIO, val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO, use_triplet=True,
        image_size=config.IMAGE_SIZE, num_workers=config.NUM_WORKERS,
        model_type=config.MODEL_TYPE)

    # 关键：只用 ImageNet 预训练权重，不加载任何微调 checkpoint
    model = create_model(model_type=config.MODEL_TYPE,
                         feature_dim=config.FEATURE_DIM,
                         pretrained=True, use_triplet=True)
    model.eval().to(config.DEVICE)

    feats, ys = [], []
    with torch.no_grad():
        for batch in all_eval_loader:
            images, blabels, _ = batch
            images = images.to(config.DEVICE)
            f = model.feature_extractor(images)   # 与训练后同一套提取路径
            feats.append(f.cpu().numpy())
            ys.extend(np.asarray(blabels))

    feats = np.vstack(feats)
    ys = np.array(ys)
    out = os.path.join(config.OUTPUT_DIR, "features",
                       f"{args.image_type}_{config.MODEL_TYPE}_PRETRAIN_features.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, features=feats, labels=ys, species_names=np.array(species_names))
    print("saved:", out, feats.shape)

if __name__ == "__main__":
    main()
