import numpy as np

# 加载新电脑提取的特征
data = np.load("outputs/features/habitat_resnet50_features.npz", allow_pickle=True)

print("1. 检查物种顺序：前 5 个物种名是啥？")
print(data['species_names'][:5])

print("\n2. 检查特征数值：第 0 个样本的前 5 个特征值是啥？")
print(data['features'][0][:5])