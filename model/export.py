# export_siamese_algorithm.py

import os
import torch
import numpy as np
import json
import torch.nn as nn # 需要 nn.Module 和 nn.Sequential

# =======================================================
# 0. 配置和目录
# =======================================================
OUTPUT_DIR = "output"
PTH_PATH = os.path.join(OUTPUT_DIR, "checkpoints_siamese.pth")
NORM_CONSTANTS_PATH = os.path.join(OUTPUT_DIR, "norm_constants.json")
OUTPUT_DIR_ALGO = "output/siamese_weights" # 导出权重的新目录

if not os.path.exists(PTH_PATH):
    raise FileNotFoundError(f"未找到模型文件: {PTH_PATH}. 请先运行训练脚本。")
if not os.path.exists(NORM_CONSTANTS_PATH):
    raise FileNotFoundError(f"未找到归一化常数文件: {NORM_CONSTANTS_PATH}. 请先运行更新后的训练脚本。")

os.makedirs(OUTPUT_DIR_ALGO, exist_ok=True)


# =======================================================
# 1. 重新定义模型架构 (必须与训练时完全一致)
# =======================================================
class SiameseColorNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        # (L,a,b) → 嵌入向量
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.ReLU()
        )
        # |e1 - e2| → 预测 log1p(DE)（归一化后的）
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )

    def forward(self, x):
        # 仅为完整性定义，实际导出中不需要前向传播
        pass 

# =======================================================
# 2. 加载权重并提取参数
# =======================================================
model = SiameseColorNet(emb_dim=128)
# 加载到 CPU 上，无需 GPU 环境
state_dict = torch.load(PTH_PATH, map_location=torch.device('cpu')) 
model.load_state_dict(state_dict)
model.eval()

print(f"正在从 {PTH_PATH} 导出模型参数...")

# 存储所有提取的参数的字典
extracted_params = {}

# 遍历模型中的所有命名参数 (W 和 B)
for name, param in model.named_parameters():
    # 将 PyTorch Tensor 转换为 NumPy 数组
    np_array = param.data.numpy() 
    
    # 规范化名称，例如：encoder.0.weight -> encoder_0_W
    clean_name = name.replace('.', '_').replace('weight', 'W').replace('bias', 'B')
    
    # 将每个参数单独保存为 .npy 文件
    save_path = os.path.join(OUTPUT_DIR_ALGO, f"{clean_name}.npy")
    np.save(save_path, np_array)
    extracted_params[clean_name] = np_array
    print(f"已保存参数: {clean_name}.npy, 形状: {np_array.shape}")

# =======================================================
# 3. 提取归一化常数
# =======================================================
with open(NORM_CONSTANTS_PATH, "r") as f:
    norm_constants = json.load(f)

y_mean = norm_constants["y_mean"]
y_std = norm_constants["y_std"]

# 将归一化常数也保存为易读的 .json 文件到新目录
with open(os.path.join(OUTPUT_DIR_ALGO, "norm_constants.json"), "w") as f:
    json.dump({"y_mean": y_mean, "y_std": y_std}, f, indent=2)
    
print(f"\n已将归一化常数和 {len(extracted_params)} 个权重/偏置矩阵导出到 {OUTPUT_DIR_ALGO}。")