import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import loadmat



# 假设小块尺寸
patch_size = (64, 64, 32)  # 需要与你的 UNet 训练数据尺寸一致
full_size = (768, 768, 384)

# 读取所有小块
output_dir = "test_result0.05"

files = sorted(os.listdir(output_dir))  # 确保文件是按顺序命名的
output_patches = [np.load(os.path.join(output_dir, f)) for f in files]

# 重新组合为完整的粒子场
reconstructed_output = np.zeros(full_size)
reconstructed_label = np.zeros(full_size)

# 读取 label 数据
label_patches = []
test_data = pd.read_csv('test_files.csv')
for label_path in test_data["label_path"]:
    label_mat = loadmat(label_path)  # 读取 .mat 文件
    label_name = label_path[-17:-4]  # 假设数据存储在 mat 文件内部
    label_patches.append(label_mat[label_name])  # 取出 NumPy 数组
# 假设按照 (4,4,4) 方式切割
index = 0
for z in range(0, full_size[2], patch_size[2]):
    for y in range(0, full_size[1], patch_size[1]):
        for x in range(0, full_size[0], patch_size[0]):
            reconstructed_output[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] = output_patches[index]
            reconstructed_label[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]] = label_patches[index]
            index += 1
def dice_loss(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def q_loss(pred, target):
    # pred_flat = pred.view(pred.shape[0], -1)  # 展平成 (batch_size, N)
    # target_flat = target.view(target.shape[0], -1)  # 同样展平
    #
    # numerator = torch.sum(pred_flat * target_flat, dim=1)
    # denominator = torch.sqrt(torch.sum(pred_flat * pred_flat, dim=1) * torch.sum(target_flat * target_flat, dim=1))
    #
    # Q = torch.sum(numerator / (denominator + 1e-8))  # 避免除零
    # pred_flat = torch.flatten(pred, start_dim=1)
    # target_flat = torch.flatten(target, start_dim=1)
    # Q = torch.sum(torch.sum(pred_flat * target_flat, dim=1) / torch.sqrt(torch.sum(pred_flat * pred_flat, dim=1) * torch.sum(target_flat * target_flat, dim=1)))
    # 展平为向量（整个场视为一个样本）
    pred_flat = pred.view(-1)  # 形状变为 (768*768*384,)
    target_flat = target.view(-1)

    # 计算余弦相似度
    dot_product = torch.sum(pred_flat * target_flat)
    norm_pred = torch.sqrt(torch.sum(pred_flat ** 2) + 1e-8)
    norm_target = torch.sqrt(torch.sum(target_flat ** 2) + 1e-8)
    q_value = dot_product / (norm_pred * norm_target)
    return q_value

# 计算损失 (使用 MSE 作为示例)
reconstructed_label = (reconstructed_label - np.min(reconstructed_label)) / (np.max(reconstructed_label) - np.min(reconstructed_label) + 1e-8)
output_tensor = torch.tensor(reconstructed_output, dtype=torch.float32)
label_tensor = torch.tensor(reconstructed_label, dtype=torch.float32)
outputs_normalized = (output_tensor - output_tensor.min()) / (output_tensor.max() - output_tensor.min() + 1e-8)
labels_normalized = label_tensor
loss_q = q_loss(outputs_normalized, labels_normalized)  # 计算 Q 损失
loss_mse = F.mse_loss(output_tensor, label_tensor)
loss_dice = dice_loss(output_tensor, label_tensor)
print(f"Dice Loss: {loss_dice.item()}")
print(f"MSE Loss: {loss_mse.item()}")
print(f"Q Loss: {loss_q.item()}")
np.save("final_output.npy", reconstructed_output)
np.save("final_label.npy", reconstructed_label)
print("Output min/max:", reconstructed_output.min(), reconstructed_output.max())
print("Label min/max:", reconstructed_label.min(), reconstructed_label.max())