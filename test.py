import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from scipy.io import loadmat
import numpy as np
from unet3d import UNet
from Aunet3d import AUNet
import os
class MatDataset(Dataset):
    def __init__(self, data_paths, label_paths):
        self.data_paths = data_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # 加载.mat文件

        data = loadmat(self.data_paths[idx])[self.data_paths[idx][-17:-4]]
        label = loadmat(self.label_paths[idx])[self.data_paths[idx][-17:-4]]
        # **归一化 (0-1)**
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)  # 避免除以零
        label = (label - np.min(label)) / (np.max(label) - np.min(label) + 1e-8)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

test_data = pd.read_csv('test_files.csv')

test_dataset = MatDataset(test_data['data_path'], test_data['label_path'])

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
output_dir="test_result0.05b"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if  torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

# 加载模型
model = AUNet().to(device)
#model = UNet().to(device)
model_weights = torch.load('model/best_networkhMLOS.pth')
model.load_state_dict(model_weights)

model.eval()
train_Q=0
total=0
with torch.no_grad():
    output = None
    q_sum=0
    for data_path,(data, labels) in zip(test_data['data_path'], test_loader):

        data = data.to(device)
        labels = labels.to(device)
        data = torch.unsqueeze(data, 1)
        labels = torch.unsqueeze(labels, 1)
        outputs = model(data)
        outputs_flat = torch.flatten(outputs, start_dim=1)
        labels_flat = torch.flatten(labels, start_dim=1)
        output = outputs.cpu().numpy()
        output_reshaped = output[0,0]
        file_name = os.path.basename(data_path)  # 获取文件名，如 'data_001.mat'
        file_name_without_extension = os.path.splitext(file_name)[0]  # 去掉扩展名 'data_001'

        # 用数据文件名命名输出文件
        print(f"Output shape: {output_reshaped.shape}")
        np.save(os.path.join(output_dir, f'{file_name_without_extension}.npy'), output_reshaped)

        print(f"Saved output {file_name_without_extension}.npy")
        # 保存为.npy格式的文件
    # np.save('test_output/' + '8' + '.npy', output)
        Q = torch.sum(torch.sum(outputs_flat * labels_flat, dim=1) / torch.sqrt(torch.sum(outputs_flat * outputs_flat, dim=1) * torch.sum(labels_flat * labels_flat, dim=1)))
        train_Q += Q.item()
        total+=1
    print(f"Total number of samples: {total}")
    print(f"everage Q : {train_Q/total}")