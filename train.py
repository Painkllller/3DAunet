import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import pandas as pd
from sklearn.metrics import accuracy_score
from unet3d import UNet
from torch.utils.tensorboard import SummaryWriter
from early_stopping import EarlyStopping
from Aunet3d import AUNet


class MatDataset(Dataset):
    def __init__(self, data_paths, label_paths):
        self.data_paths = data_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # 加载.mat文件

        data = loadmat(self.data_paths[idx])[self.data_paths[idx][-17:-4]]#需要根据你的.mat文件中的变量名进行调整
        label = loadmat(self.label_paths[idx])[self.data_paths[idx][-17:-4]]  # 同上

        # **归一化 (0-1)**
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)  # 避免除以零
        label = (label - np.min(label)) / (np.max(label) - np.min(label) + 1e-8)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# 加载训练和测试数据
train_data = pd.read_csv('train_files.csv')
test_data = pd.read_csv('test_files.csv')

# 创建数据集
train_dataset = MatDataset(train_data['data_path'], train_data['label_path'])
test_dataset = MatDataset(test_data['data_path'], test_data['label_path'])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#查看能否调用cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

# 初始化模型
model = AUNet().to(device)
#model = UNet().to(device)


# 定义损失函数和优化器
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
#lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)
num_epochs = 6000

# 加入质量因子
Lambda = 0.75
mb = 16

# 记录损失列表
train_losses = []
test_losses = []

# 设置早停
early_stopping = EarlyStopping('model')

# 训练模型
for epoch in range(num_epochs):
    model.train()
    total = 0
    train_loss = 0
    train_Q = 0
    num_data=0
    for data, labels in train_loader:
        # 前向传播
        data, labels = data.to(device), labels.to(device)
        data = torch.unsqueeze(data, 1)
        labels = torch.unsqueeze(labels, 1)
        outputs = model(data)

        outputs_flat = torch.flatten(outputs, start_dim=1)
        labels_flat = torch.flatten(labels, start_dim=1)

        # 加入Qpr,Lambda=0.75
        loss = criterion(outputs, labels) + Lambda * (1 - (torch.sum(torch.sum(outputs_flat*labels_flat,dim=1) / torch.sqrt(torch.sum(outputs_flat*outputs_flat,dim=1) * torch.sum(labels_flat*labels_flat,dim=1))) / mb))
        train_loss += loss.item()
        total += labels.size(0)

        # 计算质量因子Q
        Q = torch.sum(torch.sum(outputs_flat*labels_flat,dim=1) / torch.sqrt(torch.sum(outputs_flat*outputs_flat,dim=1) * torch.sum(labels_flat*labels_flat,dim=1)))
        train_Q += Q.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_data += 1
        if num_data % 100 == 0:
            print(f'number of data:{num_data}')

    # 保存训练损失
    train_losses.append(train_loss / total)

    print(f'Epoch {epoch+1}/{num_epochs},  Train Loss: {train_loss / total}')
    print(f'Epoch {epoch + 1}/{num_epochs},  Train Q: {train_Q / total}')

    # 测试模型
    model.eval()
    with torch.no_grad():
        total = 0
        test_loss = 0
        test_Q = 0
        output = None
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            data = torch.unsqueeze(data, 1)
            labels = torch.unsqueeze(labels, 1)

            outputs = model(data)

            outputs_flat = torch.flatten(outputs, start_dim=1)
            labels_flat = torch.flatten(labels, start_dim=1)

            total += labels.size(0)
            test_loss += criterion(outputs, labels).item() + Lambda * (1 - (torch.sum(torch.sum(outputs_flat*labels_flat,dim=1) / torch.sqrt(torch.sum(outputs_flat*outputs_flat,dim=1) * torch.sum(labels_flat*labels_flat,dim=1))) / mb))

            # 计算质量因子Q
            Q = torch.sum(torch.sum(outputs_flat * labels_flat, dim=1) / torch.sqrt(torch.sum(outputs_flat * outputs_flat, dim=1) * torch.sum(labels_flat * labels_flat, dim=1)))
            test_Q += Q.item()

            output = outputs
        output = output.cpu().numpy()

        # 保存为.npy格式的文件
        #np.save('output/'+str(epoch)+'.npy', output)

        # 保存测试损失
        test_losses.append(test_loss / total)

        # 早停
        early_stopping(test_loss / total, model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

        # 输出测试精度和测试loss
        print(f'Epoch {epoch+1}/{num_epochs},  Test Loss: {test_loss / total}')
        print(f'Epoch {epoch + 1}/{num_epochs},  Test Q: {test_Q / total}')

    #lr_scheduler.step()

# 绘制loss曲线