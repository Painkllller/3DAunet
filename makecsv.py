import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 设置文件夹的路径
# data_folder_path = './test'
# label_folder_path = './test_label'

data_folder_path = './SubVolumes_reconstruct'
label_folder_path = './SubVolumes_exact'
# 创建两个空列表来保存所有的.mat文件路径
data_files_list = []
label_files_list = []

# 遍历数据文件夹
for subdir, dirs, files in os.walk(data_folder_path):
    for file in files:
        # 检查文件扩展名是否为.mat
        if file.endswith('.mat'):
            # 获取文件的相对路径
            rel_path = os.path.abspath(os.path.join(subdir, file))
            # 将相对路径添加到列表中
            data_files_list.append(rel_path)

# 遍历标签文件夹
for subdir, dirs, files in os.walk(label_folder_path):
    for file in files:
        # 检查文件扩展名是否为.mat
        if file.endswith('.mat'):
            # 获取文件的相对路径
            rel_path = os.path.abspath(os.path.join(subdir, file))
            # 将相对路径添加到列表中
            label_files_list.append(rel_path)

# 创建一个DataFrame
df = pd.DataFrame({'data_path': data_files_list, 'label_path': label_files_list})

# 使用sklearn的train_test_split函数划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2,shuffle=False)
# test_df = df
# 将DataFrame保存为CSV文件
train_df.to_csv('train_files.csv', index=False)
test_df.to_csv('test_files.csv', index=False)

print('Train and test .mat file paths have been written to train_files.csv and test_files.csv')