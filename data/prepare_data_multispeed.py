import h5py
import numpy as np
import torch
import os
# import matplotlib.pyplot as plt


def matlab_to_numpy(matlab_array):
    """
    转换 MATLAB 数组到 NumPy 数组，自动调整维度顺序。
    :param matlab_array: 从 MATLAB 文件中读取的数组。
    :return: 维度顺序调整后的 NumPy 数组。
    """
    # 计算维度数量
    num_dims = matlab_array.ndim

    # 生成新的维度顺序
    new_order = list(range(num_dims))[::-1]  # 例如，对于四维就是[3, 2, 1, 0]

    # 调整维度顺序
    return np.transpose(matlab_array, axes=new_order)


def load_and_split_data(filepath, list1, list2):
    # 使用h5py打开文件
    with h5py.File(filepath, 'r') as file:
        # MATLAB文件通常将变量存储在名为'#refs#'的组中，你可能需要根据实际文件结构调整路径
        xtrain = matlab_to_numpy(np.array(file['/xtrain']))
        ytrain = matlab_to_numpy(np.array(file['/ytrain']))

        xtrain = torch.tensor(xtrain, dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.float32)

        # 打印形状以检查是否正确加载
        print("XTrain shape:", xtrain.shape)
        print("YTrain shape:", ytrain.shape)

        # # 画图检查数据，检查xtrain[0,0,:,0]是否为一个合理的信号
        # plt.plot(xtrain[0, 0, :, 0])
        # plt.show()

        # 根据提供的索引分割训练集和测试集
        x_train_set = xtrain[list1, ...]
        y_train_set = ytrain[list1, ...]
        x_test_set = xtrain[list2, ...]
        y_test_set = ytrain[list2, ...]

    return (x_train_set, y_train_set), (x_test_set, y_test_set)


# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录路径
current_dir_path = os.path.dirname(current_file_path)
# 定义要合并的相对路径
relative_path = r'raw\Data_MultiSpeed_try.mat'  # 这是加载的数据文件
# 使用os.path.join合并路径
file_path = os.path.join(current_dir_path, relative_path)
print("数据文件路径:", file_path)

# 设置切片的索引
list1 = [i for i in range(1000) if i % 5 != 4]
list2 = [i for i in range(1000) if i % 5 == 4]
print("训练集索引:", list1)
print("测试集索引:", list2)

# 加载数据并分割
train_set, test_set = load_and_split_data(file_path, list1, list2)

# 输出结果查看
print("Training set (X):", train_set[0].shape)
print("Training set (Y):", train_set[1].shape)
print("Testing set (X):", test_set[0].shape)
print("Testing set (Y):", test_set[1].shape)

# # 继续画图检查数据，train_set[0][0,0,:,0]是否为一个合理的信号
# plt.plot(train_set[0][0, 0, :, 0])
# plt.show()

# 保存数据到processed文件夹
processed_dir = os.path.join(current_dir_path, 'processed')
if not os.path.exists(processed_dir):
    processed_dir = current_dir_path

# 保存数据为mat文件
train_path = os.path.join(processed_dir, 'Data_train_rig_base.pth')
test_path = os.path.join(processed_dir, 'Data_test_rig_base.pth')

# with h5py.File(train_path, 'w') as train_file:
#     train_file.create_dataset('XData', data=train_set[0])
#     train_file.create_dataset('YData', data=train_set[1])

# with h5py.File(test_path, 'w') as test_file:
#     test_file.create_dataset('XData', data=test_set[0])
#     test_file.create_dataset('YData', data=test_set[1])

torch.save({'XData': train_set[0], 'YData': train_set[1]}, train_path)
torch.save({'XData': test_set[0], 'YData': test_set[1]}, test_path)
