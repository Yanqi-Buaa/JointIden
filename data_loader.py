import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(file_path):
    # 加载 .pth 文件
    data = torch.load(file_path)
    XData = data['XData']
    YData = data['YData']

    return XData, YData


def normalize_data(X_train, X_test, dim):
    # 计算训练集的均值和标准差，按训练集的均值和标准差进行归一化
    X_train_mean = X_train.mean(dim, keepdim=True)
    X_train_std = X_train.std(dim, keepdim=True)
    X_train_norm = (X_train - X_train_mean) / X_train_std
    X_test_norm = (X_test - X_train_mean) / X_train_std

    return X_train_norm, X_test_norm


def get_dataloaders(train_file_path, test_file_path, norm_dim, batch_size):
    X_train, y_train = load_data(train_file_path)
    X_test, y_test = load_data(test_file_path)

    X_train, X_test = normalize_data(X_train, X_test, norm_dim)

    if torch.cuda.is_available():
        # 将张量移动到默认的 GPU 上
        X_train = X_train.to('cuda')
        y_train = y_train.to('cuda')
        X_test = X_test.to('cuda')
        y_test = y_test.to('cuda')
        print("Moved Tensor to:", X_train.device)
    else:
        print("CUDA is not available. Tensor is on CPU.")

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader
