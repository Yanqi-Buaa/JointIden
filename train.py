import torch
import torch.nn as nn
import torch.optim as optim
import configargparse
# import models.model as mdl
from data_loader import get_dataloaders
import time


def train_model(model, trainloader, testloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            # inputs, labels = inputs.float(), labels.float()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(1)

        epoch_loss = running_loss / len(trainloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        if (epoch+1) % 10 == 0:
            # eval_model(model, trainloader, testloader, num_classes=4)
            eval_model_basic(model, trainloader, testloader)


def eval_model(model, trainloader, testloader, num_classes):
    correct_per_class_train = torch.zeros(num_classes)
    total_per_class_train = torch.zeros(num_classes)
    correct_per_class_test = torch.zeros(num_classes)
    total_per_class_test = torch.zeros(num_classes)

    model.eval()
    with torch.no_grad():
        # 计算训练集上的分类正确率
        for inputs, labels in trainloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            true_labels = torch.argmax(labels, dim=1)

            for label, prediction in zip(true_labels, predicted):
                total_per_class_train[label] += 1
                if label == prediction:
                    correct_per_class_train[label] += 1

        # 计算测试集上的分类正确率
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            true_labels = torch.argmax(labels, dim=1)

            for label, prediction in zip(true_labels, predicted):
                total_per_class_test[label] += 1
                if label == prediction:
                    correct_per_class_test[label] += 1

    # 输出每个类别的正确率
    for i in range(num_classes):
        train_acc = 100 * correct_per_class_train[i] / total_per_class_train[i]
        test_acc = 100 * correct_per_class_test[i] / total_per_class_test[i]
        print(f'Class {i} - Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')

    model.train()  # Switch back to training mode


def eval_model_basic(model, trainloader, testloader):
    model.eval()
    with torch.no_grad():
        correct_train = 0
        total_train = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.float(), labels.float()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        correct_test = 0
        total_test = 0
        for inputs, labels in testloader:
            inputs, labels = inputs.float(), labels.float()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        test_accuracy = 100 * correct_test / total_test
        print(f'Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Accuracy: {test_accuracy:.2f}%')

    model.train()  # 切换回训练模式
    return train_accuracy, test_accuracy


def getmodel(opts):
    model_name = opts.model_name
    if model_name == 'TransformerEncoder':
        from models.model.TransformerEncoder import TransformerEncoder as mdl
        model = mdl(opts.input_dim, opts.model_dim, opts.num_heads, opts.num_layers, opts.num_classes)
    elif model_name == 'MultiTransformerEncoder':
        from models.model.MultiTransformerClassifier import MultiTransformerClassifier as mdl
        model = mdl(opts.input_dim, opts.model_dim, opts.num_heads, opts.num_layers, opts.trans_output_dim, opts.n_segments, opts.num_classes)
    elif model_name == 'GRUClassifier':
        from models.model.GRUClassifier import GRUClassifier as mdl
        model = mdl(opts.input_dim, opts.model_dim, opts.num_layers, opts.num_classes)

    return model


def main():
    # 创建解析器
    parser = configargparse.ArgParser(default_config_files=['config/config.yaml'])

    # 添加参数
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    # # 训练参数
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    # # TransformerEncoder基础参数
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--input_dim', type=int, help='Input dimension')
    parser.add_argument('--model_dim', type=int, help='Model dimension')
    parser.add_argument('--num_heads', type=int, help='Number of heads')
    parser.add_argument('--num_layers', type=int, help='Number of layers')
    parser.add_argument('--n_segments', type=int, default=[], help='Number of segments')  # 多个Transformer的数量,单个Transformer时不需要设置
    parser.add_argument('--trans_output_dim', type=int, default=[], help='Transformer output dimension')  # Transformer的输出维度,只有在n_segments>1时才需要设置
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    # # 数据集参数
    parser.add_argument('--train_data_path', type=str, help='Path to the training data')
    parser.add_argument('--test_data_path', type=str, help='Path to the test data')
    parser.add_argument('--norm_dim', type=int, nargs='+', help='Dimensions to normalize the data')

    # 解析参数
    opts = parser.parse_args()

    # 打印参数
    print("Configuration:")
    print(f"Batch size: {opts.batch_size}")
    print(f"Learning rate: {opts.learning_rate}")
    print(f"Number of epochs: {opts.num_epochs}")
    print(f"Model name: {opts.model_name}")
    print(f"Input dimension: {opts.input_dim}")
    print(f"Model dimension: {opts.model_dim}")
    print(f"Number of heads: {opts.num_heads}")
    print(f"Number of layers: {opts.num_layers}")
    print(f"Transformer output dimension: {opts.trans_output_dim}")
    print(f"Number of segments: {opts.n_segments}")
    print(f"Number of classes: {opts.num_classes}")
    print(f"Normalization dimension: {opts.norm_dim}")
    print(f"Train file path: {opts.train_data_path}")
    print(f"Test file path: {opts.test_data_path}")

    # 加载数据
    # train_file_path = r'data\raw\Data_train.mat'
    # test_file_path = r'data\raw\Data_test.mat'
    train_loader, test_loader = get_dataloaders(opts.train_data_path, opts.test_data_path, opts.norm_dim, opts.batch_size)

    model = getmodel(opts)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate)

    model = model.to('cuda')
    start_time = time.time()
    train_model(model, train_loader, test_loader, criterion, optimizer, opts.num_epochs)
    end_time = time.time() - start_time
    print(f"Training time: {end_time:.2f} seconds")


if __name__ == "__main__":

    main()
