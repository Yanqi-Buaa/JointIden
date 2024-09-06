print("Script is being executed")


import configargparse


def main():
    print("Inside main()")
    # 创建解析器，并指定默认的配置文件路径
    print("Creating parser...")
    parser = configargparse.ArgParser(default_config_files=['config/config1.yaml'])

    # 添加命令行参数和配置文件参数
    print("Adding arguments...")
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    # parser.add_argument('-c', '--config', type=str, help='config file path')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--norm_dim', type=int, nargs='+', help='Dimensions to normalize the data')
    parser.add_argument('--train_data_path', type=str, help='Path to the training data')

    # 解析参数
    print("Parsing arguments...")
    args = parser.parse_args()
    print("Arguments parsed")

    # 检查解析后的参数值
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Dimensions to normalize: {args.norm_dim}")
    print(f"Training data path: {args.train_data_path}")


if __name__ == "__main__":
    print("Entering main()")
    main()
