# 测试读取配置文件
try:
    with open('config/config_tran_default.yaml', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print("读取成功，文件内容：")
    for line in lines:
        print(line)
except UnicodeDecodeError as e:
    print(f"读取错误：{e}")
