import os


# 创建必要的目录
directories = ["data/raw", "data/processed", "data/external",
               "models/checkpoints", "models/final",
               "notebooks", "scripts", "config", "logs", "reports/figures"]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
