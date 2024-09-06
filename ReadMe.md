# 不同model的默认配置对应关系
注意配置文件中不要出现中文，包括注释，否则gbk和utf-8编码转换很麻烦

0、GRU
```sh
python train.py -c config/try_GRU.yaml
```

1、最基础的测试模型
TransformerEncoder+分类器，配置文件config_tran_default.yaml
```sh
python train.py -c config/config_tran_default.yaml --num_epochs 5
python train.py -c config/try_tran.yaml
```

2、多编码器融合 MultiTransformerEncoder
```sh
python train.py -c config/config_MultiTrans.yaml --num_epochs 5
python train.py -c config/config_MultiTrans1.yaml
python train.py -c config/config_try.yaml
```

3、测试
```sh
python main.py -c config/config1.yaml --num_epochs 5
```