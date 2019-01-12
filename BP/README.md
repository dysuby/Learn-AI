## 文件结构

- `model/` 存放训练好模型的文件夹
- `predict` 存放预测结果
- `test/` 测试集
- `train/` 训练集
- `BP.py` BP 代码实现
- `utils.py` 一些工具函数

## 使用说明

```
D:\BP> py .\BP.py
---------------------- Usage ----------------------
<train>    --- begin to train
<continue> --- load model and continue training
<test>     --- load model, predict test data and save
```

- `train` 重新训练
- `continue` 加载已有的模型继续训练，用于调整参数
- `test` 使用测试集测试
