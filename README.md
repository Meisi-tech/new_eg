# 基础机器学习 Demo（mock 数据）

一个零依赖外部数据的机器学习入门示例：用 **逻辑回归** 做二分类，
数据全部由 `numpy` 随机生成，可一键复现，适合初学者理解「数据 → 训练 → 评估」的完整流程。

## 场景

根据用户的 **年龄** 和 **年收入**，预测其是否会 **购买** 某产品（二分类：购买 / 不购买）。

## 流程

1. 生成 mock 数据（1000 条，含噪声）
2. 划分训练集 / 测试集（8:2，按标签分层抽样）
3. 特征标准化（StandardScaler）
4. 训练逻辑回归模型（LogisticRegression）
5. 在测试集上评估：准确率、混淆矩阵、分类报告
6. 画出决策边界并保存为图片

## 运行

```bash
# 安装依赖（建议使用虚拟环境）
pip install -r requirements.txt

# 运行
python ml_demo.py
```

运行后会在 `output/decision_boundary.png` 生成决策边界可视化图。

## 依赖

- numpy
- scikit-learn
- matplotlib

## 示例输出

```
测试集准确率 Accuracy: 0.95+

混淆矩阵:
[[正确不购买  误判购买]
 [误判不购买  正确购买]]

分类报告:
              precision  recall  f1-score   support
       不购买       0.95      0.96      0.95       ...
        购买        0.96      0.95      0.95       ...
```

> 数值会因随机种子固定而保持稳定。
