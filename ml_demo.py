"""
基础机器学习示例：用 mock 数据演示「逻辑回归」二分类
场景：根据用户的「年龄」和「年收入」预测其是否会购买某产品

全程不依赖任何外部数据文件，数据由 numpy 随机生成，可一键复现。
运行：python ml_demo.py
"""

import os

import numpy as np
import matplotlib

# 无图形界面环境下使用 Agg 后端，避免报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置中文字体，避免图中中文显示为方块
matplotlib.rcParams["font.family"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans SC",
    "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_SEED = 42
N_SAMPLES = 1000


def make_mock_data(n=N_SAMPLES, seed=RANDOM_SEED):
    """生成 mock 数据：两个特征（年龄、年收入），一个二分类标签。"""
    rng = np.random.default_rng(seed)

    # 特征 1：年龄，范围 18~65
    age = rng.uniform(18, 65, size=n)
    # 特征 2：年收入（单位：万元），范围 3~60
    income = rng.uniform(3, 60, size=n)

    # 构造一个“隐藏规则”作为标签：
    # 年纪适中且收入较高的人更可能购买 -> 用一条线性边界 + 噪声生成标签
    boundary = 0.18 * age + 0.9 * income - 35.0
    noise = rng.normal(0, 3.0, size=n)
    # 当 boundary + noise > 0 时购买（标签=1），否则不购买（标签=0）
    y = (boundary + noise > 0).astype(int)

    X = np.column_stack([age, income])
    return X, y


def main():
    print("=" * 50)
    print("基础机器学习 demo：逻辑回归二分类（mock 数据）")
    print("=" * 50)

    # 1) 生成数据
    X, y = make_mock_data()
    print(f"生成样本数: {len(y)}，正样本(购买)占比: {y.mean():.1%}")

    # 2) 划分训练集 / 测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"训练集: {len(X_train)} 条，测试集: {len(X_test)} 条")

    # 3) 特征标准化（让逻辑回归收敛更快、更稳定）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4) 训练模型
    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    model.fit(X_train_s, y_train)

    # 5) 在测试集上预测并评估
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率 Accuracy: {acc:.3f}")
    print("\n混淆矩阵 (行=真实, 列=预测):")
    print(confusion_matrix(y_test, y_pred))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=["不购买", "购买"]))

    # 6) 保存决策边界可视化
    os.makedirs("output", exist_ok=True)
    plot_path = os.path.join("output", "decision_boundary.png")
    plot_decision_boundary(scaler, model, X, y, plot_path)
    print(f"\n决策边界图已保存到: {plot_path}")

    print("\n✅ 跑通完成！")


def plot_decision_boundary(scaler, model, X, y, save_path):
    """画出两类样本散点 + 模型学到的决策边界。"""
    # 在标准化空间里构造网格，再映射回原始特征空间计算边界
    age_min, age_max = X[:, 0].min(), X[:, 0].max()
    inc_min, inc_max = X[:, 1].min(), X[:, 1].max()

    ax, ay = np.meshgrid(
        np.linspace(age_min, age_max, 200),
        np.linspace(inc_min, inc_max, 200),
    )
    grid = np.column_stack([ax.ravel(), ay.ravel()])
    grid_s = scaler.transform(grid)
    proba = model.predict_proba(grid_s)[:, 1].reshape(ax.shape)

    plt.figure(figsize=(7, 5))
    # 背景着色：购买概率
    contour = plt.contourf(ax, ay, proba, levels=20, cmap="RdBu_r", alpha=0.6)
    plt.colorbar(contour, label="购买概率")
    # 样本散点
    plt.scatter(
        X[y == 0, 0], X[y == 0, 1], c="tab:blue", s=12, label="不购买", alpha=0.7
    )
    plt.scatter(
        X[y == 1, 0], X[y == 1, 1], c="tab:red", s=12, label="购买", alpha=0.7
    )
    plt.xlabel("年龄")
    plt.ylabel("年收入 (万元)")
    plt.title("逻辑回归决策边界（mock 数据）")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
