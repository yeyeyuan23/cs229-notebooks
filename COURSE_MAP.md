# 讲义第一章 → 当前代码练习

依据 **CS229 Lecture Notes，Tengyu Ma / Andrew Ng，2026-08-23**。以章节内容为准，不依赖视频编号。下面是讲义印刷页码，PDF 阅读器页码需加 1。

| 讲义内容 | 页码 | 第一章 notebook 中练什么 |
|---|---:|---|
| 房价例子、模型与输入表示 | 9–10 | 数组创建、切片、行列、截距列、循环预测与矩阵预测 |
| 1.1 LMS algorithm | 10–14 | 损失与梯度、数值梯度检查、批量 GD、逐样本 SGD、学习率和训练实验 |
| 1.2 The normal equations | 14–16 | 梯度为零到线性方程、`np.linalg.solve`、同数据对照 GD 与 NE |
| §1.1–1.2 的框架对照（自学延伸） | 10–16 | PyTorch 张量、自动求导、优化器与完整训练；同设置对照 NumPy 梯度、参数、误差曲线和验证 MSE |
| 1.3 Probabilistic interpretation | 16–18 | 独立高斯误差、对数似然、用候选参数验证与平方误差的排序关系 |
| 1.4 Locally weighted linear regression | 18–20 | 多项式特征、局部权重、加权正规方程、degree/tau 的验证集比较 |

入口：[第一章 notebook](chapters/01_linear_regression/linear_regression.ipynb)。

练习用 `m` 表示样本数、`p` 表示含截距的设计矩阵列数；讲义的输入特征数 `d=p-1`。讲义的平方误差目标用求和，本练习沿用平均，梯度相差 `m` 倍，比较代码或学习率时注意这个约定。

本仓库当前提供第一章。讲义第二章是 **Classification and logistic regression**，与本章的 linear regression 区分。
