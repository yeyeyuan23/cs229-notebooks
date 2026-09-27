"""Pair short teaching steps with code, preserving learner implementations.

build_problem_sets owns algorithm scaffolds and data. This module owns the
reading order and explanations. apply_layout accepts a starter or a learner
notebook; function source and written answers are carried across unchanged.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from textwrap import dedent

import nbformat as nbf


PURPOSES = {
    ("ps0", 1): "用给定的截距和斜率，一次算出 5 个样本的预测值。理解矩阵的每行是一个样本；本题不训练参数。",
    ("ps0", 2): "推导并实现平方误差的梯度：它告诉我们每个参数变化时，损失如何变化。先推公式，再把公式翻译成 NumPy。",
    ("ps0", 3): "检查上一题的梯度有没有写对。用两种方法算同一个位置的梯度，看结果是否接近；本题代码已经提供。",
    ("ps1", 1): "从训练散点中找一条最合适的直线：求截距和斜率，再用验证集检查预测误差。你需要实现求参数和算 MSE 两个函数。",
    ("ps1", 2): "用很多次小步更新，再求一遍 Problem 1 的最佳直线，检查结果是否与正规方程接近。请先运行 P1 的完整实验，得到 theta_ne。",
    ("ps1", 3): "让模型能拟合弯曲的数据。你只需要实现 polynomial_features；后面会先用三次模型逐步演示，再比较 1、3、10 次模型。先运行 P1，复用求参数和算 MSE 的函数。",
    ("ps1", 4): "预测一个位置时，让附近的训练样本更有影响。为每个查询位置拟合一条局部直线，比较不同邻域宽度 tau 的预测效果。复用 P1 的 add_intercept 和 mse。",
    ("ps2", 1): "从训练数据学习一个二分类模型：输入特征，输出属于类别 1 的概率。先写概率转换，再写损失，最后写训练循环。",
    ("ps2", 2): "检验一个看似准确的模型有没有漏掉少数类。实现指标后，在同一验证集比较全猜 0 的模型与 P1 模型；先运行 P1 的完整实验。",
    ("ps2", 3): "让训练过程更重视少数类，观察哪些预测变好了、哪些变差了。复用前两题的函数；验证数据保持不变。",
    ("ps2", 4): "让两层神经网络学会 XOR：两个输入不同则输出 1，相同则输出 0。分别实现前向预测和反向求梯度，再用给定循环训练。这里只检查四个训练点，不评价泛化。",
    ("ps3", 1): "把图片的许多颜色替换成 4 种代表颜色。先实现分组、更新中心，再把两步循环起来，最后还原成图片。",
    ("ps3", 2): "把二维点压缩成一个数，再从这个数重建二维位置。观察保留一个主要方向后，哪些信息丢失了。",
    ("ps3", 3): "选做：假设一批数来自两个高斯分布，在不知道各样本来源的情况下估计两个分布。先估计归属概率，再用概率更新参数，交替执行。",
}


# These descriptions sit immediately above the named function's code cell.
FUNCTION_GUIDES = {
    "linear_predictions": r"""
        **要写什么：** `linear_predictions(X, theta)` 把已有参数用于预测。
        `X` 为 `(m,d)`，`theta` 为 `(d,)`，返回每个样本的预测组成的 `(m,)` 数组。
        本题 `theta=[1,2]`，第一行 `X[0]=[1,-2]`，所以第一个预测是 `1+2*(-2)=-3`。
        用一次矩阵乘法完成，不写循环。下格只定义函数，完整实验格才会调用它。
    """,
    "squared_error_gradient": r"""
        **要写什么：** `squared_error_gradient(X, y, theta)` 返回每个参数的偏导数。
        `X:(m,d)`、`y:(m,)`、`theta:(d,)`，返回梯度 `(d,)`，不更新参数。

        $$J(\theta)=\frac{1}{2m}\sum_i(\hat y_i-y_i)^2,\qquad \hat y=X\theta.$$

        在上方答题框先对单个参数 $\theta_j$ 求导，再组合成向量。
        `X @ theta - y` 是逐样本误差 `(m,)`；不要先把它平均成一个 loss 再做矩阵乘法。
        完成后去 P3 用数值梯度检查。
    """,
    "squared_error_loss": r"""
        **已给出：** `squared_error_loss(X, y, theta)` 在一组参数上计算一个损失值。
        它等于 MSE 的一半，与 P2 推导的 $J$ 一致。接下来的数值梯度会反复调用它。
        输入 `X:(m,d)`、`y:(m,)`、`theta:(d,)`；返回一个数。
    """,
    "numerical_gradient": r"""
        **已给出：** `numerical_gradient(fn, theta)` 每次只扰动一个参数。
        `fn` 是“输入参数，返回一个损失”的函数；`epsilon` 是很小的移动距离。

        $$\frac{\partial J}{\partial\theta_j}\approx
        \frac{J(\theta+\epsilon e_j)-J(\theta-\epsilon e_j)}{2\epsilon}.$$

        下面代码返回与 `theta` 同 shape 的数组。读懂即可，不需要改写。
    """,
    "add_intercept": r"""
        **已给出：** `add_intercept(x)` 为每个输入加一个常数 1，使截距也能参与矩阵乘法。
        例如输入 `[2,3]`，返回 `[[1,2],[1,3]]`。
        输入 `(m,)`，输出 `X:(m,2)`。每行的预测是 $\theta_0+\theta_1x_i$。
        运行下格定义函数；不需要填写 TODO。
    """,
    "fit_normal_equation": r"""
        **要写什么：** `fit_normal_equation(X, y)` 找到使训练平方误差最小的参数。
        输入 `X:(m,d)`、`y:(m,)`；返回 `theta:(d,)`。

        用上方推导得到的方程构造 `A` 和 `b`，再调用 `np.linalg.solve(A,b)`。
        `solve` 的含义是求解 `A @ theta = b`，不要显式计算矩阵逆。
        本题 `d=2`，但不要写死参数数量，P3 会用到更多列。
        小例子：`X=[[1,0],[1,1]]`、`y=[1,3]`，应得到截距 1、斜率 2。
    """,
    "mse": r"""
        **要写什么：** `mse(y_true, y_pred)` 衡量预测平均错了多少。
        两个输入都是 `(m,)`，返回一个数：先逐项相减，平方，再取平均。

        $$\mathrm{MSE}=\frac1m\sum_i(y_i-\hat y_i)^2.$$

        例如真实值 `[1,3]`、预测 `[2,3]`，平方误差为 `[1,0]`，MSE 应为 `0.5`。
        它是损失 $J$ 的两倍；两者的最小值出现在同一组参数处。
    """,
    "fit_gradient_descent": r"""
        **要写什么：** `fit_gradient_descent` 从全零参数开始，反复计算梯度并更新。
        `X:(m,d)`、`y:(m,)`；返回最终参数 `(d,)` 与记录 MSE 的一维 `history`。
        `learning_rate` 是步长，`steps` 是参数更新次数。

        循环里按顺序处理：预测 `(m,)` → 误差 `(m,)` → 梯度 `(d,)` → 更新参数。
        `history` 只存一个个 MSE 数字，不参与求梯度。具体更新式写在上方答题框。

        本题按已有约定，每 10 次更新记录一次**更新前**的预测 MSE。
        `step % 10 == 0` 对应 0、10、20……，不是只更新这些步。
        1000 次更新会留下 100 个记录点。保留相同约定，后面画图的横轴才一致。
    """,
    "polynomial_features": r"""
        **要写什么：** `polynomial_features(x, degree)` 将每个数展开成一行多项式特征。
        `degree` 是最高次数，例如 `degree=3` 表示计算到三次方。

        输入 `x=np.array([2.,3.])`、`degree=3`，应输出：

        ```text
        [[1., 2., 4.,  8.],
         [1., 3., 9., 27.]]
        ```

        每行依次是 `[1,x,x²,x³]`。第一列是全 1，表示截距。
        输入 shape 为 `(m,)`，输出为 `(m,degree+1)`；输出矩阵通常记为 `Phi`，读作 phi。
        **这一步只转换输入，不使用 y，也不求 theta。** 不需要再调用 `add_intercept`。

        实现提示：依次算 `x**0` 到 `x**degree`，把得到的各列用 `np.column_stack` 并排放好。
        下格保留 TODO，填写你自己的实现，再运行这个函数定义格。
    """,
    "predict_lwr": r"""
        **要写什么：** `predict_lwr(x_train,y_train,x_query,tau)` 为每个查询位置预测一个值。
        训练输入、目标都为 `(m,)`，查询位置为 `(n,)`，返回预测 `(n,)`。
        `tau>0` 控制邻域宽度：较小的 tau 让远处训练点的影响更小。

        对 `x_query` 中的每个 `q`，按下面四步组织函数：

        1. 用 P1 的 `add_intercept(x_train)` 构造 `X:(m,2)`。
        2. 计算权重 $w_i(q)=\exp(-(x_i-q)^2/(2\tau^2))$，将它们放进对角矩阵 `W`。
        3. 从 $J_q(\theta)=\frac12\sum_i w_i(q)(X_i\theta-y_i)^2$ 推导加权正规方程，
           用 `np.linalg.solve` 求这一个 `q` 的局部参数 `(2,)`。
        4. 用 `[1,q]` 与局部参数算一个预测，按顺序存入输出；换下一个 `q` 重新算权重和参数。

        例如传入 `x_query=[0.,1.]` 时应返回两个预测值，而非两组参数。
        查询位置不需要真实标签；所有拟合都只使用训练集。
    """,
    "sigmoid": r"""
        **要写什么：** `sigmoid(z)` 将模型分数变成 0 到 1 的概率，输入输出 shape 相同。

        $$\sigma(z)=\frac{1}{1+e^{-z}}.$$

        `z=0` 时结果为 `0.5`。请处理 `[-1000.,0.,1000.]` 这样的极端输入，
        输出应接近 `[0.,0.5,1.]` 且没有 NaN；可按正负数分支避免指数溢出。
    """,
    "binary_cross_entropy": r"""
        **要写什么：** `binary_cross_entropy(y, probability, sample_weight=None)` 衡量概率预测好坏。
        `y` 和 `probability` 都为 `(m,)`，返回一个损失值。这里使用自然对数。

        $$L_w=-\frac{\sum_i w_i[y_i\log p_i+(1-y_i)\log(1-p_i)]}{\sum_i w_i}.$$

        `None` 表示所有权重为 1，即普通平均损失。传入的权重为非负 `(m,)`，总和必须大于 0。
        取对数前将概率裁剪到 `[1e-12,1-1e-12]`。例如真实标签 1 的预测概率为 0.9，
        比预测概率 0.1 的损失小。P3 会复用这个函数来改变样本权重。
    """,
    "fit_logistic_regression": r"""
        **要写什么：** `fit_logistic_regression` 将刚才两个函数串成训练循环。
        `X:(m,d)` 已有截距列，`y:(m,)` 是 0/1 标签。返回参数 `(d,)` 和长度为 `steps` 的 loss 数组。

        从零参数开始，每轮算 `X @ theta` → 用 `sigmoid` 转成概率 → 记录更新前交叉熵
        → 按上方推导的梯度更新参数。`sample_weight=None` 等于全 1 权重；
        有权重时梯度也要加权，且除以权重总和，不能只改 loss。
        先验证全 1 权重与 None 等价。概率转类别的阈值 0.5 留到后面的完整实验使用。
    """,
    "classification_metrics": r"""
        **要写什么：** `classification_metrics(y_true,y_pred)` 接收两个 `(m,)` 的 0/1 数组，
        返回下面这些键组成的字典。`y_pred` 是类别，不是概率；约定 1 为正类。

        | 键 | 如何计算 |
        |---|---|
        | `TP` / `TN` | 真实、预测同为 1 / 同为 0 的数量 |
        | `FP` / `FN` | 真实 0 预测 1 / 真实 1 预测 0 的数量 |
        | `accuracy` | `(TP+TN)/m` |
        | `precision` | `TP/(TP+FP)`，预测为正的样本里多少是真的 |
        | `recall` | `TP/(TP+FN)`，真实正类里找到了多少 |
        | `specificity` | `TN/(TN+FP)`，真实负类里判对了多少 |
        | `balanced_accuracy` | `(recall+specificity)/2` |
        | `f1` | `2*precision*recall/(precision+recall)` |

        分母为 0 时本练习约定该指标返回 `0.0`。
        小例子：真实 `[0,0,1,1]`、预测 `[0,1,0,1]`，四种计数各为 1。
    """,
    "init_mlp": r"""
        **已给出：** `init_mlp(rng)` 创建网络的初始参数字典，运行即可。
        `W1:(2,4)` 把两个输入送到四个隐藏单元；`W2:(4,1)` 再合成一个输出。
        `b1:(4,)` 和 `b2:(1,)` 是偏置，不需要给 XOR 输入加截距列。
    """,
    "mlp_forward": r"""
        **要写什么：** `mlp_forward(X,params)` 用当前网络参数计算概率。

        $$Z_1=XW_1+b_1,\quad H=\tanh(Z_1),\quad Z_2=HW_2+b_2,\quad P=\sigma(Z_2).$$

        输入 `X:(m,2)` 和参数字典，返回 `(probability,cache)`。
        `probability` 必须是 `(m,1)`，`cache` 用字典保存 `H` 和 `probability`，
        供下一步反向传播复用。先在注释里写每个中间量的 shape，再写运算。
    """,
    "mlp_backward": r"""
        **要写什么：** `mlp_backward(X,y,params,cache)` 计算平均交叉熵对每个网络参数的梯度。
        它返回含 `W1,b1,W2,b2` 的字典，各梯度 shape 与对应参数相同；这里不更新参数。

        先将 `y:(m,)` 变为 `(m,1)` 再与概率相减，否则会广播出错误的 `(m,m)`。
        从输出往隐藏层使用链式法则；`tanh` 的导数可写为 `1-H**2`。
        按样本数归一化，对偏置沿样本轴求和。在代码注释中记录各中间量 shape。
    """,
    "assign_clusters": r"""
        **要写什么：** `assign_clusters(X,centroids)` 为每个样本选最近的中心。
        `X:(m,d)`、中心 `centroids:(k,d)`，返回整数编号数组 `assignment:(m,)`。
        算平方欧氏距离，取距离最小的中心编号；相同时取编号较小者。
        例如一维样本 `[[0.],[9.]]`、中心 `[[1.],[8.]]`，应返回 `[0,1]`。
    """,
    "update_centroids": r"""
        **要写什么：** `update_centroids(X,assignment,k,old_centroids)` 根据分组重算中心。
        对每个簇内的样本逐列取平均，返回 `(k,d)`。例如一簇中两个点为 `[0,2]` 与 `[2,4]`，
        新中心是 `[1,3]`。如果某簇没有点，保留它的旧中心，避免空数组均值产生 NaN。
    """,
    "fit_kmeans": r"""
        **要写什么：** `fit_kmeans(X,k,rng,max_steps=50)` 将刚才两步循环起来。
        先用传入的 `rng` 从 `X` 不放回抽取 `k` 行作为中心，分配样本并记录初始损失。
        每轮更新中心、重新分配、再记一次损失，最多运行 `max_steps` 轮。

        $$J=\frac1m\sum_i\lVert X_i-c_{a_i}\rVert^2.$$

        `a_i` 是样本的簇编号，`c` 是中心。返回 `(centroids,assignment,history)`，
        shape 依次是 `(k,d)`、`(m,)`、一维数组。最终编号应对应最终中心。
    """,
    "fit_pca": r"""
        **要写什么：** `fit_pca(X,n_components=1)` 找出数据变化最大的方向。
        输入 `X:(m,d)`，返回逐列均值 `mean:(d,)` 与主方向 `components:(d,k)`，其中 `k=n_components`。
        先减均值，本题按 $C=X_c^TX_c/m$ 定义协方差，用 `np.linalg.eigh` 求特征值和特征向量，
        按特征值从大到小取前 k 列。本题是 `(180,2)` → 均值 `(2,)` 与方向 `(2,1)`。
        特征向量的正负都可以，不要把符号不同误判成错误。
    """,
    "pca_transform": r"""
        **要写什么：** `pca_transform(X,mean,components)` 将数据投影到刚才学到的方向。
        使用训练时的均值中心化后，再投影。返回 `Z:(m,k)`；本题每个二维点变为一个坐标，得到 `(180,1)`。
        此函数不重新拟合方向。请在注释里解释保留了哪个方向的信息。
    """,
    "pca_inverse_transform": r"""
        **要写什么：** `pca_inverse_transform(Z,mean,components)` 把低维坐标放回原空间，
        再加回均值。输入 `Z:(m,k)`，返回 `(m,d)`；本题恢复为 `(180,2)`。
        丢掉的方向无法恢复，因此“重建”通常不等于原始数据。
    """,
    "gmm_e_step": r"""
        **要写什么：** `gmm_e_step(x,weights,means,variances)` 估计每个观测来自两个分布的概率。
        `x:(m,)`；另外三个数组均为 `(2,)`，分别是比例、均值、方差（不是标准差）。

        $$f_k(x_i)=\frac{\exp(-(x_i-\mu_k)^2/(2\sigma_k^2))}{\sqrt{2\pi\sigma_k^2}},
        \qquad r_{ik}=\frac{\pi_k f_k(x_i)}{\sum_j\pi_j f_j(x_i)}.$$

        返回 `responsibilities:(m,2)`；例如一行 `[0.3,0.7]` 表示两个归属概率，行和必须为 1。
        若归一化分母为 0，要检查实现并报错，不要任意修改概率后继续。
    """,
    "gmm_m_step": r"""
        **要写什么：** `gmm_m_step(x,responsibilities)` 用刚才的归属概率更新分布参数。
        令 $N_k=\sum_i r_{ik}$。新比例是 $N_k/m$，新均值是以 $r_{ik}$ 加权的 `x` 平均值，
        新方差是相对**本轮新均值**的加权平方偏差平均值；后两者的分母都是 $N_k$。
        方差设下限 `1e-8`；若 $N_k=0$ 则检查实现并报错。
        返回 `(weights,means,variances)`，每个数组为 `(2,)`。比例和为 1，方差为正。
    """,
}


FLOWS = {
    ("ps0", 1): "用给定的 X 和 theta 调用预测函数，再对照手算结果。填完函数后把下格 RUN_P1_CHECKS 改为 True。",
    ("ps0", 3): "analytic 调用你写的解析梯度，numeric 用扰动法独立估计；两者都应为 (2,)，误差在容差内。",
    ("ps1", 1): "训练 x → 加截距列 → 拟合 theta_ne → 用同一参数预测验证集 → 算 MSE。baseline 对所有验证样本都预测训练 y 的均值。",
    ("ps1", 2): "调用梯度下降 → 与 P1 的 theta_ne 比较 → 画训练 MSE。每 10 步记录一次，所以记录下标 0、1、2 对应 step 0、10、20；横轴乘 10 是为恢复真实迭代编号。MSE 下降趋稳即可，不要求降到 0。",
    ("ps1", 4): "分别用 tau=0.1、0.5、2.0 预测验证集并比较 MSE，再预测密集 grid 画曲线。grid 只是画图用的位置，不参与训练。最后在答题框写加权方程和你的选择。",
    ("ps2", 1): "先测概率和损失的边界情况，再训练 theta_log；用验证输入算 probability，最后以 0.5 为阈值得到 prediction。这两个变量会供下一题使用。",
    ("ps2", 2): "先用四个样本核对计数，再分别计算全零预测与 P1 prediction 的指标。引用实际 accuracy、recall、balanced_accuracy 解释差别。",
    ("ps2", 3): "用训练标签统计 n0、n1 → 给正类权重 n0/n1、负类权重 1 → 用加权函数从零训练 → 在原验证集预测 → 按同一 0.5 阈值比较指标。",
    ("ps2", 4): "初始化参数 → 前向得到概率和 cache → 反向得到各梯度 → 更新各参数，重复 5000 次。最后检查四个 XOR 标签；这是训练拟合检查。",
    ("ps3", 1): "image 展平为 pixels:(2304,3) → K-means 找 4 个颜色中心 → 用各自中心替换像素 → reshape 成原图大小。这里只减少颜色种类，不代表 NPY 文件一定更小。",
    ("ps3", 2): "拟合均值和主方向 → 投影 Z → 重建二维点 → 比较重建 MSE 和全部预测成均值的基线 → 画原点与重建点。",
    ("ps3", 3): "初始化两个分布 → E-step 求概率 → M-step 更新参数，重复 30 轮。监测 helper 只用于记录 log-likelihood，数值容差内应不下降；E/M 更新仍由你实现。",
}


LOCAL_CHECKS = {
    "linear_predictions": 'assert np.allclose(linear_predictions(np.array([[1., 2.], [1., 3.]]), np.array([1., 2.])), [5., 7.])',
    "fit_normal_equation": 'assert np.allclose(fit_normal_equation(np.array([[1., 0.], [1., 1.]]), np.array([1., 3.])), [1., 2.])',
    "mse": 'assert np.isclose(mse(np.array([1., 3.]), np.array([2., 3.])), 0.5)',
    "sigmoid": 'assert np.allclose(sigmoid(np.array([-1000., 0., 1000.])), [0., 0.5, 1.])',
    "binary_cross_entropy": 'assert binary_cross_entropy(np.array([1.]), np.array([0.9])) < binary_cross_entropy(np.array([1.]), np.array([0.1]))',
    "classification_metrics": 'example = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))\nassert all(example[key] == 1 for key in ("TP", "TN", "FP", "FN"))',
    "assign_clusters": 'assert np.array_equal(assign_clusters(np.array([[0.], [9.]]), np.array([[1.], [8.]])), [0, 1])',
    "update_centroids": 'example = update_centroids(np.array([[0., 2.], [2., 4.]]), np.array([0, 0]), 1, np.array([[0., 0.]]))\nassert np.allclose(example, [[1., 3.]])',
}


def markdown(source, cell_id):
    return nbf.v4.new_markdown_cell(dedent(source).strip(), id=cell_id)


def coding(source, cell_id):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n", id=cell_id)


def split_definitions(cell):
    """Keep exact function text; split only between top-level statements."""
    tree = ast.parse(cell.source)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return [(None, deepcopy(cell))]
    if len(functions) == 1 and len(tree.body) == 1:
        return [(functions[0].name, deepcopy(cell))]
    lines = cell.source.splitlines(keepends=True)
    chunks = []
    cursor = 0
    for node in functions:
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        if ''.join(lines[cursor:start]).strip():
            chunks.append((None, ''.join(lines[cursor:start])))
        chunks.append((node.name, ''.join(lines[start:node.end_lineno])))
        cursor = node.end_lineno
    if ''.join(lines[cursor:]).strip():
        chunks.append((None, ''.join(lines[cursor:])))
    return [(name, coding(source, f'{cell.id}-{i}')) for i, (name, source) in enumerate(chunks)]


def local_check(name, prefix):
    flag = f'TRY_{name.upper()}'
    body = '\n'.join('    ' + line for line in LOCAL_CHECKS[name].splitlines())
    return [
        markdown(f'**只检查刚才这个函数：** 填完并运行函数定义后，把下格 `{flag}` 改为 `True`。\n'
                 '先用这个小例子核对输入输出，再继续下一步。', prefix + '-check-guide'),
        coding(f'{flag} = False\nif {flag}:\n{body}\n    print("{name}: example passed")', prefix + '-check'),
    ]


def polynomial_walkthrough(old_check):
    """Split P3's data flow into runnable units, retaining the final experiment."""
    tree = ast.parse(old_check.source)
    flag = tree.body[0]
    if not (isinstance(flag, ast.Assign) and isinstance(flag.targets[0], ast.Name)
            and flag.targets[0].id == 'RUN_P3_CHECKS'):
        raise ValueError('Unexpected P3 check layout; refusing to replace learner code')
    flag_source = ast.get_source_segment(old_check.source, flag)
    # Keep any learner edits to the final experiment; move only its flag definition.
    final_source = ''.join(old_check.source.splitlines(keepends=True)[flag.end_lineno:])
    steps = [
        ("2. 用两个数检查函数", "这一格只调用刚才的函数，不训练模型。填写函数后将 `RUN_P3_CHECKS` 改为 `True`；\n"
         "本题后续各格都会使用这个开关。若小例子失败，先修函数，再往下运行。",
         flag_source + '\nif RUN_P3_CHECKS:\n    example = polynomial_features(np.array([2., 3.]), 3)\n'
         '    print(example)\n    assert example.shape == (2, 4)\n'
         '    assert np.allclose(example, [[1., 2., 4., 8.], [1., 3., 9., 27.]])'),
        ("3. 转换训练集和验证集", r"""
            先固定 `degree=3`，只做一个模型。`Phi_train` 就是转换后的训练输入，
            `Phi_valid` 是转换后的验证输入；Phi 只是变量名里用来表示特征矩阵的词。

            `x_train:(60,)` → `Phi_train:(60,4)`，`x_valid:(40,)` → `Phi_valid:(40,4)`。
            每一行都是 `[1,x,x²,x³]`，`y_train`、`y_valid` 保持不变。
            下格已经写好，运行后看看前三行和 shape。
        """, """
            if RUN_P3_CHECKS:
                degree = 3
                Phi_train = polynomial_features(x_train, degree)
                Phi_valid = polynomial_features(x_valid, degree)
                print("Phi_train:", Phi_train.shape, "Phi_valid:", Phi_valid.shape)
                print("First three training rows:")
                print(Phi_train[:3])
                assert Phi_train.shape == (len(x_train), degree + 1)
                assert Phi_valid.shape == (len(x_valid), degree + 1)
        """),
        ("4. 用训练数据拟合参数", r"""
            复用 P1 写过的 `fit_normal_equation`，这次输入的矩阵有 4 列。
            输出 `theta_poly:(4,)` 分别对应常数项、x、x²、x³ 的系数。

            $$\hat y=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3.$$

            只有这一步在求参数，输入必须使用训练集。验证集随后只用来预测和评价。
        """, """
            if RUN_P3_CHECKS:
                theta_poly = fit_normal_equation(Phi_train, y_train)
                print("theta for degree=3:", theta_poly)
                assert theta_poly.shape == (degree + 1,)
        """),
        ("5. 用同一组参数做预测", "`Phi_train @ theta_poly` 给出训练集的 60 个预测；\n"
         "`Phi_valid @ theta_poly` 给出验证集的 40 个预测。每一行特征和参数相乘，就得到一个预测值。",
         """
            if RUN_P3_CHECKS:
                train_prediction = Phi_train @ theta_poly
                valid_prediction = Phi_valid @ theta_poly
                print("prediction shapes:", train_prediction.shape, valid_prediction.shape)
                print("First five valid predictions:", valid_prediction[:5])
        """),
        ("6. 计算两组预测误差", "复用 P1 的 `mse`。训练 MSE 衡量已用于拟合的数据，验证 MSE 衡量未用于拟合的数据。\n"
         "每次调用返回一个数；真实值与预测值必须来自同一批样本。",
         """
            if RUN_P3_CHECKS:
                train_mse = mse(y_train, train_prediction)
                valid_mse = mse(y_valid, valid_prediction)
                print("degree=3 train MSE:", train_mse)
                print("degree=3 valid MSE:", valid_mse)
        """),
        ("7. 画出这一个模型的曲线", "`grid` 是训练输入范围内的 400 个密集横坐标，用来把曲线画平滑。\n"
         "先用同一个 degree 转换 grid，再用刚才训练的 theta_poly 预测。这里没有重新训练。\n"
         "图的横轴是原始 x，纵轴是 y 或预测 y；这与 P2 的训练误差曲线不同。",
         """
            if RUN_P3_CHECKS:
                grid = np.linspace(x_train.min(), x_train.max(), 400)
                Phi_grid = polynomial_features(grid, degree)
                grid_prediction = Phi_grid @ theta_poly
                plt.figure(figsize=(9, 4))
                plt.scatter(x_train, y_train, s=15, color="black", alpha=0.5, label="train data")
                plt.plot(grid, grid_prediction, label="degree 3")
                plt.xlabel("x")
                plt.ylabel("y / predicted y")
                plt.legend()
                plt.title("One polynomial model: degree 3")
                plt.show()
        """),
        ("8. 串起来：比较三个模型", """
            下面把刚才的“转换 → 拟合 → 预测 → 评价 → 画图”放进一个循环，分别用 degree=1、3、10。
            每个 degree 单独训练一组参数：1 次是直线，3 次有 4 个系数，10 次有 11 个系数。
            三条线是三个模型的预测，不是分别画 x、x³、x¹⁰。

            保留三行 MSE 输出和三条曲线，在下方答题框选出本次验证 MSE 最低的 degree。
            训练、验证误差都高且错过明显趋势，支持欠拟合判断；训练更好、验证反而变差，
            才支持过拟合判断。不要预先认定 10 次一定过拟合。
            高次多项式也可能有数值问题，求解报错不直接等于过拟合。
        """, final_source),
    ]
    cells = []
    for index, (title, description, source) in enumerate(steps, 2):
        prefix = f'ps1-p3-step{index}'
        cells.extend([markdown(f'### {title}\n\n' + dedent(description).strip(), prefix + '-guide'),
                      coding(source, prefix + '-code')])
    return cells


def apply_layout(notebook, slug):
    """Pure transformation: do not touch disk or evaluate learner code."""
    if notebook.metadata.get('cs229', {}).get('teaching_layout') == 'paired-steps-v1':
        return deepcopy(notebook)
    result = deepcopy(notebook)
    cells = []
    problem = None
    step = 0
    for original in notebook.cells:
        cell = deepcopy(original)
        if cell.cell_type == 'markdown' and cell.source.startswith('## Problem '):
            title = cell.source.splitlines()[0]
            problem = int(title.split()[2])
            step = 0
            cell.source = title + '\n\n**本题目的：** ' + PURPOSES[(slug, problem)]
            if (slug, problem) == ('ps1', 1):
                cell.source += r'\n\n先在下方答题框从 $J(\theta)=\frac{1}{2m}\|X\theta-y\|^2$ 令梯度为零，推导关于 theta 的方程。'.replace(r'\n', '\n')
            if (slug, problem) == ('ps2', 1):
                cell.source += '\n\n先在答题框推导平均二元交叉熵的梯度。公式在下方损失函数旁，推导时先取所有样本权重为 1。'
            cells.append(cell)
            continue
        if cell.cell_type == 'markdown' and cell.source.startswith('## Checks'):
            problem = None
        if problem is None or cell.cell_type != 'code':
            if cell.cell_type == 'code' and cells and cells[-1].cell_type == 'code':
                cells.append(markdown('### 加载并查看本题集的数据\n\n上格准备了库和数据目录。下格读取数据、打印 shape，并展示实验输入；按顺序运行即可。', f'{slug}-load-data-guide'))
            cells.append(cell)
            continue
        if slug == 'ps1' and problem == 3 and 'RUN_P3_CHECKS' in cell.source:
            cells.extend(polynomial_walkthrough(cell))
            continue
        for name, chunk in split_definitions(cell):
            step += 1
            prefix = f'{slug}-p{problem}-step{step}'
            if name in FUNCTION_GUIDES:
                cells.append(markdown(f'### {step}. {name}\n\n' + dedent(FUNCTION_GUIDES[name]).strip(), prefix + '-guide'))
                cells.append(chunk)
                if name in LOCAL_CHECKS:
                    cells.extend(local_check(name, prefix))
            elif 'RUN_P' in chunk.source:
                body = FLOWS[(slug, problem)]
                if (slug, problem) == ('ps2', 3):
                    body += '\n\n下格仍需要你填写实验代码：保持与普通模型相同的学习率和更新次数。'
                    body += '\n比较 recall、specificity、balanced_accuracy、precision，在代码注释里记录哪些错误变多或变少。'
                cells.append(markdown(f'### {step}. 串起来：完整实验\n\n{body}\n\n完成相关函数后，打开下格的 `RUN_P{problem}_CHECKS` 并运行。', prefix + '-guide'))
                cells.append(chunk)
            else:
                explanation = '先运行这格准备本题数据，后面的函数将使用它。'
                if slug == 'ps2' and problem == 4:
                    explanation += '每行是 XOR 的两个输入，`y_xor` 是相应的 0/1 标签；输入不加截距列。'
                if slug == 'ps3' and problem == 3:
                    explanation += '`x_gmm` 是 200 个一维观测，学习时不提供它们属于哪一组。'
                cells.append(markdown(f'### {step}. 准备数据\n\n{explanation}', prefix + '-guide'))
                cells.append(chunk)
    result.cells = cells
    result.metadata['cs229']['teaching_layout'] = 'paired-steps-v1'
    return result
