#!/usr/bin/env python3
"""Build a CS229 Chapter 1 practice notebook; keep existing learner work by default."""
from __future__ import annotations
import argparse
from pathlib import Path
from textwrap import dedent
from inspect import cleandoc
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTES_URL = 'https://cs229.stanford.edu/main_notes.pdf'
NOTEBOOK_PATH = ROOT / 'chapters/01_linear_regression/linear_regression.ipynb'
DATA_DIR = NOTEBOOK_PATH.parent / 'data'


def md(text):
    return nbf.v4.new_markdown_cell(cleandoc(text).strip())


def code(text, exercise=None):
    cell = nbf.v4.new_code_cell(dedent(text).strip() + '\n')
    if exercise:
        cell.metadata['exercise'] = exercise
    return cell


def make_notebook(title, cells, slug):
    intro = md(f'''# {title}

    依据 **CS229 Lecture Notes，Tengyu Ma / Andrew Ng，2026-08-23**。
    [在线讲义]({NOTES_URL})。
    文中页码均为讲义印刷页码，PDF 阅读器页码需加 1。

    先沿着问题读懂公式，再自己把它写成代码。推导已经提供，代码练习标在 `TODO` 中；
    检查会显示“待完成”或“通过”，无需切换开关。出现其他错误时，修正后重新运行该格。
    这些是按讲义编写的自学实验，不是 Stanford 官方作业。''')
    nb = nbf.v4.new_notebook(cells=[intro, *cells])
    for i, c in enumerate(nb.cells):
        c.id = f'{slug}-notes-{i:02d}'
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.metadata['language_info'] = {'name': 'python', 'version': '3.11'}
    nb.metadata['cs229'] = {
        'offering': 'Spring 2026', 'artifact': 'notes-led-practice',
        'learning_mode': 'notes-to-code', 'notes_date': '2026-08-23',
        'notes_url': NOTES_URL, 'solutions_included': False,
    }
    return nb


HELPER = '''
def run_exercise(name, check):
    """An unfinished exercise is skipped; real errors are not hidden."""
    try:
        check()
    except NotImplementedError:
        print(f"待完成：{name}")
    else:
        print(f"通过：{name}")
'''


def _build_numpy_warmup():
    cells = [
        md('''## Goal

        **把讲义中的“一个样本”变成数组，再把“一批预测”变成矩阵运算。**
        房价例子贯穿本套练习：数组与索引 → 截距与设计矩阵 → 预测 → 损失 → 梯度检查。
        对照讲义第 9–12、15–16 页。每个问题里既有公式，也有需要你敲的代码。

        ## Setup

        讲义第 9 页展示了下列 5 行房屋记录。这里使用这 5 行，**不是讲义提到的完整 47 条数据**。
        面积单位为平方英尺，价格单位为千美元；卧室数是第二个输入特征。'''),
        code('''import numpy as np
np.set_printoptions(precision=4, suppress=True)

housing_rows = [
    [2104, 3, 400],
    [1600, 3, 330],
    [2400, 3, 369],
    [1416, 2, 232],
    [3000, 4, 540],
]
''' + HELPER),
        md('''## Steps

        ### Problem 1 — 从表格到样本

        一行是一个样本，一列是一个量。先把上面的 Python 列表变成浮点数组，
        再拆成 `features:(5,2)` 和 `target:(5,)`。

        **语法工具：** `np.array(data, dtype=float)` 创建数组；`a[:, :2]` 取全部行的前两列；
        `a[:, 2]` 取第三列；`a[0]` 取第一行。`(5,)` 是一维，`(5,1)` 是二维的单列。

        **代码练习：** 实现 `prepare_housing(rows)`，返回 `(features, target)`。
        然后实现 `describe_housing`：返回第一套房的特征、最后一套房的价格、面积列均值。
        这里先练数组和索引，不拟合模型。'''),
        code('''def prepare_housing(rows):
    # START TODO: 创建浮点数组，拆出两列输入和一列目标。
    raise NotImplementedError("Create and slice the housing array")
    # END TODO


def describe_housing(features, target):
    # 返回 (first_features:(2,), last_price:float, mean_area:float)。
    # START TODO: 练习行索引、负索引、列切片和 np.mean。
    raise NotImplementedError("Inspect rows and compute mean area")
    # END TODO
''', 'housing_arrays'),
        code('''def check_housing_arrays():
    features, target = prepare_housing(housing_rows)
    assert features.shape == (5, 2) and target.shape == (5,)
    assert np.issubdtype(features.dtype, np.floating)
    assert np.allclose(features[0], [2104, 3])
    assert np.allclose(target, [400, 330, 369, 232, 540])
    first, last, mean_area = describe_housing(features, target)
    assert np.allclose(first, [2104, 3])
    assert last == 540 and np.isclose(mean_area, 2104)
    print("第一套房：", first, "最后价格：", last, "平均面积：", mean_area)

run_exercise("数组与索引", check_housing_arrays)
'''),
        md(r'''### Problem 2 — 从一套房到一批预测

        讲义的模型是 $h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2$。
        在每行最前面加一个 1，就能把截距放进参数向量：

        $$X_i=[1,x_{i1},x_{i2}],\qquad \hat y_i=X_i\theta,\qquad \hat y=X\theta.$$

        这里 $X:(m,p)$、$\theta:(p,)$，所以预测是 $(m,)$。
        `p` 是含截距的总列数；讲义里的 `d` 不含截距，因此 `p=d+1`。

        **语法工具：** `np.ones(m)` 创建全 1 数组；`np.column_stack` 把列并排拼好；
        `@` 是矩阵乘法，`*` 是逐元素乘法；`np.dot(a,b)` 是两个一维向量的内积。

        **代码练习：** 实现 `make_design_matrix(features)`，给二维输入添加截距列；
        实现 `predict_loop`，用循环每次预测一行，再转换成一维数组。
        实现矩阵版本 `linear_predictions`，对照两种写法。'''),
        code('''def make_design_matrix(features):
    # features:(m,d) -> X:(m,d+1)，第一列为 1。
    # START TODO
    raise NotImplementedError("Add the intercept column")
    # END TODO


def predict_loop(X, theta):
    # X:(m,p), theta:(p,) -> predictions:(m,)。每次遍历一行。
    # START TODO: 使用列表收集每行的内积，最后转成数组。
    raise NotImplementedError("Predict each row in a loop")
    # END TODO


def linear_predictions(X, theta):
    # START TODO: 用一次矩阵乘法预测所有样本。
    raise NotImplementedError("Implement vectorized predictions")
    # END TODO
''', 'prediction'),
        code('''def check_predictions():
    features, target = prepare_housing(housing_rows)
    X = make_design_matrix(features)
    theta = np.array([50., 0.1, 10.])  # 给定参数，用于预测；这里还没有训练。
    by_loop = predict_loop(X, theta)
    by_matrix = linear_predictions(X, theta)
    assert X.shape == (5, 3) and np.all(X[:, 0] == 1)
    assert by_loop.shape == by_matrix.shape == (5,)
    assert np.allclose(by_loop, by_matrix)
    assert np.isclose(by_matrix[0], 290.4)
    # 非方阵测试：输出长度应取决于样本数，而不是参数数。
    small_X = np.array([[1., -2.], [1., 0.], [1., 3.]])
    assert np.allclose(linear_predictions(small_X, np.array([1., 2.])), [-3., 1., 7.])
    print("预测：", by_matrix, "真实价格：", target)

run_exercise("循环与矩阵预测", check_predictions)
'''),
        md(r'''### Problem 3 — 用损失描述“预测错了多少”

        逐样本误差 $r=X\theta-y$ 是向量；损失是把这些误差平方后汇总成一个数。
        讲义使用 $J_{\rm sum}=\frac12\sum_i r_i^2$。
        本练习使用 $J=J_{\rm sum}/m=\frac1{2m}\sum_i r_i^2$，便于比较样本数不同的数据。
        两者最优参数相同，但梯度相差 `m` 倍，学习率不能直接照搬。MSE 等于这里的 $2J$。

        **语法工具：** `a ** 2` 对数组逐元素平方，`np.mean(a)` 返回平均值。
        不要把 `y:(m,)` 改成 `(m,1)`：它与一维预测相减会广播成 `(m,m)`。

        **代码练习：** 实现 `squared_error_loss(X,y,theta)`；再实现 `compare_parameters`，
        对每组候选参数计算损失，返回 `(losses, best_theta)`。`np.argmin` 返回最小值的下标。'''),
        code('''def squared_error_loss(X, y, theta):
    # 返回 J = MSE / 2，一个标量。
    # START TODO
    raise NotImplementedError("Compute half of mean squared error")
    # END TODO


def compare_parameters(X, y, candidates):
    # candidates:(k,p) -> (losses:(k,), best_theta:(p,))。
    # START TODO: 遍历候选参数，收集损失，选择最小者。
    raise NotImplementedError("Evaluate candidate parameters")
    # END TODO
''', 'loss'),
        code('''# 用小数据检查公式，避免房价数值掩盖基础运算。
X_small = np.array([[1., -2.], [1., -1.], [1., 0.], [1., 1.], [1., 2.]])
y_small = np.array([-3.1, -1., 1.1, 2.9, 5.2])
theta_small = np.array([1., 2.])


def check_loss():
    loss = squared_error_loss(X_small, y_small, theta_small)
    assert np.ndim(loss) == 0 and np.isclose(loss, 0.007)
    candidates = np.array([[0., 0.], [1., 2.], [5., -1.]])
    losses, best = compare_parameters(X_small, y_small, candidates)
    assert losses.shape == (3,) and np.allclose(best, [1., 2.])
    assert np.isclose(losses[1], loss)
    print("各组参数的损失：", losses, "最佳候选：", best)

run_exercise("损失与参数比较", check_loss)
'''),
        md(r'''### Problem 4 — 梯度是每个参数该如何调整的依据

        对一个参数用链式法则，再把所有样本的贡献相加：

        $$\frac{\partial J}{\partial\theta_j}
        =\frac1m\sum_i (X_i\theta-y_i)X_{ij},\qquad
        \nabla_\theta J=\frac1mX^T(X\theta-y).$$

        形状顺序是 $(m,p)@(p,)\to(m,)$，再由 $(p,m)@(m,)\to(p,)$。
        梯度与参数的 shape 相同；标量损失不能代替误差向量参与这里的矩阵乘法。

        **代码练习：** 实现 `squared_error_gradient`。再实现 `numerical_gradient`：
        每次只给第 `j` 个参数加、减 `epsilon`，按下面公式估计偏导数。

        $$g_j\approx\frac{J(\theta+\epsilon e_j)-J(\theta-\epsilon e_j)}{2\epsilon}.$$

        **语法工具：** `np.zeros_like(theta, dtype=float)` 创建同 shape 的零数组；
        `step[j]=epsilon` 改一个元素，`.T` 转置二维矩阵。
        两种梯度一致后，这套计算就能直接用于本章后面的训练循环。'''),
        code('''def squared_error_gradient(X, y, theta):
    # START TODO: 返回梯度 (p,)，不在这里更新 theta。
    raise NotImplementedError("Compute the analytical gradient")
    # END TODO


def numerical_gradient(fn, theta, epsilon=1e-6):
    # fn 接收参数向量，返回一个损失值；输出与 theta 同 shape。
    # START TODO: 逐个参数做中心差分。
    raise NotImplementedError("Compute finite-difference gradients")
    # END TODO
''', 'gradient'),
        code('''def check_gradients():
    # 使用未拟合的参数；若只在最优点检查，很多错误会被零梯度掩盖。
    theta = np.array([0.2, -0.3])
    analytic = squared_error_gradient(X_small, y_small, theta)
    numeric = numerical_gradient(lambda t: squared_error_loss(X_small, y_small, t), theta)
    assert analytic.shape == numeric.shape == theta.shape
    assert np.allclose(analytic, numeric, atol=1e-6)
    X_extra = np.array([[1., 0., 2.], [1., 1., -1.], [1., 3., 0.], [1., -2., 1.]])
    y_extra = np.array([1., -1., 3., 2.])
    t_extra = np.array([0.1, 0.4, -0.2])
    assert np.allclose(
        squared_error_gradient(X_extra, y_extra, t_extra),
        numerical_gradient(lambda t: squared_error_loss(X_extra, y_extra, t), t_extra),
        atol=1e-6,
    )
    print("解析梯度：", analytic, "数值梯度：", numeric)

run_exercise("梯度检查", check_gradients)
'''),
        md('''## Checks

        判断自己掌握了没有：换一组参数或增加一行样本，预测、损失、梯度仍能工作；
        能解释“预测和误差是一组数，损失是一个数，梯度是每个参数一个数”。
        可以直接运行函数观察结果，错误和查语法都是练习的一部分。

        ## Next Steps

        接下来复用这条计算链，用训练数据寻找参数，而不是只计算给定参数的预测。
        后续练习继续复用预测、损失与梯度函数。'''),
    ]
    return cells


def _build_regression():
    cells = [
        md('''## Goal

        **同一个回归问题，沿讲义第 1 章走完整条思路。**
        P1 用梯度下降寻找参数（§1.1，第 10–14 页）；P2 用正规方程求相同目标（§1.2，第 14–16 页）；
        P3 解释为什么平方误差合理（§1.3，第 16–18 页）；P4 比较多项式与局部模型（§1.4，第 18–20 页）。
        各节先给精简推导，再留函数和完整实验给你写，不需要填写推导答题框。

        ## Setup

        先使用讲义展示的 5 行房价记录理解训练方法；它们不足以评价泛化。
        再使用原有合成曲线的 train/valid/test 文件比较模型，数据保持不变。
        训练集拟合参数，验证集选择 degree/tau，测试集留待选择结束后评价。'''),
        code('''from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)

housing = np.array([
    [2104., 3., 400.], [1600., 3., 330.], [2400., 3., 369.],
    [1416., 2., 232.], [3000., 4., 540.],
])
# 只在这 5 个训练点上比较优化方法。标准化仅基于这些训练输入。
raw_features, y_housing = housing[:, :2], housing[:, 2]
feature_mean = raw_features.mean(axis=0)
feature_scale = raw_features.std(axis=0)
X_housing = np.column_stack([np.ones(len(housing)), (raw_features - feature_mean) / feature_scale])
# 所有房价都仍以千美元为单位。不同尺度若直接进入 GD，步长可能不稳定。

candidates = [Path('data'), Path('chapters/01_linear_regression/data')]
DATA_DIR = next((p for p in candidates if (p / 'train.csv').is_file()), None)
if DATA_DIR is None:
    raise FileNotFoundError('Find chapters/01_linear_regression/data')


def load_xy(filename):
    table = np.loadtxt(DATA_DIR / filename, delimiter=',', skiprows=1)
    return table[:, 0], table[:, 1]

x_train, y_train = load_xy('train.csv')
x_valid, y_valid = load_xy('valid.csv')
x_test, y_test = load_xy('test.csv')
print('Housing:', X_housing.shape, y_housing.shape)
print('Curve train/valid/test:', len(x_train), len(x_valid), len(x_test))
''' + HELPER),
        md(r'''## Steps

        ### Problem 1 — 让误差推动参数变化

        模型、目标、更新连在一起，就是讲义 §1.1 的 LMS 思路：

        $$\hat y=X\theta,\quad J(\theta)=\frac1{2m}\|X\theta-y\|^2,\quad
        \theta\leftarrow\theta-\alpha\frac1mX^T(X\theta-y).$$

        这是**批量梯度下降**：一次更新使用全部样本。
        对单个样本，链式法则给出 $(X_i\theta-y_i)X_i$；
        每看到一个样本就更新一次，是**随机梯度下降**。
        为避免固定顺序的影响，本练习每个 epoch 随机打乱样本顺序。

        讲义用半平方误差的总和，这里使用平均形式；梯度多了 `1/m`，最优参数相同。
        `alpha` 是学习率，`epoch` 是把训练集完整遍历一遍。
        `m` 为样本数，`p` 为含截距的列数；讲义 `d=p-1`。

        **函数：** `mse` 计算均方误差，`fit_gradient_descent` 实现批量 GD。
        GD 每 10 步记录一次更新前 MSE，后面画图时横轴乘 10。
        **新增练习：** 实现 `fit_sgd(X,y,rng,learning_rate,epochs)`，从零参数开始，
        每个 epoch 用 `rng.permutation(len(y))` 遍历样本下标；记录该 epoch 全部更新后的 MSE。
        返回 `(theta:(p,), history:(epochs,))`。语法工具：`np.zeros`、`np.dot`、列表的 `append`。'''),
        code('''def mse(y_true, y_pred):
    # START TODO
    raise NotImplementedError("Compute mean squared error")
    # END TODO


def fit_gradient_descent(X, y, learning_rate=0.05, steps=1000):
    theta = np.zeros(X.shape[1])
    history = []
    for step in range(steps):
        # START TODO: 预测 -> 误差 -> 梯度 -> 更新，每 10 步记录更新前 MSE。
        raise NotImplementedError("Implement batch gradient descent")
        # END TODO
    return theta, np.array(history)


def fit_sgd(X, y, rng, learning_rate=0.01, epochs=100):
    # START TODO: 每个 epoch 打乱下标，逐样本更新，随后记录全体训练 MSE。
    raise NotImplementedError("Implement stochastic gradient descent")
    # END TODO
''', 'optimizers'),
        md('''**你来组织实验：** 实现 `compare_optimizers(X,y)`，用 GD 的 `lr=0.1, steps=2000`
        和 SGD 的 `lr=0.01, epochs=200` 分别拟合；随机数使用 `np.random.default_rng(229)`。
        返回 `(theta_gd, history_gd, theta_sgd, history_sgd)`。
        打印两组训练 MSE，并画两张误差图：GD 横轴是更新步数，SGD 横轴是 epoch。
        两条横轴的单位不同，不能仅凭曲线长度判断谁更快。
        正确结果应比全部预测为训练均值的 baseline 更好；SGD 不要求每个 epoch 都下降。'''),
        code('''def compare_optimizers(X, y):
    # START TODO: 调用两个训练器，计算 MSE，画图，返回四个结果。
    raise NotImplementedError("Run the housing optimization experiment")
    # END TODO


def check_optimizers():
    gd, h_gd, sgd, h_sgd = compare_optimizers(X_housing, y_housing)
    assert gd.shape == sgd.shape == (3,)
    assert h_gd.shape == (200,) and h_sgd.shape == (200,)
    assert np.all(np.isfinite(h_gd)) and np.all(np.isfinite(h_sgd))
    assert h_gd[-1] < h_gd[0] and h_sgd[-1] < h_sgd[0]
    baseline = mse(y_housing, np.full_like(y_housing, y_housing.mean()))
    assert mse(y_housing, X_housing @ gd) < baseline
    assert mse(y_housing, X_housing @ sgd) < baseline
    zero_gd, _ = fit_gradient_descent(X_housing, y_housing, learning_rate=0., steps=20)
    zero_sgd, _ = fit_sgd(X_housing, y_housing, np.random.default_rng(229), learning_rate=0., epochs=2)
    assert np.allclose(zero_gd, 0) and np.allclose(zero_sgd, 0)

run_exercise("房价：GD 与 SGD", check_optimizers)
''', 'optimization_experiment'),
        md(r'''### Problem 2 — 同一个最小值，也可以直接解出来

        讲义 §1.2 把损失展开后求导：

        $$J=\frac1{2m}(\theta^TX^TX\theta-2\theta^TX^Ty+y^Ty),$$
        $$\nabla_\theta J=\frac1m(X^TX\theta-X^Ty)=0
        \quad\Longrightarrow\quad X^TX\theta=X^Ty.$$

        当 $X$ 满列秩时，最小值唯一。程序用 `np.linalg.solve(A,b)` 解 $A\theta=b$，
        不必显式求逆。若列线性相关，正规方程可能无法这样求解；不要把报错误当成过拟合。

        **函数：** `fit_normal_equation` 解正规方程，`add_intercept` 为一维输入添加截距列。
        **新增练习：** 实现 `compare_solvers`，在同一份合成曲线训练集上用 NE 和 GD 各求一组参数，
        用训练得到的参数预测同一验证集，返回 `(theta_ne, theta_gd, valid_mse_ne, valid_mse_gd)`。
        打印参数、两组验证 MSE 和训练均值 baseline，并把验证数据与拟合直线画在一起。
        这是复习完整计算链，不是再推一遍公式。'''),
        code('''def add_intercept(x):
    # START TODO: 一维输入 (m,) -> 设计矩阵 (m,2)。
    raise NotImplementedError("Add a column of ones")
    # END TODO


def fit_normal_equation(X, y):
    # START TODO
    raise NotImplementedError("Solve the normal equations")
    # END TODO


def compare_solvers(x_train, y_train, x_valid, y_valid):
    # START TODO: 准备设计矩阵、两次拟合、预测、评价和作图。
    raise NotImplementedError("Compare normal equations with gradient descent")
    # END TODO
''', 'solvers'),
        code('''def check_solvers():
    small_X = np.array([[1., 0.], [1., 1.]])
    assert np.allclose(fit_normal_equation(small_X, np.array([1., 3.])), [1., 2.])
    theta_ne, theta_gd, valid_ne, valid_gd = compare_solvers(x_train, y_train, x_valid, y_valid)
    assert theta_ne.shape == theta_gd.shape == (2,)
    assert np.allclose(theta_ne, theta_gd, atol=1e-6)
    assert np.isclose(valid_ne, mse(y_valid, add_intercept(x_valid) @ theta_ne))
    assert np.isclose(valid_ne, valid_gd)
    assert valid_ne < mse(y_valid, np.full_like(y_valid, y_train.mean()))
    # 讲义房价例子：验证两种方法也能处理多个输入特征。
    housing_ne = fit_normal_equation(X_housing, y_housing)
    housing_gd, _ = fit_gradient_descent(X_housing, y_housing, learning_rate=0.1, steps=2000)
    assert np.allclose(housing_ne, housing_gd, atol=1e-3)

run_exercise("同一数据、两种求解方法", check_solvers)
'''),
        md(r'''### Problem 3 — 为什么选择平方误差

        讲义 §1.3 假设 $y_i=X_i\theta+\epsilon_i$，误差独立同分布，且
        $\epsilon_i\sim N(0,\sigma^2)$。
        独立性让各样本密度相乘，取对数后变为相加：

        $$\ell(\theta)=-\frac m2\log(2\pi\sigma^2)
        -\frac1{2\sigma^2}\sum_i(y_i-X_i\theta)^2.$$

        固定正数 $\sigma$ 后，第一项与 $\theta$ 无关。
        最大化似然等价于最小化平方误差。这里 `sigma` 是标准差，`sigma**2` 才是方差；
        这是平方误差的一种概率解释，不要求每个回归数据集都真的服从高斯分布。

        **代码练习：** 实现 `gaussian_log_likelihood(X,y,theta,sigma)`，返回一个标量。
        使用 `np.log`、`np.sum`，`sigma<=0` 时抛出 `ValueError`。
        实现 `compare_likelihoods`，对候选参数计算 MSE 与 log-likelihood，返回两组一维数组。
        测试会检验：固定 sigma 时，MSE 越小，log-likelihood 越大。'''),
        code('''def gaussian_log_likelihood(X, y, theta, sigma):
    # START TODO: 检查 sigma，计算高斯误差模型下的对数似然。
    raise NotImplementedError("Compute Gaussian log likelihood")
    # END TODO


def compare_likelihoods(X, y, candidates, sigma):
    # candidates:(k,p) -> (mses:(k,), log_likelihoods:(k,))。
    # START TODO: 对相同候选参数计算两种评价。
    raise NotImplementedError("Compare likelihood and squared error")
    # END TODO
''', 'likelihood'),
        code('''def check_likelihoods():
    X = np.array([[1., 0.], [1., 1.], [1., 2.]])
    y = np.array([1., 3., 5.])
    candidates = np.array([[1., 2.], [0., 0.], [2., 2.]])
    mses, ll = compare_likelihoods(X, y, candidates, sigma=0.5)
    assert mses.shape == ll.shape == (3,)
    assert np.all(np.isfinite(ll))
    assert np.isclose(ll[0], -len(y) / 2 * np.log(2 * np.pi * 0.5**2))
    assert np.allclose(ll[0] - ll, len(y) / (2 * 0.5**2) * mses)
    assert np.array_equal(np.argsort(mses), np.argsort(-ll))
    _, ll_wider = compare_likelihoods(X, y, candidates, sigma=2.)
    assert np.array_equal(np.argsort(mses), np.argsort(-ll_wider))
    try:
        gaussian_log_likelihood(X, y, candidates[0], sigma=0.)
    except ValueError:
        pass
    else:
        raise AssertionError("sigma must be positive")
    print("候选参数 MSE：", mses, "对数似然：", ll)

run_exercise("平方误差与最大似然", check_likelihoods)
'''),
        md(r'''### Problem 4 — 数据弯曲时，改变模型如何拟合

        讲义 §1.4 先比较直线与多项式：增加 $x^2,x^3,\ldots$，
        模型对原始输入可以弯曲，但对参数仍然是线性的。

        $$\phi_k(x)=[1,x,\ldots,x^k],\qquad \hat y=\phi_k(x)\theta.$$

        `polynomial_features(x,degree)` 输入 `(m,)`，输出 `(m,degree+1)`。
        例如 `[2,3]` 的三次特征是 `[[1,2,4,8],[1,3,9,27]]`。
        **语法工具：** `x**power` 一次计算整列；`range(degree+1)` 包含零次方；
        `np.column_stack` 拼列；`np.linspace` 生成只用于画图的密集横坐标。

        接着，讲义提出另一种方式：每次预测查询点 $q$，重点拟合附近的训练样本。

        $$w_i(q)=\exp\left(-\frac{(x_i-q)^2}{2\tau^2}\right),\quad
        J_q=\frac12\sum_i w_i(q)(X_i\theta-y_i)^2,$$
        $$X^TW_qX\theta_q=X^TW_qy,\quad \hat y(q)=[1,q]\theta_q.$$

        权重随查询点改变，换一个 $q$ 就重新拟合一组局部参数。
        `tau` 是带宽：小值强调更近的点，大值纳入更远的点。
        权重不是概率密度，无需加高斯密度的归一化系数。

        **代码练习：** 实现多项式特征和 `predict_lwr`。
        后者对查询数组 `(n,)` 遍历，计算权重、`np.diag(weights)`、加权正规方程及预测，
        返回 `(n,)`；`tau<=0` 时抛出 `ValueError`。拟合只使用训练数据，不读取查询点标签。'''),
        code('''def polynomial_features(x, degree):
    # START TODO
    raise NotImplementedError("Build powers zero through degree")
    # END TODO


def predict_lwr(x_train, y_train, x_query, tau):
    # START TODO: 每个查询位置重新加权并拟合，不显式求矩阵逆。
    raise NotImplementedError("Fit one local model per query point")
    # END TODO
''', 'flexible_models'),
        md('''**你来写完整比较实验：** 实现 `compare_models`，在同一训练集分别拟合
        degree 为 `1,3,10` 的多项式，以及 tau 为 `0.1,0.5,2.0` 的 LWR。
        用同一验证集算 MSE，打印六行结果，并分别画多项式和 LWR 的三条预测曲线。
        返回 `results`，每项是包含 `model`、`setting`、`train_mse`、`valid_mse` 的字典。
        最后用 `min(results, key=...)` 选出验证 MSE 最低的设置并打印。

        训练误差小而验证误差更大，才是本次实验里过拟合的证据；不要预先断言高次一定更差。
        LWR 在训练点上的误差包含该点自身，常常偏乐观，应重点看验证误差。
        原有数据由 `0.4*x + sin(1.7*x) + noise` 生成，线性模型的误差平台可能来自曲率，
        不是梯度下降没有学会。讲义中的图是概念示例；这里的数值结论需要你实际运行。'''),
        code('''def compare_models(x_train, y_train, x_valid, y_valid):
    # START TODO: 六个设置各自拟合，计算两组 MSE，画两张图，返回六项字典。
    raise NotImplementedError("Run polynomial and local regression experiments")
    # END TODO


def check_models():
    assert np.allclose(polynomial_features(np.array([2., 3.]), 3), [[1., 2., 4., 8.], [1., 3., 9., 27.]])
    assert polynomial_features(np.array([2., 3.]), 0).shape == (2, 1)
    # 精确线性数据：任意合理带宽都应恢复同一条直线。
    x = np.linspace(-2., 2., 9)
    q = np.array([-0.7, 0., 0.8])
    for tau in (0.5, 2.):
        pred = predict_lwr(x, 1. + 2. * x, q, tau)
        assert pred.shape == q.shape and np.allclose(pred, 1. + 2. * q, atol=1e-8)
    try:
        predict_lwr(x, 1. + 2. * x, q, 0.)
    except ValueError:
        pass
    else:
        raise AssertionError("tau must be positive")
    results = compare_models(x_train, y_train, x_valid, y_valid)
    assert len(results) == 6
    settings = {(r['model'], r['setting']) for r in results}
    assert settings == {('polynomial', 1), ('polynomial', 3), ('polynomial', 10), ('lwr', 0.1), ('lwr', 0.5), ('lwr', 2.)}
    for r in results:
        assert np.isfinite(r['train_mse']) and np.isfinite(r['valid_mse'])
        if r['model'] == 'polynomial':
            theta = fit_normal_equation(polynomial_features(x_train, r['setting']), y_train)
            train_pred = polynomial_features(x_train, r['setting']) @ theta
            valid_pred = polynomial_features(x_valid, r['setting']) @ theta
        else:
            train_pred = predict_lwr(x_train, y_train, x_train, r['setting'])
            valid_pred = predict_lwr(x_train, y_train, x_valid, r['setting'])
        assert np.isclose(r['train_mse'], mse(y_train, train_pred))
        assert np.isclose(r['valid_mse'], mse(y_valid, valid_pred))

run_exercise("多项式与局部回归", check_models)
''', 'model_experiment'),
        md('''## Checks

        回看这条主线：**表示模型 → 定义损失 → 优化参数 → 解释目标 → 比较模型能力**。
        能独立从输入数据组织到预测和评价，比记住某一个库函数更重要。
        观察结果可以口头讨论，不需要另写推导或长篇实验报告。

        ## Next Steps

        先把基础与回归做顺，再决定下一套题的形式。分类和无监督学习已移出当前学习主线。
        之后的 PyTorch 练习可以复用同一份数据、同一种损失，对照手写梯度与自动求导。'''),
    ]
    return cells



def build_torch_bridge():
    """Keep the first framework experiment on the same regression objective."""
    return [
        md(r'''## 同一个回归实验：从 NumPy 到 PyTorch

        现在就可以学 PyTorch：会写预测、损失和一次 GD 更新后，直接重做同一个实验。
        这是讲义 §1.1–1.2 的代码延伸，讲义本身不要求使用 PyTorch。
        **模型、数据与目标都不变，只把手写求导交给自动求导，再比较结果。**

        ### 先比较一次梯度

        $$J(\theta)=\tfrac12\operatorname{mean}[(X\theta-y)^2],\qquad
        g_{\rm NumPy}=X^T(X\theta-y)/m.$$

        PyTorch 记录从参数到损失的计算过程，`loss.backward()` 按链式法则求导，
        结果放在参数的 `.grad` 中。这不是差分近似，也不是参数更新。
        保持 `X:(m,p)`、`y:(m,)`、`theta:(p,)`；CPU 与 `torch.float64` 便于和 NumPy 对照。

        **语法工具：** `torch.as_tensor(a, dtype=torch.float64)` 转成张量；
        `torch.tensor(a, dtype=torch.float64, requires_grad=True)` 创建需要求导的参数副本；
        `torch.mean(a ** 2)` 求平方的平均；`t.detach().numpy()` 返回不参与求导的 NumPy 数组。
        返回梯度后，用 `squared_error_gradient` 在同一个未拟合参数上比较。

        在当前 notebook 的 kernel 环境安装 `torch` 后重启 kernel、Run All。
        未安装时这里只提示缺少依赖，前面的 NumPy 部分照常运行。
        需要查语法时参考 [官方自动求导教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。'''),
        code('''try:
    import torch
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    torch = None
    print("缺少依赖：PyTorch。请在当前 kernel 环境安装 torch，再重启 kernel。")
else:
    print("PyTorch:", torch.__version__, "本实验使用 CPU / float64")


def torch_gradient(X, y, theta):
    # 返回 NumPy 梯度数组 (p,)，输入 theta 不应被修改。
    # START TODO: 转张量 -> 前向预测 -> 半均方误差 -> backward -> 取梯度。
    raise NotImplementedError("Compute the same gradient with autograd")
    # END TODO
''', 'torch_gradient'),
        code('''def check_torch_gradient():
    theta = np.array([0.2, -0.3])
    original = theta.copy()
    gradient = torch_gradient(X_small, y_small, theta)
    manual = squared_error_gradient(X_small, y_small, theta)
    assert isinstance(gradient, np.ndarray) and gradient.shape == theta.shape
    assert np.allclose(gradient, manual, rtol=1e-10, atol=1e-10)
    assert np.array_equal(theta, original), "Computing a gradient must not update theta"
    # 换成三列输入，检查代码没有写死参数个数。
    X = np.array([[1., 0., 2.], [1., 1., -1.], [1., 3., 0.], [1., -2., 1.]])
    y = np.array([1., -1., 3., 2.])
    t = np.array([0.1, 0.4, -0.2])
    assert np.allclose(torch_gradient(X, y, t), squared_error_gradient(X, y, t), atol=1e-10)
    print("NumPy / PyTorch 梯度：", manual, gradient)

if torch is not None:
    run_exercise("NumPy 与自动求导梯度", check_torch_gradient)
'''),
        md('''### 再写完整训练循环

        `torch.optim.SGD([theta], lr=learning_rate)` 创建更新参数的优化器。
        虽然优化器名字是 SGD，这次每一步使用全部训练样本，因此实际是批量梯度下降。
        默认不使用 momentum 或 weight decay。

        每步按 **清空旧梯度 → 预测与损失 → 求梯度 → 更新参数** 的顺序组织。
        `.zero_grad()` 清空之前累积的梯度；`.backward()` 求本次梯度；`.step()` 执行更新。
        `.item()` 将标量张量变成 Python 数值，适合记录误差。
        本练习从全零参数开始，与 NumPy GD 一样，每 10 步记录一次**更新前的 MSE**。
        注意目标是半均方误差，记录的 MSE 是它的两倍。

        **代码练习：** 实现 `fit_torch_gradient_descent`，返回 NumPy 参数和误差历史。
        再实现 `compare_numpy_torch`：使用相同训练集、全零初值、学习率和更新次数分别训练，
        在相同验证集打印两者 MSE，并把两条训练误差画在同一张图中。
        横轴为实际更新步数 `np.arange(len(history)) * 10`。
        返回 `(theta_numpy, history_numpy, theta_torch, history_torch)`。

        先让默认设置通过，再把两种实现的学习率一起改成 `0.01`，观察收敛是否同步变化。
        如果结果不同，先检查损失的系数、shape、初值与梯度是否清空。
        做完这个对照后，再用同一曲线尝试小神经网络。
        参考 [官方优化器与训练循环教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。'''),
        code('''def fit_torch_gradient_descent(X, y, learning_rate=0.05, steps=1000):
    # 返回 (theta:(p,), history:(ceil(steps/10),))，均为 NumPy 数组。
    # START TODO: 用 float64 参数、SGD 优化器写批量训练循环。
    raise NotImplementedError("Train linear regression with PyTorch")
    # END TODO


def compare_numpy_torch(X_train, y_train, X_valid, y_valid, learning_rate=0.05, steps=1000):
    # START TODO: 用相同设置训练，比较验证 MSE，画训练误差，返回四个结果。
    raise NotImplementedError("Compare NumPy and PyTorch on the same regression")
    # END TODO
''', 'torch_training'),
        code('''def check_torch_training():
    # 对照任意少量更新，比只在最优点对照更容易发现训练循环错误。
    for steps in (1, 17):
        expected, expected_history = fit_gradient_descent(X_small, y_small, learning_rate=0.05, steps=steps)
        actual, actual_history = fit_torch_gradient_descent(X_small, y_small, learning_rate=0.05, steps=steps)
        assert isinstance(actual, np.ndarray) and isinstance(actual_history, np.ndarray)
        assert actual.shape == (2,) and actual_history.shape == expected_history.shape
        assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)
        assert np.allclose(actual_history, expected_history, rtol=1e-10, atol=1e-10)
    frozen, _ = fit_torch_gradient_descent(X_small, y_small, learning_rate=0., steps=20)
    assert np.allclose(frozen, 0)
    X_train, X_valid = add_intercept(x_train), add_intercept(x_valid)
    n, hn, t, ht = compare_numpy_torch(X_train, y_train, X_valid, y_valid)
    expected, expected_history = fit_gradient_descent(X_train, y_train)
    assert n.shape == t.shape == (2,) and hn.shape == ht.shape == (100,)
    assert np.allclose(n, expected) and np.allclose(hn, expected_history)
    assert np.allclose(t, n, rtol=1e-8, atol=1e-8)
    assert np.allclose(ht, hn, rtol=1e-8, atol=1e-8)
    assert np.isclose(mse(y_valid, X_valid @ t), mse(y_valid, X_valid @ n))

if torch is not None:
    run_exercise("NumPy 与 PyTorch 训练实验", check_torch_training)
'''),
    ]


def build_chapter():
    warmup = _build_numpy_warmup()[1:-1]
    regression = _build_regression()[1:-1]
    # A single setup and a single narrative; do not split the chapter into problem sets.
    for cell in warmup:
        if cell.cell_type == 'markdown':
            cell.source = cell.source.replace('## Steps\n\n', '')
            cell.source = cell.source.replace('### Problem 1 — 从表格到样本', '## 回归的准备：把样本变成计算\n\n### 数组与索引')
            cell.source = cell.source.replace('### Problem 2 — 从一套房到一批预测', '### 从一套房到一批预测')
            cell.source = cell.source.replace('### Problem 3 — 用损失描述“预测错了多少”', '### 用损失描述“预测错了多少”')
            cell.source = cell.source.replace('### Problem 4 — 梯度是每个参数该如何调整的依据', '### 梯度是每个参数该如何调整的依据')
    for cell in regression:
        if cell.cell_type == 'markdown':
            cell.source = cell.source.replace('## Steps\n\n', '')
            cell.source = cell.source.replace('### Problem 1 — 让误差推动参数变化', '## 1.1 LMS：让误差推动参数变化')
            cell.source = cell.source.replace('### Problem 2 — 同一个最小值，也可以直接解出来', '## 1.2 正规方程：直接求同一个最小值')
            cell.source = cell.source.replace('### Problem 3 — 为什么选择平方误差', '## 1.3 概率解释：为什么选择平方误差')
            cell.source = cell.source.replace('### Problem 4 — 数据弯曲时，改变模型如何拟合', '## 1.4 局部加权回归：数据弯曲时如何拟合')
        elif HELPER.strip() in cell.source:
            cell.source = cell.source.replace(HELPER.strip(), '').rstrip() + '\n'
            cell.source = '# 后续训练输入：同一房价例子标准化后用于比较优化方法。\n' + cell.source
    bridge_index = next(i for i, cell in enumerate(regression) if cell.cell_type == 'markdown' and cell.source.startswith('## 1.3'))
    regression[bridge_index:bridge_index] = build_torch_bridge()
    goal = md("""## Goal

    把你读完的讲义第一章变成可以独立写出的代码：
    **样本与预测 → 损失与梯度 → GD/SGD → 正规方程 → 概率解释 → 多项式与局部回归。**
    NumPy 语法嵌在对应任务中，推导由本页提供，函数与完整实验由你实现。
    做完以后，换数据、换参数和改变模型都应该能说明原因，不靠复述讲义。

    ## Setup

    依赖 `numpy`、`matplotlib` 和 Jupyter；框架对照实验另需 `torch`。
    写过预测、损失和 GD 后即可做 PyTorch 对照，无需等整章完成。
    先用讲义第 9 页展示的 5 行房价记录练数组与优化；
    **这不是讲义完整的 47 条数据，不能凭这 5 行评价泛化。**
    面积单位为平方英尺，价格单位为千美元；卧室数是第二个特征。
    再用原有合成曲线的训练、验证、测试文件比较模型：训练拟合，验证选设置，测试留到最后。
    房价训练输入会做标准化，标准化统计量仅从该训练输入计算。
    """)
    ending = md("""## Checks

    回看整章：表示模型、定义目标、求参数、解释目标、比较模型能力。
    修改一组输入、参数或学习率再运行，是检验自己是否理解的办法。
    观察结果可以直接在聊天里讨论，页面不要求你抄推导或填写长篇答题框。
    每个检查的“待完成”表示代码尚未实现，不代表已经通过。

    ## Next Steps

    掌握本章的计算与实验后，可用同一数据尝试小神经网络，或继续阅读讲义后续章节。
    """)
    return make_notebook('CS229 第 1 章 — 线性回归：从讲义到代码', [goal, *warmup, *regression, ending], 'ch01')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='explicitly replace the notebook with a blank template; back up learner work first')
    args = parser.parse_args()
    if NOTEBOOK_PATH.exists() and not args.force:
        print(f'KEEP {NOTEBOOK_PATH.relative_to(ROOT)} (learner work preserved)')
        return
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_chapter(), NOTEBOOK_PATH)
    print(f'WRITE {NOTEBOOK_PATH.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
