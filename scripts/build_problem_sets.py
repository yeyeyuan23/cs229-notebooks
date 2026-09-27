#!/usr/bin/env python3
"""Build source-grounded, simplified CS229 practice problem sets."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import numpy as np

from lesson_layout import apply_layout


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "problem_sets"
COURSE = "https://cs229.stanford.edu/index.html-spr26"
VIDEOS = "https://www.youtube.com/playlist?list=PLaqpC4kq8Gpw"
NOTES = "https://cs229.stanford.edu/main_notes.pdf"
OFFICIAL_2020 = "https://cs229.stanford.edu/summer2020/"
PUBLIC_2025 = "https://github.com/MDzimah/Stanford-University-CS229-Machine-Learning"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def answer_box(prompt: str):
    return md(
        f"""
        > **你的答案（请双击本格后填写）**
        >
        > TODO：{prompt}
        """
    )


def notebook(title: str, subtitle: str, cells: list, slug: str):
    header = md(
        f"""
        # {title}

        {subtitle}

        这不是 Stanford 官方作业副本，而是用于自学的**简化原创版本**。题型参考公开的
        CS229 problem sets；公式以 2026 Main Notes 为准，数据由本仓库确定性生成。

        - [Spring 2026 课程主页]({COURSE})
        - [Spring 2026 公开视频]({VIDEOS})
        - [CS229 2026 Main Notes]({NOTES})
        - [Stanford 官方公开的 Summer 2020 problem sets]({OFFICIAL_2020})
        - [公开可见的 Summer 2025 作业结构参考]({PUBLIC_2025})

        **使用规则：** 先手推，再写代码。不要先看学生 solution。每道题完成后才把对应的
        `RUN_*_CHECKS` 改成 `True`。检查不通过时，从 shape 和公式开始定位。

        **怎么读题：** `m` 表示样本数，`d` 表示设计矩阵的列数（含截距列时也计入）。
        `X` 的每一行是一条样本，`y[i]` 是对应目标；`(m,)` 是一维数组，和 `(m, 1)` 不同。
        “实现”指填写函数中的 TODO；“运行检查”指执行已经给出的调用代码；“解释”写在答题框中。
        函数定义格和检查格需要分别运行。检查通过只代表已检查的条件成立，不代替解释实验结果。
        """
    )
    all_cells = [header, *cells]
    for index, cell in enumerate(all_cells):
        cell.id = f"{slug}-cell-{index:02d}"
    nb = nbf.v4.new_notebook(cells=all_cells)
    nb.metadata["kernelspec"] = {
        "display_name": "dev",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    nb.metadata["cs229"] = {
        "offering": "Spring 2026",
        "artifact": "simplified-source-grounded-problem-set",
        "solutions_included": False,
    }
    return apply_layout(nb, slug)


def setup_cell(folder: str):
    return code(
        f"""
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt

        np.set_printoptions(precision=4, suppress=True)
        rng = np.random.default_rng(229)

        candidates = [Path("data"), Path("problem_sets/{folder}/data")]
        DATA_DIR = next((path for path in candidates if path.exists()), None)
        if DATA_DIR is None:
            raise FileNotFoundError("Run: python scripts/build_problem_sets.py")

        print("Data directory:", DATA_DIR.resolve())
        """
    )


def build_ps0():
    cells = [
        md(
            r"""
            ## Goal

            这套预备题只补 CS229 后续代码必须用到的矩阵和梯度。完成后，你应该能看懂
            $X\theta$ 的 shape，并能验证平方误差的梯度。

            ## Setup

            只使用 NumPy。下面的数据和参数是题目给定的，不需要自己发明业务背景。
            """
        ),
        code(
            """
            import numpy as np
            np.set_printoptions(precision=4, suppress=True)

            X = np.array([
                [1.0, -2.0],
                [1.0, -1.0],
                [1.0,  0.0],
                [1.0,  1.0],
                [1.0,  2.0],
            ])
            y = np.array([-3.1, -1.0, 1.1, 2.9, 5.2])
            theta = np.array([1.0, 2.0])

            print("X:", X.shape, "y:", y.shape, "theta:", theta.shape)
            """
        ),
        md(
            r"""
            ## Problem 1 — 从 shape 判断矩阵乘法

            **为什么做：** 从 Lecture 2 开始，每个模型都要同时处理很多样本。这里先确认
            $(5,2)@(2,)\rightarrow(5,)$ 的含义。

            **已知：** 上方 `X` 有 5 行，每行是 `[1, x_i]`；第一列的 1 用来表示截距。
            `theta = [theta_0, theta_1]` 是给定参数，模型对一个样本的预测为
            $\hat y_i=\theta_0+\theta_1x_i$。本题只计算预测，不训练参数，也不需要用 `y`。

            1. 运行前，在纸上写出 `X @ theta` 的 shape。
            2. 实现 `linear_predictions(X, theta)`：输入 `(m, d)` 和 `(d,)`，返回 `(m,)`，不能使用 Python 循环。
            3. 在答题框解释 `predictions[i]`，并手算第一行的预测，再打开 `RUN_P1_CHECKS` 对照。
            """
        ),
        code(
            """
            def linear_predictions(X, theta):
                # Return one prediction per row of X. Expected shape: (m,).
                # START TODO
                raise NotImplementedError("Implement X @ theta")
                # END TODO
            """
        ),
        answer_box("写出输出 shape，并解释 `predictions[i]` 的含义。"),
        code(
            """
            RUN_P1_CHECKS = False
            if RUN_P1_CHECKS:
                predictions = linear_predictions(X, theta)
                assert predictions.shape == (5,)
                assert np.allclose(predictions, [-3, -1, 1, 3, 5])
                print("P1 passed:", predictions)
            else:
                print("P1 checks are off. Complete the TODO, then set RUN_P1_CHECKS=True.")
            """
        ),
        md(
            r"""
            ## Problem 2 — 推导平方误差梯度

            **已知与目标：** 继续使用上方数据。现在把 `theta` 看作待调整的参数，
            `X` 和 `y` 是常量。对 $m$ 个样本定义损失

            $$J(\theta)=\frac{1}{2m}\lVert X\theta-y\rVert_2^2.$$

            这里 $\lVert r\rVert_2^2=\sum_{i=1}^m r_i^2$，$r=X\theta-y$ 是每个样本的预测误差。
            `J` 是一个数；梯度则为每个参数提供一个偏导数，不是一个数。

            **任务：** 先把损失写成样本求和的形式，对任意参数 $\theta_j$ 求偏导，
            再组合成向量表达式。标出 `X`、`theta`、`r` 和梯度的 shape。
            将推导写入答题框，然后完成 `squared_error_gradient(X, y, theta)`，返回 `(d,)`。
            本題无需更新参数。检查在 Problem 3 中进行。
            """
        ),
        answer_box("在这里写推导，不要只写最终公式。"),
        code(
            """
            def squared_error_gradient(X, y, theta):
                # Return dJ/dtheta with shape (n_features,).
                # START TODO
                raise NotImplementedError("Translate your gradient formula to NumPy")
                # END TODO
            """
        ),
        md(
            r"""
            ## Problem 3 — 用数值梯度检查手推公式

            **为什么做：** 推导看起来正确并不等于代码正确。数值梯度用很小的扰动检查每个参数。
            下面的检查代码已给出；你只需要完成 Problem 2。

            **方法：** 令 $e_j$ 为仅第 $j$ 个位置为 1 的向量，固定其他参数，只把
            $\theta_j$ 向两侧各移动 $\epsilon=10^{-6}$。用中心差分近似该偏导：

            $$\frac{\partial J}{\partial\theta_j}\approx
            \frac{J(\theta+\epsilon e_j)-J(\theta-\epsilon e_j)}{2\epsilon}.$$

            **要做什么：** 运行下格定义和 `RUN_P3_CHECKS=True` 的分支，比较 `analytic`
            （你的解析梯度）与 `numeric`（给定代码的数值梯度）。两者都应为 `(2,)`，
            并通过 `np.allclose(..., atol=1e-6)`。若失败，先核对损失中的 `1/2` 和样本数 `m`。
            无需另写优化器；这一步只检查当前参数位置的梯度。
            """
        ),
        code(
            """
            def squared_error_loss(X, y, theta):
                residual = X @ theta - y
                return 0.5 * np.mean(residual ** 2)

            def numerical_gradient(fn, theta, epsilon=1e-6):
                grad = np.zeros_like(theta, dtype=float)
                for j in range(len(theta)):
                    step = np.zeros_like(theta, dtype=float)
                    step[j] = epsilon
                    grad[j] = (fn(theta + step) - fn(theta - step)) / (2 * epsilon)
                return grad

            RUN_P3_CHECKS = False
            if RUN_P3_CHECKS:
                analytic = squared_error_gradient(X, y, theta)
                numeric = numerical_gradient(lambda t: squared_error_loss(X, y, t), theta)
                print("analytic:", analytic)
                print("numeric :", numeric)
                assert np.allclose(analytic, numeric, atol=1e-6)
                print("P3 passed")
            else:
                print("P3 checks are off. Complete P2, then set RUN_P3_CHECKS=True.")
            """
        ),
        md(
            """
            ## Checks

            - [ ] 我能在运算前写出 `X @ theta` 的 shape。
            - [ ] 我能解释为什么梯度必须和 `theta` 同 shape。
            - [ ] 我的解析梯度通过数值梯度检查。

            ## Next Steps

            完成后进入 PS1，把这些工具用于真正的回归实验。
            """
        ),
    ]
    return notebook("CS229 Simplified PS0 — Math and NumPy Review", "对应入门准备，不冒充 Spring 2026 私有 pset0。", cells, "ps0")


def build_ps1():
    cells = [
        md(
            r"""
            ## Goal

            从同一份回归数据出发，练习最小二乘的两种求解方法，再扩展模型：
            P1 正规方程与 P2 梯度下降对应线性回归基础；P3 是在此基础上的多项式特征扩展，
            不是要求你从 Lecture 2 中自行补出所有概念；P4 使用局部加权回归。

            ## Setup

            每行数据是一对 `(x_i, y_i)`：`x_i` 是一个数，`y_i` 是要预测的连续值。
            `x_train` 与 `y_train` 都为 `(60,)`；验证集和测试集各 40 条。

            - **train（训练集）：** 只用这部分数据估计模型参数 `theta`。
            - **valid（验证集）：** 把训练好的模型拿来预测，比较 degree 或 tau 等设置，不在这里重新拟合。
            - **test（测试集）：** 留待模型选择结束后的最终评估；本题集不要求使用它。

            先运行数据格观察训练散点。P1 提供的 `mse` 会在 P2–P4 复用；P3 不依赖 P2，
            P4 复用 P1 的 `add_intercept` 和 `mse`。从头执行函数定义后，再打开已完成题目的检查开关。
            """
        ),
        setup_cell("ps1_regression"),
        code(
            """
            def load_xy(filename):
                data = np.loadtxt(DATA_DIR / filename, delimiter=",", skiprows=1)
                return data[:, 0], data[:, 1]

            x_train, y_train = load_xy("train.csv")
            x_valid, y_valid = load_xy("valid.csv")
            x_test, y_test = load_xy("test.csv")

            print("train:", x_train.shape, y_train.shape)
            print("valid:", x_valid.shape, y_valid.shape)
            print("test :", x_test.shape, y_test.shape)

            plt.scatter(x_train, y_train, s=20, label="train")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Regression data")
            plt.legend()
            plt.show()
            """
        ),
        md(
            r"""
            ## Problem 1 — Normal equation

            **课堂连接：** Lecture 2，线性模型与最小二乘。

            **模型：** 用一条直线 $\hat y=\theta_0+\theta_1x$ 预测 `y`。
            把每个输入数 `x_i` 写成一行 `[1, x_i]`，这一步记作 $\phi(x_i)=[1,x_i]$。
            将所有行堆叠得到设计矩阵 $X\in\mathbb{R}^{m\times2}$；于是全部预测为 `X @ theta`。
            给定的 `add_intercept` 已经完成这一步，不需要再添加一次截距列。

            **先推导：** 从 $J(\theta)=\frac{1}{2m}\lVert X\theta-y\rVert^2$ 出发，
            令梯度为零，整理成关于 `theta` 的线性方程组，写在下方答题框。

            **再实现两个函数：**

            1. `fit_normal_equation(X, y)`：接收 `(m, d)` 和 `(m,)`，返回 `(d,)`。
               本题 `d=2`，但不要把 2 写死，P3 会传入更多列。使用 `np.linalg.solve(A, b)`
               求解你推导的方程；它求的是 `A @ theta = b`，不需要显式计算逆矩阵。
            2. `mse(y_true, y_pred)`：接收两个同长度的一维数组，返回均方误差
               $\mathrm{MSE}=\frac1m\sum_i(y_i-\hat y_i)^2$。注意 MSE 是 `J` 的两倍。

            **检查与交付：** 下格只在训练集拟合，再分别报告 train/valid MSE。
            基线是不看 `x`、对所有样本都预测训练集 `y` 均值的模型。
            本数据应通过验证 MSE 小于基线的断言；这并不表示直线已解释所有规律。
            """
        ),
        answer_box("写出从令梯度为零到 normal equation 的推导，并标注矩阵 shape。"),
        code(
            """
            def add_intercept(x):
                return np.column_stack([np.ones_like(x), x])

            def fit_normal_equation(X, y):
                # Fit theta without explicitly computing a matrix inverse.
                # START TODO
                raise NotImplementedError("Use np.linalg.solve")
                # END TODO

            def mse(y_true, y_pred):
                # START TODO
                raise NotImplementedError("Return mean squared error")
                # END TODO
            """
        ),
        code(
            """
            RUN_P1_CHECKS = False
            if RUN_P1_CHECKS:
                X_train = add_intercept(x_train)
                X_valid = add_intercept(x_valid)
                theta_ne = fit_normal_equation(X_train, y_train)
                valid_pred = X_valid @ theta_ne
                baseline = np.full_like(y_valid, y_train.mean())
                print("theta:", theta_ne)
                print("train MSE:", mse(y_train, X_train @ theta_ne))
                print("valid MSE:", mse(y_valid, valid_pred))
                print("baseline MSE:", mse(y_valid, baseline))
                assert theta_ne.shape == (2,)
                assert mse(y_valid, valid_pred) < mse(y_valid, baseline)
            else:
                print("P1 checks are off.")
            """
        ),
        md(
            r"""
            ## Problem 2 — Gradient descent

            **课堂连接：** Lecture 2，迭代优化。

            **目标：** 在和 P1 完全相同的训练数据、直线模型和损失 `J` 上，
            通过多次小步更新找到参数，比较它与正规方程求解是否一致。

            **先写更新式：** 在答题框写出 `J` 对参数的梯度和学习率为 $\alpha$ 时的一次更新。
            区分三个量：`predictions` 和 `errors` 都是 `(m,)`；用于画图的 loss 是一个数；
            梯度为 `(d,)`，必须和 `theta` 相同。不能用标量 loss 代替逐样本误差来计算梯度。

            **函数约定：** `fit_gradient_descent(X, y, learning_rate, steps)` 从零参数开始，
            每次用全部训练样本计算 `J` 的梯度，再更新一次参数。返回最终 `(d,)` 参数与一维 `history`。
            `learning_rate=0.05` 是每步的步长，`steps=1000` 是更新次数。

            **记录与比较：** 在第 0、10、20、… 次更新之前记录当前预测的训练 MSE，
            每 10 步向 `history` 追加一个数（1000 次更新时有 100 个记录点）。
            这里优化的是 `J=MSE/2`，图上显示的是 MSE；两者最小值位置相同。
            下格横轴因此使用 `np.arange(len(history)) * 10`，纵轴为训练 MSE。
            先运行 P1 检查格生成 `theta_ne`，再运行本题检查，比较最终参数是否接近。

            **合理预期：** 默认步长下 MSE 应下降并逐渐稳定；不要求降到 0。
            两种方法得到接近参数说明求解一致，不说明直线模型一定足够好。
            """
        ),
        answer_box("写出一次 gradient-descent update，包括学习率。"),
        code(
            """
            def fit_gradient_descent(X, y, learning_rate=0.05, steps=1000):
                theta = np.zeros(X.shape[1])
                history = []
                for step in range(steps):
                    # START TODO
                    raise NotImplementedError("Compute gradient and update theta")
                    # END TODO
                return theta, np.array(history)
            """
        ),
        code(
            """
            RUN_P2_CHECKS = False
            if RUN_P2_CHECKS:
                theta_gd, history = fit_gradient_descent(add_intercept(x_train), y_train)
                print("GD theta:", theta_gd)
                print("normal-equation theta:", theta_ne)
                assert len(history) > 1 and history[-1] < history[0]
                assert np.allclose(theta_gd, theta_ne, atol=0.1)
                plt.plot(np.arange(len(history)) * 10, history)
                plt.xlabel("step")
                plt.ylabel("training MSE")
                plt.title("Gradient-descent learning curve")
                plt.show()
            else:
                print("P2 checks are off. Run P1 first, then complete P2.")
            """
        ),
        md(
            r"""
            ## Problem 3 — Polynomial feature map

            **这题在做什么：** P1 只允许拟合直线。如果散点呈现弯曲趋势，继续训练同一条直线
            也不会让它变成曲线。本题保留最小二乘的训练方法，改变输入给模型的特征，
            比较最高次数为 1、3、10 的三个多项式模型。这是线性回归的扩展练习。

            **1. 先读懂符号：把一个数变成一行特征**

            $x$ 仍然是原始输入的一个数；`degree`（记作 $k$）表示多项式的最高次数。
            $\phi_k$（读作 phi，下标 k）是一个转换规则，称为特征映射：

            $$\phi_k(x)=[1,x,x^2,\ldots,x^k].$$

            例如 `degree=3` 时，输入 `x=2` 转换为 `[1, 2, 4, 8]`。
            第一项 `1` 对应截距，后面依次是原始输入的一次方、二次方、三次方。
            这些数都由同一个 `x` 算出，不是新的观测，也不是需要学习的参数。

            **2. 明确要拟合的模型**

            | degree | 一个样本的特征（代码写法） | 参数个数 |
            |---|---|---|
            | 1 | `[1, x]` | 2 |
            | 3 | `[1, x, x**2, x**3]` | 4 |
            | 10 | `[1, x, ..., x**10]` | 11 |

            三个模型的预测公式分别为：

            $$\begin{aligned}
            k=1:&\quad \hat y=\theta_0+\theta_1x,\\
            k=3:&\quad \hat y=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3,\\
            k=10:&\quad \hat y=\sum_{j=0}^{10}\theta_jx^j.
            \end{aligned}$$

            每个 degree 都要**单独训练一组参数**，只使用 `(x_train, y_train)`。
            模型可以对 `x` 呈曲线，但参数 `theta` 仍以一次方相加，所以仍可使用 P1 的最小二乘方法。

            **3. 你需要实现的部分：`polynomial_features`**

            输入 `x` 是包含 `m` 个数的一维数组 `(m,)`，`degree` 是非负整数。
            返回矩阵 `Phi`，shape 为 `(m, degree + 1)`：每行对应一个样本，
            第 `j` 列为该样本的 `x**j`（列从 0 开始数）。不用自己拟合参数，也不用再调用 `add_intercept`。

            用小例子检查接口：`x = np.array([2., 3.])`、`degree=2` 时，应返回

            ```text
            [[1., 2., 4.],
             [1., 3., 9.]]
            ```

            **4. 下面已给出的实验代码会做什么**

            1. 对每个 degree 分别生成 `Phi_train` 和 `Phi_valid`；训练集有 60 行，验证集有 40 行。
            2. 用 `fit_normal_equation(Phi_train, y_train)` 得到 `(degree + 1,)` 的 `theta_poly`。
            3. 用同一组参数计算 `Phi_train @ theta_poly` 和 `Phi_valid @ theta_poly`，
               分别与 `y_train`、`y_valid` 比较得到 train/valid MSE。验证集不能重新训练参数。
            4. 在训练输入范围内取 400 个按大小排列的 `grid` 点，用每个模型预测这些点的 `y`。
               **“三条曲线”指三个模型的预测曲线**：横轴是原始 `x`，纵轴是预测的 $\hat y$，
               每条线对应一个 degree；散点是训练数据。不是画三个幂函数，也不是画 loss 随迭代变化。

            **完成顺序：** 确保 P1 的 `fit_normal_equation` 和 `mse` 已定义并可用，填写本题唯一的
            函数 TODO，运行函数定义格，再将 `RUN_P3_CHECKS=True` 并运行检查格。
            实验循环、MSE 输出和绘图已经提供，不需要重写。

            **5. 需要交付和解释什么**

            保留三行 degree/train MSE/valid MSE 输出及一张含三条预测曲线的图。在答题框回答：

            - 哪个 degree 的验证 MSE 最低？按本次实验你会选择哪个？
            - 若简单模型训练、验证误差都较高，且曲线错过明显弯曲趋势，这是欠拟合的证据。
            - 若提高次数使训练误差下降、验证误差反而上升，或曲线出现数据不支持的剧烈摆动，
              才有理由怀疑过拟合。**不要预先认定 degree=10 一定过拟合**；证据不足时明确写出。

            高次幂还可能放大数值误差。若求解报错或曲线异常，先核对特征列和 shape；
            不要直接把数值问题解释成过拟合。比较未正则化模型时，训练 MSE 通常应随次数增加而不升高。
            """
        ),
        code(
            """
            def polynomial_features(x, degree):
                # Return shape (m, degree + 1), including the intercept column.
                # START TODO
                raise NotImplementedError("Build powers 0 through degree")
                # END TODO
            """
        ),
        code(
            """
            RUN_P3_CHECKS = False
            if RUN_P3_CHECKS:
                example = polynomial_features(np.array([2., 3.]), 2)
                assert example.shape == (2, 3)
                assert np.allclose(example, [[1., 2., 4.], [1., 3., 9.]])
                grid = np.linspace(x_train.min(), x_train.max(), 400)
                plt.figure(figsize=(9, 4))
                for degree in (1, 3, 10):
                    Phi_train = polynomial_features(x_train, degree)
                    Phi_valid = polynomial_features(x_valid, degree)
                    assert Phi_train.shape == (len(x_train), degree + 1)
                    assert Phi_valid.shape == (len(x_valid), degree + 1)
                    theta_poly = fit_normal_equation(Phi_train, y_train)
                    train_mse = mse(y_train, Phi_train @ theta_poly)
                    valid_mse = mse(y_valid, Phi_valid @ theta_poly)
                    print(f"degree={degree:2d} train={train_mse:.4f} valid={valid_mse:.4f}")
                    plt.plot(grid, polynomial_features(grid, degree) @ theta_poly, label=f"degree {degree}")
                plt.scatter(x_train, y_train, s=15, color="black", alpha=0.5, label="train data")
                plt.xlabel("x")
                plt.ylabel("y / predicted y")
                plt.legend()
                plt.title("Polynomial regression")
                plt.show()
            else:
                print("P3 checks are off. Complete P1 first.")
            """
        ),
        answer_box("列出三个 degree 的 train/valid MSE，选择验证 MSE 最低者；结合曲线说明欠拟合或过拟合的证据，证据不足也请说明。"),
        md(
            r"""
            ## Problem 4 — Locally weighted regression

            **课堂连接：** Lecture 3，LWR。

            **目标：** 预测某个位置 `q` 时，让离它近的训练样本更有影响。
            和 P1 的全局直线不同，本题为每个查询位置重新拟合一条局部直线，只取这条直线在 `q` 处的预测。

            **输入与输出：** `x_train`、`y_train` 为 `(m,)`，`x_query` 是待预测位置组成的 `(n,)` 数组，
            `tau` 是正数，控制邻域宽度。`predict_lwr` 返回 `(n,)`，不需要查询点的真实 `y`。

            **对 `x_query` 中的每个位置 `q`，依次做：**

            1. 构造训练设计矩阵 `X=add_intercept(x_train)`，每行仍为 `[1, x_i]`。
            2. 计算每条训练样本相对 `q` 的权重
               $w_i(q)=\exp(-(x_i-q)^2/(2\tau^2))$。距离越远，权重越小。
            3. 令 `W` 为对角线上放这些权重的 `(m,m)` 矩阵。对局部损失
               $J_q(\theta)=\frac12\sum_i w_i(q)(\theta_0+\theta_1x_i-y_i)^2$
               求梯度并令其为零，得到加权正规方程，用 `np.linalg.solve` 求局部 `(2,)` 参数。
            4. 用 `[1, q]` 与局部参数计算一个预测值，按查询点顺序存入输出。
               可以循环查询点；每换一个 `q` 都要重新算权重和参数。

            **实验与交付：** 下格分别用 `tau=0.1, 0.5, 2.0` 预测验证集并输出 MSE，
            再在 `grid` 上画三条预测曲线（横轴 `x`，纵轴预测 `y`）。
            在答题框写出加权正规方程，比较三组结果并选择验证误差最低的 tau。
            小 tau 往往更灵活，大 tau 往往更平滑；不要求本次结果严格遵循某个排名。
            """
        ),
        code(
            """
            def predict_lwr(x_train, y_train, x_query, tau):
                # Return one LWR prediction per query point.
                # START TODO
                raise NotImplementedError("Compute weights and solve one local model per query")
                # END TODO
            """
        ),
        code(
            """
            RUN_P4_CHECKS = False
            if RUN_P4_CHECKS:
                grid = np.linspace(x_train.min(), x_train.max(), 300)
                for tau in (0.1, 0.5, 2.0):
                    valid_pred = predict_lwr(x_train, y_train, x_valid, tau)
                    print(f"tau={tau:.1f} valid MSE={mse(y_valid, valid_pred):.4f}")
                    plt.plot(grid, predict_lwr(x_train, y_train, grid, tau), label=f"tau={tau}")
                plt.scatter(x_train, y_train, s=15, color="black", alpha=0.5)
                plt.xlabel("x")
                plt.ylabel("y / predicted y")
                plt.legend()
                plt.title("Locally weighted regression")
                plt.show()
            else:
                print("P4 checks are off.")
            """
        ),
        answer_box("写出加权正规方程；列出各 tau 的 valid MSE 和选择。结合曲线解释小邻域对局部噪声的敏感性，以及大邻域可能遗漏的变化。"),
        md(
            """
            ## Checks

            - [ ] 所有设计矩阵的第一列都是 1。
            - [ ] 没有显式计算矩阵逆；使用 `np.linalg.solve`。
            - [ ] gradient descent 与 normal equation 得到接近的参数。
            - [ ] 对 polynomial degree 和 LWR tau 的判断来自验证集，不来自测试集。

            ## Next Steps

            完成 PS1 后再进入分类；不要提前使用分类指标解释回归模型。
            """
        ),
    ]
    return notebook("CS229 Simplified PS1 — Regression Experiments", "参考 CS229 feature maps、gradient descent 与 LWR 作业题型。", cells, "ps1")


def build_ps2():
    cells = [
        md(
            r"""
            ## Goal

            对应 Spring 2026 Lecture 3–8。先在 logistic regression 的语境里正式引入 binary
            classifier 和 cross-entropy，再研究类别不平衡，最后实现一个两层神经网络。

            ## Setup

            `label` 为 0 或 1。正类样本较少，这是题目故意设置的实验条件。

            加载函数已给每行加入截距：`X_train` 为 `(200, 3)`，每行为 `[1, x1, x2]`，
            `y_train` 为 `(200,)`；验证集 100 行。不要再添加一列 1。
            训练集用于拟合，验证集用于比较；所有模型都用概率阈值 0.5 生成类别。
            P2 需要先运行 P1 检查格得到预测，P3 复用 P1–P2 函数；P4 使用另外给出的 XOR 数据。
            """
        ),
        setup_cell("ps2_classification"),
        code(
            """
            def load_classification(filename):
                data = np.loadtxt(DATA_DIR / filename, delimiter=",", skiprows=1)
                X = data[:, :2]
                y = data[:, 2].astype(int)
                X = np.column_stack([np.ones(len(X)), X])
                return X, y

            X_train, y_train = load_classification("train.csv")
            X_valid, y_valid = load_classification("valid.csv")
            print("train:", X_train.shape, y_train.shape, "positive rate:", y_train.mean())
            print("valid:", X_valid.shape, y_valid.shape, "positive rate:", y_valid.mean())
            """
        ),
        md(
            r"""
            ## Problem 1 — Logistic regression from scratch

            **课堂连接：** Lecture 3，binary classification 与 logistic regression。

            $$p(y=1\mid x;\theta)=\sigma(\theta^\top x),\qquad
            \sigma(z)=\frac{1}{1+e^{-z}}.$$

            **符号与目标：** 每个样本的三个输入（包括截距项）与参数 `theta` 内积得到分数 `z`，
            sigmoid 把分数转换成正类概率 `p`。`z=X @ theta` 和 `p` 都为 `(m,)`，
            `theta` 为 `(3,)`。概率不是类别；最后用 `p >= 0.5` 判为 1，否则判为 0。

            **先推导：** 令 $p_i=\sigma(\theta^\top x_i)$，从平均二元交叉熵

            $$L(\theta)=-\frac1m\sum_{i=1}^m[y_i\log p_i+(1-y_i)\log(1-p_i)]$$

            出发，对 `theta` 求梯度并标 shape。这里使用自然对数。

            **再实现三个函数，接口约定如下：**

            - `sigmoid(z)`：逐元素转换，返回与 `z` 同 shape 的值。需能处理正负极大数，
              如 `np.array([-1000., 0., 1000.])`；可按正负分支改写公式，避免直接计算过大的指数。
            - `binary_cross_entropy(y, probability, sample_weight=None)`：返回一个数。
              在取对数前把概率裁剪到 `[1e-12, 1-1e-12]`。
            - `fit_logistic_regression(...)`：从零参数开始，每步使用全部训练样本的损失梯度更新，
              每次更新前保存一个 loss。返回最终 `(d,)` 参数和长度为 `steps` 的 `history`。

            **为 P3 预留的权重规则：** `sample_weight=None` 表示每条样本权重为 1。
            若传入 `(m,)` 的非负权重 $w_i$（总和大于 0），损失定义为

            $$L_w=-\frac{\sum_i w_i[y_i\log p_i+(1-y_i)\log(1-p_i)]}{\sum_i w_i}.$$

            梯度必须与这个加权损失一致，分母同样是权重总和；不能只在 loss 中加权。
            先验证全 1 权重与 `None` 等价，再做 P3 的非均匀权重实验。

            **交付与检查：** 推导写入答题框；运行下格，查看参数、前五个验证概率和下降的 loss。
            此处的检查还不能判断少数类识别好不好，那是 P2 的任务。
            """
        ),
        answer_box("推导 binary cross-entropy gradient，并写出每个量的 shape。"),
        code(
            """
            def sigmoid(z):
                # START TODO
                raise NotImplementedError("Implement a numerically stable sigmoid")
                # END TODO

            def binary_cross_entropy(y, probability, sample_weight=None):
                # START TODO
                raise NotImplementedError("Clip probabilities and return weighted mean loss")
                # END TODO

            def fit_logistic_regression(X, y, sample_weight=None, learning_rate=0.1, steps=2000):
                theta = np.zeros(X.shape[1])
                history = []
                for step in range(steps):
                    # START TODO
                    raise NotImplementedError("Compute probabilities, gradient, and update")
                    # END TODO
                return theta, np.array(history)
            """
        ),
        code(
            """
            RUN_P1_CHECKS = False
            if RUN_P1_CHECKS:
                extreme = sigmoid(np.array([-1000., 0., 1000.]))
                assert np.all(np.isfinite(extreme))
                assert np.allclose(extreme, [0., 0.5, 1.])
                assert np.isfinite(binary_cross_entropy(np.array([1., 0.]), np.array([0., 1.])))
                theta_log, history = fit_logistic_regression(X_train, y_train)
                probability = sigmoid(X_valid @ theta_log)
                prediction = (probability >= 0.5).astype(int)
                print("theta:", theta_log)
                print("first probabilities:", probability[:5])
                assert theta_log.shape == (3,)
                assert np.all((probability >= 0) & (probability <= 1))
                assert history[-1] < history[0]
            else:
                print("P1 checks are off.")
            """
        ),
        md(
            r"""
            ## Problem 2 — 为什么 accuracy 会骗人

            **题型来源：** 新版公开作业中的 imbalanced classification 实验。

            **问题：** 验证集多数样本的标签为 0。如果把所有样本都猜成 0，总体正确率可能很高，
            但真正的正类一个也找不到。本题要把“整体正确”和“能识别正类”区分开。

            **函数输入：** `classification_metrics(y_true, y_pred)` 接收两个同长度 `(m,)`
            的 0/1 数组；`y_pred` 是类别，不是概率。约定 1 为正类、0 为负类。

            **需要返回的字典键与含义：**

            | 键 | 定义 |
            |---|---|
            | `TP` | 真实为 1，预测也为 1 的数量 |
            | `TN` | 真实为 0，预测也为 0 的数量 |
            | `FP` | 真实为 0，却预测为 1 的数量 |
            | `FN` | 真实为 1，却预测为 0 的数量 |
            | `accuracy` | `(TP+TN)/m` |
            | `precision` | `TP/(TP+FP)`：预测为正的样本中有多少是真的正类 |
            | `recall` | `TP/(TP+FN)`：真实正类中有多少被找到了 |
            | `specificity` | `TN/(TN+FP)`：真实负类中有多少被判对 |
            | `balanced_accuracy` | `(recall+specificity)/2`：两个类别的正确率各占一半 |
            | `f1` | `2*precision*recall/(precision+recall)` |

            本练习约定上述指标的分母为 0 时返回 `0.0`，避免全零预测导致 precision 报错。
            两个类别的样本数由 `y_true` 决定，不由模型预测决定。

            **操作与交付：** 完成函数后先运行小例子检查，再在同一验证集比较全零 baseline 和
            P1 的 `prediction`。在答题框引用两组的 accuracy、recall 和 balanced accuracy，
            解释为什么仅报告 accuracy 会遗漏少数类失败。
            """
        ),
        code(
            """
            def classification_metrics(y_true, y_pred):
                # Return a dictionary containing counts and requested metrics.
                # START TODO
                raise NotImplementedError("Compute confusion counts and metrics")
                # END TODO
            """
        ),
        code(
            """
            RUN_P2_CHECKS = False
            if RUN_P2_CHECKS:
                example = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
                assert all(example[key] == 1 for key in ("TP", "TN", "FP", "FN"))
                assert np.isclose(example["balanced_accuracy"], 0.5)
                all_zero = np.zeros_like(y_valid)
                print("all-zero baseline:", classification_metrics(y_valid, all_zero))
                print("logistic model  :", classification_metrics(y_valid, prediction))
            else:
                print("P2 checks are off. Complete and run P1 first.")
            """
        ),
        answer_box("引用你算出的数字，解释为什么 stakeholder 不能只看 accuracy。"),
        md(
            r"""
            ## Problem 3 — 给少数类更高权重

            **目标：** 让损失函数更重视训练集中较少的正类，观察模型判别结果如何改变。
            复用 P1 的加权训练函数及 P2 的指标函数，不修改验证集的标签或类别比例。

            **需要自己完成下格的实验：**

            1. 仅从 `y_train` 统计负类个数 `n0` 和正类个数 `n1`。
            2. 构造 `(m,)` 的 `sample_weight`：负类权重为 1，正类权重为 `n0/n1`。
               这样两类的总权重相等。权重控制的是训练损失，不是改变标签。
            3. 调用 `fit_logistic_regression` 重新从零训练，传入权重，保持普通模型的学习率和迭代次数。
            4. 在原来的 `X_valid` 上得到概率，再用同一个阈值 0.5 转成类别。
            5. 用 `classification_metrics` 并排输出普通模型与加权模型的指标。

            **比较口径：** 本数据的 majority（多数类）是 0，其类内正确率为 `specificity`；
            minority（少数类）是 1，其类内正确率为 `recall`。同时报告 `balanced_accuracy` 和 `precision`。
            少数类 recall 可能提高，多数类正确率或 precision 可能下降；按实际数字解释，不要求所有指标提高。
            在下格实验代码中用注释记录比较结果，以及哪类错误因此变多或变少。
            """
        ),
        code(
            """
            RUN_P3_CHECKS = False
            if RUN_P3_CHECKS:
                # START TODO
                raise NotImplementedError("Create weights, refit, and compare metrics")
                # END TODO
            else:
                print("P3 checks are off.")
            """
        ),
        md(
            r"""
            ## Problem 4 — 两层神经网络处理 XOR

            **课堂连接：** Lecture 7–8。线性决策边界无法解决 XOR，两层网络可以。

            **数据与任务：** XOR（异或）表示两个输入不同则标签为 1，相同则为 0。
            下格给出四种输入和标签。本题只要求网络拟合这四个点，不把训练准确率当作泛化结果。
            `X_xor` 为 `(4, 2)`，没有截距列；网络的偏置由 `b1`、`b2` 单独提供。

            **前向计算约定：** 网络为 `2 -> 4 -> 1`，分别指输入维数、隐藏单元数、输出单元数。

            $$Z_1=XW_1+b_1,\quad H=\tanh(Z_1),\quad Z_2=HW_2+b_2,\quad P=\sigma(Z_2).$$

            `W1` 为 `(2,4)`，`b1` 为 `(4,)`，`W2` 为 `(4,1)`，`b2` 为 `(1,)`。
            偏置沿样本行广播。先在纸上补出 `Z1`、`H`、`Z2`、`P` 的 shape，再推导平均二元交叉熵的梯度。

            **需要实现的函数：**

            - `mlp_forward(X, params)` 返回 `(probability, cache)`。`probability` 固定为 `(m,1)`；
              `cache` 使用字典，保存 `H` 和 `probability`，供反向传播复用。
            - `mlp_backward(X, y, params, cache)` 返回字典，键必须是 `W1`、`b1`、`W2`、`b2`，
              每个梯度的 shape 与对应参数完全相同。先把 `(m,)` 的 `y` 转为 `(m,1)` 再与概率相减，
              否则广播会错误地产生 `(m,m)`。损失按样本取平均，因此梯度也需要按样本数归一化。

            **实现提示与交付：** 对 `tanh` 求导时可复用隐藏层值，导数为 `1-H**2`。
            在函数旁用注释写出各中间量 shape 和反向链式求导步骤；训练循环和参数更新已经提供。
            运行 `RUN_P4_CHECKS`，记录四个概率，检查阈值 0.5 下四个标签均正确。
            若不收敛，先检查梯度平均、偏置求和轴和参数 shape。
            """
        ),
        code(
            """
            X_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
            y_xor = np.array([0., 1., 1., 0.])

            def init_mlp(rng):
                return {
                    "W1": rng.normal(scale=0.5, size=(2, 4)),
                    "b1": np.zeros(4),
                    "W2": rng.normal(scale=0.5, size=(4, 1)),
                    "b2": np.zeros(1),
                }

            def mlp_forward(X, params):
                # START TODO
                raise NotImplementedError("Return probability and a cache for backprop")
                # END TODO

            def mlp_backward(X, y, params, cache):
                # START TODO
                raise NotImplementedError("Return gradients matching every parameter shape")
                # END TODO
            """
        ),
        code(
            """
            RUN_P4_CHECKS = False
            if RUN_P4_CHECKS:
                params = init_mlp(np.random.default_rng(229))
                for step in range(5000):
                    probability, cache = mlp_forward(X_xor, params)
                    grads = mlp_backward(X_xor, y_xor, params, cache)
                    for name in params:
                        assert grads[name].shape == params[name].shape
                        params[name] -= 0.1 * grads[name]
                probability, _ = mlp_forward(X_xor, params)
                print("XOR probabilities:", probability.ravel())
                assert np.mean((probability.ravel() >= 0.5) == y_xor) == 1.0
            else:
                print("P4 checks are off.")
            """
        ),
        md(
            """
            ## Checks

            - [ ] 我能解释 classifier 为什么从 Lecture 3 才出现。
            - [ ] sigmoid 和 cross-entropy 在极端输入下不产生 NaN。
            - [ ] 我报告了 minority-class 指标，而不只报告 accuracy。
            - [ ] backprop 返回的每个梯度与对应参数 shape 相同。

            ## Next Steps

            完成后进入 PS3，处理没有标签或带隐变量的数据。
            """
        ),
    ]
    return notebook("CS229 Simplified PS2 — Classification and Neural Networks", "参考 CS229 logistic stability、imbalanced labels 与 simple-NN 作业题型。", cells, "ps2")


def build_ps3():
    cells = [
        md(
            r"""
            ## Goal

            对应 Spring 2026 Lecture 9–10。通过三个可见实验理解：K-means 如何压缩颜色、
            PCA 如何寻找主要方向、EM 如何在隐变量存在时交替更新。

            ## Setup

            数据和小图像都由构建脚本生成，不需要联网。

            P1 使用 `image`：shape 为 `(48,48,3)`，最后一维依次是 RGB 三个颜色通道，
            数值位于 `[0,1]`。P2 使用 `points`：180 个二维点，shape 为 `(180,2)`。
            P3 使用该题单独生成的一维 `x_gmm`。三题的数据不同，不需要把前一题的输出作为后一题输入。
            """
        ),
        setup_cell("ps3_unsupervised"),
        code(
            """
            points = np.loadtxt(DATA_DIR / "points.csv", delimiter=",", skiprows=1)
            image = np.load(DATA_DIR / "tiny_image.npy")
            print("points:", points.shape, "image:", image.shape)

            fig, axes = plt.subplots(1, 2, figsize=(9, 3))
            axes[0].scatter(points[:, 0], points[:, 1], s=12)
            axes[0].set_title("Unlabelled points")
            axes[1].imshow(image)
            axes[1].set_title("Original tiny image")
            axes[1].axis("off")
            plt.show()
            """
        ),
        md(
            r"""
            ## Problem 1 — K-means 与图像压缩

            **课堂连接：** Lecture 9。

            **目标：** 原图有许多不同颜色，我们只保留 4 个代表颜色。把一个像素的 `[R,G,B]`
            当作一个三维样本，在颜色空间聚类；这里不使用像素在图像中的位置。
            `image.reshape(-1,3)` 将图像变成 `(2304,3)` 的 `pixels`，下方检查格已完成这一步。

            **符号：** 输入 `X` 为 `(m,d)`；`k=4` 是簇的数量；`centroids` 为 `(k,d)`，
            每行是一个簇的中心；`assignment[i]` 为第 `i` 个样本所属簇的编号，取值 `0` 到 `k-1`。
            一个中心对应重建图像的一种颜色。

            **需要实现的三个函数：**

            1. `assign_clusters(X, centroids)`：计算样本到每个中心的平方欧氏距离，
               返回最近中心的整数编号数组 `(m,)`。距离相同取编号最小的中心。
            2. `update_centroids(X, assignment, k, old_centroids)`：对每个簇内的样本逐列取平均，
               返回 `(k,d)` 新中心。若某簇没有样本，保留该簇的旧中心，避免空数组均值产生 NaN。
            3. `fit_kmeans(X, k, rng, max_steps=50)`：用传入的 `rng` 从 `X` 中不放回抽取 `k` 行作初始中心，
               再交替分配样本和更新中心。运行固定 50 轮即可，也可在中心不再变化时提前停止。

            **损失与返回值：** 本题统一记录平均平方距离

            $$J=\frac1m\sum_i\lVert X_i-c_{a_i}\rVert_2^2,$$

            其中 $a_i$ 为样本的簇编号，$c_{a_i}$ 为对应中心。初始化后先分配一次并记录 `J`；
            每轮更新中心后重新分配，再记录 `J`。返回 `(centroids, assignment, history)`，
            分别为 `(k,d)`、`(m,)` 和一维损失数组。最终 assignment 必须对应最终中心。

            **检查与交付：** 下格用中心颜色替换像素，再恢复原图 shape，展示原图与 4 色图。
            保留对比图和不增加的 loss 记录，在答题框解释两步为何都不会使目标变大。
            这里“压缩”指减少颜色种类，不要求写压缩文件，也不保证当前 NumPy 文件体积变小。
            """
        ),
        code(
            """
            def assign_clusters(X, centroids):
                # START TODO
                raise NotImplementedError("Return nearest-centroid index for every row")
                # END TODO

            def update_centroids(X, assignment, k, old_centroids):
                # START TODO
                raise NotImplementedError("Return one mean vector per cluster; handle empty clusters")
                # END TODO

            def fit_kmeans(X, k, rng, max_steps=50):
                # START TODO
                raise NotImplementedError("Initialize and alternate assignment/update")
                # END TODO
            """
        ),
        code(
            """
            RUN_P1_CHECKS = False
            if RUN_P1_CHECKS:
                pixels = image.reshape(-1, 3)
                centroids, assignment, history = fit_kmeans(pixels, 4, np.random.default_rng(229))
                compressed = centroids[assignment].reshape(image.shape)
                print("loss history:", history)
                assert len(history) >= 2
                assert np.all(np.isfinite(history))
                assert compressed.shape == image.shape
                assert np.all(np.diff(history) <= 1e-10)
                fig, axes = plt.subplots(1, 2, figsize=(7, 3))
                axes[0].imshow(image); axes[0].set_title("original")
                axes[1].imshow(np.clip(compressed, 0, 1)); axes[1].set_title("4 colors")
                for ax in axes: ax.axis("off")
                plt.show()
            else:
                print("P1 checks are off.")
            """
        ),
        answer_box("解释为什么每次 assignment/update 都不会增加 K-means objective。"),
        md(
            r"""
            ## Problem 2 — PCA 降到一维再重建

            **课堂连接：** Lecture 10。

            **目标：** 每个点原来要用两个坐标表示；现在找到一个方向，使每个点只保留沿该方向的
            一个坐标。再用这一个坐标近似恢复原来的二维点，观察丢失了多少信息。

            **需要实现的函数与步骤：**

            1. `fit_pca(X, n_components=1)` 接收 `(m,d)` 数据，返回 `(mean, components)`。
               `mean` 是每列的均值，shape 为 `(d,)`。用它减去每一行，得到中心化数据 `X_centered`。
            2. 本题协方差统一使用 $C=X_{centered}^\top X_{centered}/m$，shape 为 `(d,d)`。
               用 `np.linalg.eigh(C)` 得到特征值与特征向量；特征向量在返回矩阵的**列**中，
               特征值默认从小到大排列。按从大到小选前 `n_components` 个对应方向。
               `components` 保持二维 shape `(d,n_components)`，不要把单列压成一维。
            3. `pca_transform(X, mean, components)`：先用同一个 `mean` 中心化，再沿所选方向投影，
               返回 `(m,n_components)` 的低维坐标 `Z`。
            4. `pca_inverse_transform(Z, mean, components)`：把低维坐标映射回原空间，再加回 `mean`，
               返回 `(m,d)`。利用 shape 判断这两步乘法是否需要转置。

            **本题的具体 shape：** `points` 为 `(180,2)`，`mean` 为 `(2,)`，
            `components` 为 `(2,1)`，`Z` 为 `(180,1)`，重建结果为 `(180,2)`。
            特征向量整体换号仍是同一个主方向，不应据此判错。

            **检查与交付：** 下格报告按全部坐标元素平均的重建 MSE，对比把每个点都重建为 `mean` 的基线，
            并画原始点与重建点。重建点应位于通过均值的一条直线上，误差低于均值基线。
            在函数旁用注释说明投影保留了哪个方向、丢失了哪个方向的信息。
            """
        ),
        code(
            """
            def fit_pca(X, n_components=1):
                # Return mean and principal directions sorted by decreasing eigenvalue.
                # START TODO
                raise NotImplementedError("Center, form covariance, and use np.linalg.eigh")
                # END TODO

            def pca_transform(X, mean, components):
                # START TODO
                raise NotImplementedError("Project centered data")
                # END TODO

            def pca_inverse_transform(Z, mean, components):
                # START TODO
                raise NotImplementedError("Reconstruct in original space")
                # END TODO
            """
        ),
        code(
            """
            RUN_P2_CHECKS = False
            if RUN_P2_CHECKS:
                mean, components = fit_pca(points, n_components=1)
                Z = pca_transform(points, mean, components)
                reconstructed = pca_inverse_transform(Z, mean, components)
                error = np.mean((points - reconstructed) ** 2)
                baseline = np.mean((points - mean) ** 2)
                print("components:", components.shape, "Z:", Z.shape)
                print("reconstruction MSE:", error, "baseline:", baseline)
                assert components.shape == (2, 1)
                assert Z.shape == (len(points), 1)
                assert error < baseline
                plt.figure(figsize=(6, 5))
                plt.scatter(points[:, 0], points[:, 1], s=12, alpha=0.5, label="original")
                plt.scatter(reconstructed[:, 0], reconstructed[:, 1], s=12, label="reconstructed")
                plt.xlabel("coordinate 1")
                plt.ylabel("coordinate 2")
                plt.axis("equal")
                plt.legend()
                plt.title("PCA: 2D to 1D and back")
                plt.show()
            else:
                print("P2 checks are off.")
            """
        ),
        md(
            r"""
            ## Problem 3 — 一维两成分 GMM 的 EM（选做）

            **课堂连接：** Lecture 9–10。

            **背景：** 假设每个观测数来自两个高斯分布之一，但不知道它来自哪一个。
            GMM（高斯混合模型）同时估计两组分布参数和每条观测属于各组的概率。
            EM 交替执行“根据当前参数估计归属”（E-step）和“根据归属更新参数”（M-step）。
            本题是选做，需先理解高斯密度与条件概率；不调用 sklearn。

            **数据与参数：** 下格的 `x_gmm` 为 `(200,)`，训练时不提供分组标签。
            `weights`、`means`、`variances` 都为 `(2,)`，第 `k` 项分别表示
            混合比例 $\pi_k$、均值 $\mu_k$、**方差** $\sigma_k^2$（不是标准差）。
            权重和为 1，方差必须为正。单个成分的密度为

            $$f_k(x_i)=\frac{1}{\sqrt{2\pi\sigma_k^2}}
            \exp\left(-\frac{(x_i-\mu_k)^2}{2\sigma_k^2}\right).$$

            **E-step 接口：** `gmm_e_step(x, weights, means, variances)` 返回 `(m,2)` 的
            `responsibilities`。元素 $r_{ik}$ 表示“已知观测 $x_i$ 后，它来自成分 $k$ 的概率”：

            $$r_{ik}=\frac{\pi_k f_k(x_i)}{\sum_{j=0}^1\pi_j f_j(x_i)}.$$

            两列分别对应两个成分；对每一行归一化，使其和为 1。它是软分配，可以是 0.3/0.7，
            不像 K-means 只返回一个簇编号。

            **M-step 接口：** `gmm_m_step(x, responsibilities)` 返回三个 `(2,)` 数组
            `(weights, means, variances)`。令 $N_k=\sum_i r_{ik}$，用它更新：

            - 混合比例为 $N_k/m$。
            - 均值是用 $r_{ik}$ 加权的 `x` 平均值，分母为 $N_k$。
            - 方差是用 $r_{ik}$ 加权的平方偏差平均值，偏差必须相对**本轮新均值**计算，分母仍为 $N_k$。

            **数值约定：** 更新方差时设下限 `1e-8`。给定初始化下两个成分都应有有效样本；
            如果 $N_k$ 为零或归一化分母为零，应明确报错并检查实现，不要给概率随意加常数后继续。
            更极端数据需要在对数域计算，本题不要求实现该扩展。

            **检查与交付：** 下格已提供 30 轮循环和对数似然
            $\ell=\sum_i\log(\sum_k\pi_k f_k(x_i))$ 的监测代码。
            完成两函数后打开检查，保留参数与似然变化输出。每行 responsibility 之和应为 1，
            参数有限、权重和为 1，似然应在数值容差内不下降。在函数旁用注释解释 soft responsibility 的含义。
            """
        ),
        code(
            """
            gmm_rng = np.random.default_rng(229)
            x_gmm = np.concatenate([
                gmm_rng.normal(-2.0, 0.5, size=80),
                gmm_rng.normal(2.0, 0.8, size=120),
            ])

            def gmm_e_step(x, weights, means, variances):
                # START TODO
                raise NotImplementedError("Return responsibilities with shape (m, k)")
                # END TODO

            def gmm_m_step(x, responsibilities):
                # START TODO
                raise NotImplementedError("Return updated weights, means, and variances")
                # END TODO
            """
        ),
        code(
            """
            RUN_P3_CHECKS = False
            if RUN_P3_CHECKS:
                def gmm_log_likelihood(x, weights, means, variances):
                    # Stable monitoring helper; the E/M updates remain your task.
                    log_density = -0.5 * (
                        np.log(2 * np.pi * variances)[None, :]
                        + (x[:, None] - means[None, :]) ** 2 / variances[None, :]
                    )
                    return np.logaddexp.reduce(np.log(weights)[None, :] + log_density, axis=1).sum()

                weights = np.array([0.5, 0.5])
                means = np.array([-1.0, 1.0])
                variances = np.array([1.0, 1.0])
                history = [gmm_log_likelihood(x_gmm, weights, means, variances)]
                for _ in range(30):
                    responsibilities = gmm_e_step(x_gmm, weights, means, variances)
                    assert responsibilities.shape == (len(x_gmm), 2)
                    assert np.all(np.isfinite(responsibilities))
                    assert np.all(responsibilities >= 0)
                    assert np.allclose(responsibilities.sum(axis=1), 1.0)
                    weights, means, variances = gmm_m_step(x_gmm, responsibilities)
                    assert all(a.shape == (2,) for a in (weights, means, variances))
                    assert all(np.all(np.isfinite(a)) for a in (weights, means, variances))
                    assert np.all(weights > 0) and np.isclose(weights.sum(), 1.0)
                    assert np.all(variances > 0)
                    history.append(gmm_log_likelihood(x_gmm, weights, means, variances))
                assert np.all(np.diff(history) >= -1e-8)
                print("log-likelihood:", history[0], "->", history[-1])
                print("weights:", weights)
                print("means:", means)
                print("variances:", variances)
            else:
                print("P3 checks are off. This problem is optional.")
            """
        ),
        md(
            """
            ## Checks

            - [ ] K-means objective 不随迭代增加。
            - [ ] PCA 的 components、投影和重建 shape 全部写对。
            - [ ] 我能解释 K-means hard assignment 与 GMM soft responsibility 的差别。

            ## Next Steps

            Spring 2026 的 diffusion、representation、LLM 和 RL 暂不伪装成官方 problem set。
            后续只在找到可靠课堂案例或明确标注“原创 lecture lab”时再添加。
            """
        ),
    ]
    return notebook("CS229 Simplified PS3 — Unsupervised Learning", "参考 CS229 K-means、PCA 与 semi-supervised EM 作业题型。", cells, "ps3")


def write_data():
    rng = np.random.default_rng(229)

    ps1 = OUT / "ps1_regression" / "data"
    ps1.mkdir(parents=True, exist_ok=True)
    for name, count in (("train", 60), ("valid", 40), ("test", 40)):
        x = np.sort(rng.uniform(-3.0, 3.0, size=count))
        y = 0.4 * x + np.sin(1.7 * x) + rng.normal(0, 0.18, size=count)
        np.savetxt(ps1 / f"{name}.csv", np.column_stack([x, y]), delimiter=",", header="x,y", comments="")

    ps2 = OUT / "ps2_classification" / "data"
    ps2.mkdir(parents=True, exist_ok=True)
    for name, n0, n1 in (("train", 180, 20), ("valid", 90, 10)):
        negative = rng.normal(loc=(-0.7, 0.0), scale=(1.0, 0.9), size=(n0, 2))
        positive = rng.normal(loc=(1.2, 1.0), scale=(0.8, 0.8), size=(n1, 2))
        X = np.vstack([negative, positive])
        y = np.concatenate([np.zeros(n0), np.ones(n1)])
        order = rng.permutation(len(y))
        values = np.column_stack([X[order], y[order]])
        np.savetxt(ps2 / f"{name}.csv", values, delimiter=",", header="x1,x2,label", comments="")

    ps3 = OUT / "ps3_unsupervised" / "data"
    ps3.mkdir(parents=True, exist_ok=True)
    points = np.vstack([
        rng.normal((-2.0, -1.0), (0.7, 0.35), size=(60, 2)),
        rng.normal((0.5, 2.0), (0.5, 0.8), size=(60, 2)),
        rng.normal((2.4, -0.3), (0.6, 0.45), size=(60, 2)),
    ])
    np.savetxt(ps3 / "points.csv", points, delimiter=",", header="x1,x2", comments="")

    image = np.zeros((48, 48, 3), dtype=float)
    image[:24, :24] = (0.95, 0.25, 0.20)
    image[:24, 24:] = (0.15, 0.65, 0.95)
    image[24:, :24] = (0.20, 0.80, 0.35)
    image[24:, 24:] = (0.95, 0.80, 0.15)
    gradient = np.linspace(-0.08, 0.08, 48)[:, None, None]
    image = np.clip(image + gradient + rng.normal(0, 0.025, image.shape), 0, 1)
    np.save(ps3 / "tiny_image.npy", image)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="overwrite generated starter notebooks")
    mode.add_argument("--refresh-layout", action="store_true", help="reorganize existing notebooks while preserving answers and implementations")
    args = parser.parse_args()

    if args.refresh_layout:
        for path in sorted(OUT.glob("*/*.ipynb")):
            current = nbf.read(path, as_version=4)
            slug = path.parent.name.split("_", 1)[0]
            updated = apply_layout(current, slug)
            nbf.validate(updated)
            if updated == current:
                print("UNCHANGED", path.relative_to(ROOT))
                continue
            nbf.write(updated, path)
            print("REFORMATTED", path.relative_to(ROOT))
        return

    write_data()
    specs = [
        ("ps0_foundations", "cs229_simplified_ps0_foundations.ipynb", build_ps0()),
        ("ps1_regression", "cs229_simplified_ps1_regression.ipynb", build_ps1()),
        ("ps2_classification", "cs229_simplified_ps2_classification.ipynb", build_ps2()),
        ("ps3_unsupervised", "cs229_simplified_ps3_unsupervised.ipynb", build_ps3()),
    ]
    for folder, filename, nb in specs:
        path = OUT / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.force:
            print("SKIP", path.relative_to(ROOT))
            continue
        nbf.write(nb, path)
        print("WROTE", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
