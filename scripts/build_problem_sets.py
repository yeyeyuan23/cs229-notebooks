#!/usr/bin/env python3
"""Build source-grounded, simplified CS229 practice problem sets."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import numpy as np


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

        **使用规则：**先手推，再写代码。不要先看学生 solution。每道题完成后才把对应的
        `RUN_*_CHECKS` 改成 `True`。检查不通过时，从 shape 和公式开始定位。
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
    return nb


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

            **为什么做：**从 Lecture 2 开始，每个模型都要同时处理很多样本。这里先确认
            $(5,2)@(2,)\rightarrow(5,)$ 的含义。

            1. 运行前，在纸上写出 `X @ theta` 的 shape。
            2. 实现 `linear_predictions`，不能使用 Python 循环。
            3. 用一句话解释输出向量中的一个元素代表什么。
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

            对 $m$ 个样本定义

            $$J(\theta)=\frac{1}{2m}\lVert X\theta-y\rVert_2^2.$$

            **任务：**从上式推导 $\nabla_\theta J(\theta)$。每一步标出 shape；最终结果必须与
            $\theta$ 同 shape。然后把公式翻译为 NumPy。
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
            """
            ## Problem 3 — 用数值梯度检查手推公式

            **为什么做：**推导看起来正确并不等于代码正确。数值梯度用很小的扰动检查每个参数。
            下面的检查代码已给出；你只需要完成 Problem 2。
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

            对应 Spring 2026 Lecture 2–3。你会从同一份回归数据出发，依次完成 normal
            equation、gradient descent、polynomial features 和 locally weighted regression。

            ## Setup

            数据已经分成 train/valid/test CSV。`x` 是输入，`y` 是连续目标。先运行下面的格子，
            看清数据 shape 和散点图，再开始推导。
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

            **课堂连接：**Lecture 2，线性模型与最小二乘。

            使用 $\phi(x)=[1,x]$，令 $X\in\mathbb{R}^{m\times2}$。先在纸上从
            $J(\theta)=\frac{1}{2m}\lVert X\theta-y\rVert^2$ 推导 normal equation，再实现函数。

            **预期：**返回 `(2,)` 的参数；训练和验证 MSE 都应明显小于直接预测均值的基线。
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

            **课堂连接：**Lecture 2，迭代优化。

            1. 写出平方误差对 $\theta$ 的梯度。
            2. 从全零参数开始更新。
            3. 每 100 步保存一次 loss。
            4. 将最终参数与 Problem 1 的 normal-equation 结果比较。

            **预期：**loss 整体下降；足够迭代后，两种方法的参数应接近。
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
                plt.plot(np.arange(len(history)) * 100, history)
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

            **题型来源：**CS229 problem sets 常用此实验展示模型容量与过拟合。

            实现 $\phi_k(x)=[1,x,x^2,\ldots,x^k]$，分别拟合 `degree = 1, 3, 10`。
            记录 train 和 valid MSE，并画出三条曲线。

            **需要回答：**哪个 degree 欠拟合？哪个可能过拟合？依据必须来自图和 MSE。
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
                grid = np.linspace(x_train.min(), x_train.max(), 400)
                for degree in (1, 3, 10):
                    Phi_train = polynomial_features(x_train, degree)
                    Phi_valid = polynomial_features(x_valid, degree)
                    theta_poly = fit_normal_equation(Phi_train, y_train)
                    train_mse = mse(y_train, Phi_train @ theta_poly)
                    valid_mse = mse(y_valid, Phi_valid @ theta_poly)
                    print(f"degree={degree:2d} train={train_mse:.4f} valid={valid_mse:.4f}")
                    plt.plot(grid, polynomial_features(grid, degree) @ theta_poly, label=f"degree {degree}")
                plt.scatter(x_train, y_train, s=15, color="black", alpha=0.5)
                plt.ylim(-3, 3)
                plt.legend()
                plt.title("Polynomial regression")
                plt.show()
            else:
                print("P3 checks are off. Complete P1 first.")
            """
        ),
        answer_box("结合曲线和 train/valid MSE，判断欠拟合与过拟合。"),
        md(
            r"""
            ## Problem 4 — Locally weighted regression

            **课堂连接：**Lecture 3，LWR。

            对每个查询点 $x$ 使用
            $w^{(i)}=\exp(-(x^{(i)}-x)^2/(2\tau^2))$，再解加权 normal equation。

            实现 `predict_lwr`，比较 $\tau\in\{0.1,0.5,2.0\}$ 的 validation MSE 和曲线。
            **预期：**很小的 $\tau$ 更弯曲，很大的 $\tau$ 更接近全局直线。
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
                plt.legend()
                plt.title("Locally weighted regression")
                plt.show()
            else:
                print("P4 checks are off.")
            """
        ),
        answer_box("解释 tau 如何改变 bias/variance；引用你实际得到的曲线和 valid MSE。"),
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

            **课堂连接：**Lecture 3，binary classification 与 logistic regression。

            $$p(y=1\mid x;\theta)=\sigma(\theta^\top x),\qquad
            \sigma(z)=\frac{1}{1+e^{-z}}.$$

            先推导平均 binary cross-entropy 对 $\theta$ 的梯度，再实现稳定 sigmoid、loss 和
            gradient descent。这里第一次正式使用 classifier；不是 Lecture 1 的内容。
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

            **题型来源：**新版公开作业中的 imbalanced classification 实验。

            实现 TP、TN、FP、FN、accuracy、balanced accuracy、precision、recall 和 F1。
            比较两个模型：全部预测为 0 的 baseline，以及 Problem 1 的 logistic regression。

            **需要回答：**哪个指标揭示了少数类完全没被识别？不要只抄定义，要引用实际数字。
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

            为正类样本设置更大的 `sample_weight`，重新训练 logistic regression。比较普通模型与
            加权模型的 majority/minority accuracy 和 balanced accuracy。

            **预期：**少数类 recall 通常提高，但多数类 accuracy 可能下降。这是 trade-off，
            不是“所有指标一起提高”。
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

            **课堂连接：**Lecture 7–8。线性决策边界无法解决 XOR，两层网络可以。

            网络结构固定为 `2 -> 4 -> 1`，隐藏层用 `tanh`，输出用 sigmoid。先在纸上标出
            $W_1,b_1,W_2,b_2$ 的 shape，再实现 forward 和 backward。
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
                params = init_mlp(rng)
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

            **课堂连接：**Lecture 9。

            把每个 RGB 像素当成三维样本。实现 centroid 初始化、assignment 和 centroid update，
            然后用 4 种颜色重建图像。

            **预期：**loss 在迭代中不增加；压缩图像 shape 与原图完全相同。
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
                centroids, assignment, history = fit_kmeans(pixels, 4, rng)
                compressed = centroids[assignment].reshape(image.shape)
                print("loss history:", history)
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

            **课堂连接：**Lecture 10。

            1. 对数据逐列中心化。
            2. 计算 covariance matrix。
            3. 用 `np.linalg.eigh` 找最大特征值对应方向。
            4. 投影到一维，再重建回二维。

            **预期：**主方向 shape 为 `(2, 1)`；重建误差小于“所有点都预测为均值”的基线。
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
            else:
                print("P2 checks are off.")
            """
        ),
        md(
            r"""
            ## Problem 3 — 一维两成分 GMM 的 EM（选做）

            **课堂连接：**Lecture 9–10。

            不调用 sklearn。实现 E-step 的 responsibility，以及 M-step 的 $\pi_k,\mu_k,\sigma_k^2$
            更新。为避免下溢，可以在概率中加很小的 epsilon。

            **预期：**每行 responsibility 之和为 1；log-likelihood 不应明显下降。
            """
        ),
        code(
            """
            x_gmm = np.concatenate([
                rng.normal(-2.0, 0.5, size=80),
                rng.normal(2.0, 0.8, size=120),
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
                weights = np.array([0.5, 0.5])
                means = np.array([-1.0, 1.0])
                variances = np.array([1.0, 1.0])
                for _ in range(30):
                    responsibilities = gmm_e_step(x_gmm, weights, means, variances)
                    assert np.allclose(responsibilities.sum(axis=1), 1.0)
                    weights, means, variances = gmm_m_step(x_gmm, responsibilities)
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
    parser.add_argument("--force", action="store_true", help="overwrite generated starter notebooks")
    args = parser.parse_args()

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
