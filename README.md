# CS229：从讲义到代码

以 **CS229 Lecture Notes，Tengyu Ma / Andrew Ng，2026-08-23** 为主线的中文代码练习。通过 NumPy 实现算法，再用 PyTorch 对照自动求导与训练过程。

当前提供 [第一章：线性回归](chapters/01_linear_regression/linear_regression.ipynb)。这些是讲义配套的自学实验，不是 Stanford 官方作业。

## 快速开始

使用 Python 3.11 或更新版本，在仓库根目录安装依赖并启动 JupyterLab：

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

打开 `chapters/01_linear_regression/linear_regression.ipynb`，选择安装了依赖的 Python kernel。也可以用支持 Jupyter 的编辑器打开。PyTorch 实验使用 CPU 与 `float64`，不需要 GPU。

## 练习内容

**样本与预测 → 损失与梯度 → 批量/随机梯度下降 → 正规方程 → 概率解释 → 多项式与局部回归。**

NumPy 语法嵌入对应任务；notebook 提供简短推导、公式、函数约定与检查。练习包括函数实现，以及拟合、预测、评价和作图的完整实验，无需填写推导答题框。

PyTorch 对照位于梯度下降和正规方程之后：保持数据、损失、初值、学习率与更新次数一致，比较手写梯度和自动求导，再比较参数、误差曲线与验证 MSE。

检查显示“待完成”时，表示相关函数尚未实现；其他错误会正常抛出。修改输入或学习率后重新运行，可检查实现是否适用于不同设置。重启 kernel 后 Run All 可检查对隐藏状态的依赖。

## 讲义与数据

- [CS229 官方讲义](https://cs229.stanford.edu/main_notes.pdf)：第一章的房价例子、LMS、正规方程、概率解释和局部加权回归。
- [Spring 2026 课程入口](https://cs229.stanford.edu/index.html-spr26)。
- [章节对应表](COURSE_MAP.md)：使用印刷页码；PDF 阅读器页码加 1。

房价输入取自讲义第 9 页展示的 **5 行记录**，价格为千美元。这不是完整的 47 条房屋数据，仅用于数组练习与优化方法对照，不能据此评价泛化。标准化统计量仅从训练输入计算。

合成曲线为 `y = 0.4*x + sin(1.7*x) + noise`。仓库包含训练集 60 条、验证集 40 条、测试集 40 条；训练集拟合参数，验证集选择模型设置，测试集留待选择结束后评价。

讲义使用半平方误差的总和，本练习使用半均方误差；最优参数相同，但梯度和适用学习率的尺度不同。

## 生成与验证

```bash
python scripts/build_chapters.py
python scripts/validate_chapters.py --require-torch
python scripts/validate_chapters.py --starters --require-torch
```

生成器默认保留已有 notebook，不重建数据。`--force` 会用空白模板覆盖当前 notebook。

验证脚本在内存中顺序执行 notebook，不回写文件。`--starters` 验证空白模板；模板能运行不代表练习已完成。`--require-torch` 要求所选 kernel 安装 PyTorch；可通过 `--kernel <名称>` 指定 Jupyter kernel。
