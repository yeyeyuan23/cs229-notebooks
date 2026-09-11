# CS229 Spring 2026 — Simplified Problem Sets

这是一个跟随 Stanford CS229 Spring 2026 自学的练习仓库。现在的主线不是“每节课硬凑一份 notebook”，而是四套连贯的 problem set：先推必要公式，再用 NumPy 完成与课堂概念对应的小实验。

> 这些不是 Stanford 官方 Spring 2026 作业，也不包含答案。Spring 2026 课程主页把 problem set 放在需要 Stanford 身份登录的 Ed 上；本仓库只参考公开课程资料、Stanford 过去公开的作业形式，以及学生公开上传到 GitHub 的近年题型结构，然后重新编写更短、可独立完成的版本。

## 从哪里开始

| 题集 | 对应内容 | 主要任务 |
|---|---|---|
| [PS0 Foundations](problem_sets/ps0_foundations/cs229_simplified_ps0_foundations.ipynb) | 开课前/Linear Algebra 预备 | shape、向量化、平方误差梯度、数值梯度检查 |
| [PS1 Regression](problem_sets/ps1_regression/cs229_simplified_ps1_regression.ipynb) | Lecture 2–3 | normal equation、gradient descent、polynomial features、LWR |
| [PS2 Classification](problem_sets/ps2_classification/cs229_simplified_ps2_classification.ipynb) | Lecture 3–8 | logistic regression、类别不平衡、weighted loss、两层神经网络 |
| [PS3 Unsupervised](problem_sets/ps3_unsupervised/cs229_simplified_ps3_unsupervised.ipynb) | Lecture 9–10 | K-means 图像压缩、PCA、GMM/EM（选做） |

每道题说明背景、符号、函数输入输出、操作步骤和交付要求，并区分已经提供的实验代码与需要填写的 TODO。PS1 P3 另外用具体数字解释多项式特征，明确“三条曲线”是三个模型的预测曲线；模型是否过拟合要根据实际结果判断。

空白 starter 的 `RUN_..._CHECKS` 默认都是 `False`，因此可以从头运行。已填写的 notebook 可以保留开启的检查和学习记录。检查默认关闭时，成功运行只说明题目骨架可执行，不代表 TODO 已实现正确。

## 资料与出处

- [Spring 2026 课程主页](https://cs229.stanford.edu/index.html-spr26)：课程顺序与官方入口；页面说明课程文档仅向 Stanford affiliates 开放。
- [Spring 2026 官方视频](https://www.youtube.com/playlist?list=PLaqpC4kq8Gpw)：本仓库采用的课程版本。
- [2026 Main Notes](https://cs229.stanford.edu/main_notes.pdf)：公式与概念主线。
- [Stanford Summer 2020 公开作业](https://cs229.stanford.edu/summer2020/)：公开可访问的官方 pset、starter code 与数据，用来确认传统作业形式。
- [学生公开的近年 CS229 仓库](https://github.com/MDzimah/Stanford-University-CS229-Machine-Learning)：只用于核对近年 problem-set 的主题和文件结构；本仓库不复制学生答案，也不冒充 2026 官方题目。

更精确的“课程章节 → 题集”关系见 [COURSE_MAP.md](COURSE_MAP.md)。

## VS Code 环境

打开 notebook 后点击右上角 kernel，选择现有的 `dev (Python 3.11.14)`：

```text
/opt/homebrew/Caskroom/miniconda/base/envs/dev/bin/python
```

这四套练习只依赖 `numpy`、`matplotlib`、`jupyter`。不需要为了这批题选择 base Python 3.13，也不需要 PyTorch。

## 工作方式

1. 先看对应视频/讲义，不看答案。
2. 先在 markdown 答题框里推公式、标 shape。
3. 填写一个 `START TODO` 到 `END TODO` 区域。
4. 把紧邻的 `RUN_..._CHECKS` 改为 `True`，只检查这一题。
5. 最后重启 kernel 并 Run All，确认没有依赖之前残留的变量状态。

生成器和验证器：

```bash
python scripts/build_problem_sets.py --force
python scripts/validate_problem_sets.py
python scripts/validate_problem_sets.py --starters
```

`--force` 会覆盖 notebook 中已经填写的答案，只应在明确想恢复空白 starter 时使用。固定随机种子生成的 CSV/NPY 数据也会同步重建。

验证器默认在内存中运行当前 notebook，允许已有答案和输出，不回写执行状态；`--starters` 则验证生成器的空白模板，也不会覆盖学习进度。

## 目前的边界

Lecture 11–17（diffusion、representation learning、LLM、RL）在当前公开资料中没有找到可核验的 Spring 2026 官方配套作业。因此这里暂时不伪造“官方实验”；后续若添加，会明确标为本仓库原创 lecture lab。
