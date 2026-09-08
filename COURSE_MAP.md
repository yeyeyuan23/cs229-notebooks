# CS229 Spring 2026 Course Map

本表把 Spring 2026 的课堂顺序映射到仓库里的综合 problem sets。题集是重新编写的学习材料，不是官方 Spring 2026 作业。

| Lecture | 课堂主题 | 建议练习 | 说明 |
|---:|---|---|---|
| 1 | Introduction | [PS0](problem_sets/ps0_foundations/cs229_simplified_ps0_foundations.ipynb)（可选） | 第一讲是课程概览；只补后续所需的 NumPy/线代，不引入 classifier |
| 2 | Supervised Learning Setup; Linear Regression | [PS1](problem_sets/ps1_regression/cs229_simplified_ps1_regression.ipynb) P1–P3 | normal equation、gradient descent、polynomial features |
| 3 | Weighted Least Squares; Logistic Regression; Newton's Method | [PS1](problem_sets/ps1_regression/cs229_simplified_ps1_regression.ipynb) P4；[PS2](problem_sets/ps2_classification/cs229_simplified_ps2_classification.ipynb) P1 | 先做回归，再进入 binary classifier |
| 4 | Exponential Family; GLMs; Multiclass Classification | [PS2](problem_sets/ps2_classification/cs229_simplified_ps2_classification.ipynb) P1–P3 | 用 logistic loss 和不平衡分类理解 GLM 的实际行为 |
| 5 | GDA; Naive Bayes | 暂无 | 先完成 PS2；后续可增加生成式分类 lab |
| 6 | Dataset Split; Bias/Variance; Regularization; ML Advice | [PS1](problem_sets/ps1_regression/cs229_simplified_ps1_regression.ipynb) P3；[PS2](problem_sets/ps2_classification/cs229_simplified_ps2_classification.ipynb) P2–P3 | 用 validation MSE 和 balanced metrics 做模型判断 |
| 7–8 | Neural Networks; Backpropagation | [PS2](problem_sets/ps2_classification/cs229_simplified_ps2_classification.ipynb) P4 | 小型 XOR 网络，手写 forward/backward |
| 9 | K-means; GMM | [PS3](problem_sets/ps3_unsupervised/cs229_simplified_ps3_unsupervised.ipynb) P1、P3 | 聚类、图像压缩与 EM |
| 10 | GMM with EM; PCA | [PS3](problem_sets/ps3_unsupervised/cs229_simplified_ps3_unsupervised.ipynb) P2–P3 | PCA 投影/重建；一维 GMM 为选做 |
| 11–17 | Diffusion; Representation Learning; LLM; Transformer; RL | 暂无可核验配套题 | 先跟官方视频和 Main Notes；不把原创题伪装成官方作业 |

## 为什么改成 problem sets

CS229 的公开作业通常不是“每听一讲回答几个零散问题”，而是一套作业同时包含书面推导和编程实验。典型编程题会给训练/验证/测试数据、函数接口和报告要求，让学生实现算法、画图并解释结果。本仓库因此采用同样的学习闭环，但缩小数据和代码量，保证每题可以独立完成。

## 来源边界

- **课程版本：** [Spring 2026 页面](https://cs229.stanford.edu/index.html-spr26)、[官方视频](https://www.youtube.com/playlist?list=PLaqpC4kq8Gpw)、[2026 Main Notes](https://cs229.stanford.edu/main_notes.pdf)。
- **官方公开题型参考：** [Summer 2020 assignments](https://cs229.stanford.edu/summer2020/)。
- **近年结构参考：** [学生公开的 CS229 仓库](https://github.com/MDzimah/Stanford-University-CS229-Machine-Learning)。其内容属于学生上传材料，不等同于官方公开发布，更不等同于 Spring 2026 作业。
- **本仓库内容：** 题目表述、缩小后的数据、检查代码均为本仓库重新编写；不收录学生解答。
