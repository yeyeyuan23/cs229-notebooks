# CS229 Spring 2026 Notebooks

跟踪我自学 Stanford [CS229 Spring 2026](https://cs229.stanford.edu/index.html-spr26)（机器学习）的练习记录。

练习题目按照 [Spring 2026 官方视频](https://www.youtube.com/playlist?list=PLaqpC4kq8Gpw)和
[2026 Main Notes](https://cs229.stanford.edu/main_notes.pdf)逐讲布置，风格是
**手动填空 / 手打代码**：notebook 只给目标、公式、实验任务、检查项和 `# TODO`，不给答案。

## 公开学习资料

- [CS229 Spring 2026 课程主页](https://cs229.stanford.edu/index.html-spr26)：课程信息和官方入口。
- [CS229 Lecture Notes 2026](https://cs229.stanford.edu/main_notes.pdf)：Tengyu Ma 和 Andrew Ng 编写的官方主讲义，是本仓库 notebook 的主要文字参考。
- [CS229 官方讲义归档](https://cs229.stanford.edu/notes_archive/)：按主题拆分的经典讲义和补充材料。
- [CS229 Illustrated Cheatsheets](https://stanford.edu/~shervine/teaching/cs-229/)：监督学习、无监督学习、深度学习及实用技巧的图解速查表，适合复习。
- [Spring 2026 官方视频](https://www.youtube.com/playlist?list=PLaqpC4kq8Gpw)：与本仓库 Lecture 1–17 对应的课程视频。

推荐顺序：观看对应课程视频，阅读主讲义相关章节，再独立完成 notebook 中的 `# TODO`。逐讲对应关系见 [COURSE_MAP.md](COURSE_MAP.md)。

## 结构

仓库包含 Lecture 1–17 的完整作业，每讲一个文件夹、一个 notebook。完整对应关系见
[COURSE_MAP.md](COURSE_MAP.md)。

- `lecture01/`–`lecture06/`：监督学习、GLM、生成式分类、泛化与 ML advice
- `lecture07/`–`lecture08/`：神经网络架构与反向传播
- `lecture09/`–`lecture10/`：K-means、GMM/EM、PCA
- `lecture11/`–`lecture15/`：Diffusion、Representation Learning、LLM、Transformer、MoE、SFT
- `lecture16/`–`lecture17/`：Policy Gradient、PPO、RLVR

`scripts/build_notebooks.py` 可以从统一规格生成空白作业（默认跳过已存在的 notebook，保护已经填写的答案）；
`scripts/validate_notebooks.py` 会检查 notebook 结构、来源、`TODO` 模式，并在内存中逐本从头执行。

## 环境

```bash
pip install numpy matplotlib torch jupyter
```

## 验证空白作业

```bash
python scripts/validate_notebooks.py
```

空白 notebook 的代码单元只有 `TODO` 注释，因此在填写答案前也能安全地从头运行。

如需重建全部空白作业，可以显式运行：

```bash
python scripts/build_notebooks.py --force
```

`--force` 会覆盖现有 notebook；使用前应先提交自己的答案。
