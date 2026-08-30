# CS229 Spring 2026 Course Map

主线资料：

- [Stanford Online 官方播放列表（Spring 2026）](https://www.youtube.com/playlist?list=PLaqpC4kq8Gpw)
- [CS229 Spring 2026 课程主页](https://cs229.stanford.edu/index.html-spr26)
- [CS229 2026 Main Notes](https://cs229.stanford.edu/main_notes.pdf)

每份 notebook 都是手动填空作业：给出目标、公式、实验任务、检查项和 `TODO`，不提供答案。

| Lecture | 官方/实际内容 | Main Notes | Practice notebook |
|---:|---|---|---|
| 1 | Introduction | Parts I–VI overview | [Lecture 01](lecture01/cs229_spring2026_lecture01_introduction_practice.ipynb) |
| 2 | Supervised Learning Setup | Ch. 1 Linear Regression | [Lecture 02](lecture02/cs229_spring2026_lecture02_supervised_learning_setup_practice.ipynb) |
| 3 | Weighted Least Squares; Logistic Regression; Newton's Method | Ch. 1.4, Ch. 2 | [Lecture 03](lecture03/cs229_spring2026_lecture03_weighted_least_squares_practice.ipynb) |
| 4 | Exponential Family; GLMs; Multiclass Classification | Ch. 2.3, Ch. 3 | [Lecture 04](lecture04/cs229_spring2026_lecture04_exponential_family_and_glm_classification_practice.ipynb) |
| 5 | Gaussian Discriminant Analysis; Naive Bayes | Ch. 4 | [Lecture 05](lecture05/cs229_spring2026_lecture05_gaussian_discriminant_analysis_practice.ipynb) |
| 6 | Dataset Split; Bias/Variance; Regularization; ML Advice | Ch. 8–9 | [Lecture 06](lecture06/cs229_spring2026_lecture06_dataset_split_and_ml_advice_practice.ipynb) |
| 7 | Neural Networks 1: Architecture | Ch. 7.1–7.3 | [Lecture 07](lecture07/cs229_spring2026_lecture07_neural_networks_1_architecture_practice.ipynb) |
| 8 | Neural Networks 2: Backpropagation | Ch. 7.4–7.5 | [Lecture 08](lecture08/cs229_spring2026_lecture08_neural_networks_2_backpropagation_practice.ipynb) |
| 9 | K-Means; GMM (non-EM) | Ch. 10, Ch. 11.1 | [Lecture 09](lecture09/cs229_spring2026_lecture09_k_means_and_gmm_non_em_view_practice.ipynb) |
| 10 | GMM with EM; PCA | Ch. 11–12 | [Lecture 10](lecture10/cs229_spring2026_lecture10_gmm_with_em_and_pca_practice.ipynb) |
| 11 | Diffusion Models | Ch. 14 | [Lecture 11](lecture11/cs229_spring2026_lecture11_diffusion_models_practice.ipynb) |
| 12 | Representation Learning; Foundation Models; LoRA | Ch. 14.4–16 | [Lecture 12](lecture12/cs229_spring2026_lecture12_representation_learning_practice.ipynb) |
| 13 | Semantic Retrieval; RAG; LLM Next-Token Loss | Ch. 16.3–17.2 | [Lecture 13](lecture13/cs229_spring2026_lecture13_llms_and_next_token_prediction_loss_practice.ipynb) |
| 14 | Transformers; In-Context Learning | Ch. 17.3, 17.6 | [Lecture 14](lecture14/cs229_spring2026_lecture14_transformers_and_in_context_learning_practice.ipynb) |
| 15 | Attention Variants; MoE; Prompting; SFT | Ch. 17.4–17.8 | [Lecture 15](lecture15/cs229_spring2026_lecture15_attention_variants_moe_and_sft_practice.ipynb) |
| 16 | RL Basics; Policy Gradient; REINFORCE | Ch. 19, 21.1 | [Lecture 16](lecture16/cs229_spring2026_lecture16_rl_basics_and_policy_gradient_practice.ipynb) |
| 17 | PPO; RLVR; Long-Chain Reasoning | Ch. 18, 21.2 | [Lecture 17](lecture17/cs229_spring2026_lecture17_ppo_rlvr_and_long_chain_reasoning_practice.ipynb) |

## 官方播放列表的标题错位

截至本仓库整理时，播放列表最后三个视频的 YouTube 标题存在编号/内容错位。这里按播放列表顺序和公开英文字幕中的实际授课内容整理：

- 第 15 个视频实际讲 Attention variants、MoE、zero/few-shot、SFT。
- 第 16 个视频实际讲 RL basics、policy gradient、REINFORCE。
- 第 17 个视频实际讲 PPO、RL for LLM、RLVR 和 long chain-of-thought reasoning。

每个对应 notebook 都保存了原始视频链接和这项来源说明。

## 建议使用方式

1. 先看对应 Lecture，并阅读列出的 Main Notes 章节。
2. 不运行代码，先在纸上写公式和形状。
3. 逐个填写 `TODO`，每完成一个步骤就做局部检查。
4. 完成 `Checks` 后再回看讲义。
5. 将自己的答案、结果图和总结作为单独 commit 提交。
