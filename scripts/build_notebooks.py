#!/usr/bin/env python3
"""Build the hand-typed practice notebooks for Stanford CS229 Spring 2026."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
COURSE_URL = "https://cs229.stanford.edu/index.html-spr26"
NOTES_URL = "https://cs229.stanford.edu/main_notes.pdf"
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLaqpC4kq8Gpw"


def slugify(value: str) -> str:
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def code_todo(prompt: str) -> str:
    lines = ["# TODO: " + prompt]
    return "\n".join(lines) + "\n"


def build_notebook(lecture: dict) -> nbf.NotebookNode:
    number = lecture["number"]
    title = lecture["title"]
    source_note = lecture.get("source_note")

    source_lines = [
        f"- **Official video:** [{title}]({lecture['video_url']})",
        f"- **Official playlist:** [Stanford CS229 Spring 2026]({PLAYLIST_URL})",
        f"- **Main notes:** [{lecture['notes']}]({NOTES_URL})",
        f"- **Course site:** [CS229 Spring 2026]({COURSE_URL})",
    ]
    if source_note:
        source_lines.append(f"- **Source note:** {source_note}")

    cells = [
        nbf.v4.new_markdown_cell(
            f"# CS229 Spring 2026 — Lecture {number}: {title}\n\n"
            "This is a **hand-typed practice notebook**. Read the prompt, predict shapes "
            "and outputs, then type the implementation yourself. Do not paste a solution.\n\n"
            + "\n".join(source_lines)
        ),
        nbf.v4.new_markdown_cell(
            "## Goal\n\n"
            + lecture["goal"].strip()
            + "\n\n### Working rules\n\n"
            "1. Use only the packages named in Setup.\n"
            "2. Before running a code cell, write down the expected shape of every array/tensor.\n"
            "3. Keep a fixed random seed whenever randomness is used.\n"
            "4. Complete the final checks without looking back."
        ),
        nbf.v4.new_markdown_cell(
            "## Setup\n\n"
            + lecture.get(
                "setup",
                "Use Python, NumPy, Matplotlib, and PyTorch where requested. Do not use `sklearn`.",
            )
        ),
        nbf.v4.new_code_cell(code_todo(lecture.get("setup_todo", "type the imports and set the random seed"))),
        nbf.v4.new_markdown_cell("## Steps"),
    ]

    for index, step in enumerate(lecture["steps"], start=1):
        heading, body, todo = step
        cells.append(nbf.v4.new_markdown_cell(f"### {index}. {heading}\n\n{body.strip()}"))
        if todo:
            cells.append(nbf.v4.new_code_cell(code_todo(todo)))

    cells.extend(
        [
            nbf.v4.new_markdown_cell("## Checks\n\n" + lecture["checks"].strip()),
            nbf.v4.new_markdown_cell("## Next Steps\n\n" + lecture["next_steps"].strip()),
        ]
    )

    # Stable cell IDs keep repeated builds byte-for-byte reproducible and avoid
    # meaningless notebook JSON churn in Git.
    for index, cell in enumerate(cells):
        cell["id"] = f"lecture{number:02d}-cell{index:02d}"

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.metadata["cs229"] = {
        "offering": "Spring 2026",
        "lecture": number,
        "video_url": lecture["video_url"],
        "main_notes": lecture["notes"],
        "mode": "hand-typed-practice-no-solutions",
    }
    return notebook


LECTURES = [
    {
        "number": 1,
        "title": "Introduction",
        "video_url": "https://www.youtube.com/watch?v=DATnpGoGhM8",
        "notes": "Course overview; Main Notes Parts I–VI",
        "goal": """
Build a concrete map of the course. By the end, you should be able to turn a vague
application into inputs, targets, a model, a loss, a data-collection process, and an
evaluation plan, and distinguish supervised, unsupervised, foundation-model, and
reinforcement-learning settings.
""",
        "setup": "Use the Python standard library and NumPy only. This lecture is mostly conceptual.",
        "setup_todo": "import NumPy and create a deterministic random generator",
        "steps": [
            (
                "Classify learning settings",
                """
For each scenario below, label it supervised learning, unsupervised learning,
representation/foundation-model learning, or reinforcement learning. Give one sentence of evidence.

1. Predict a house price from labeled sales.
2. Group news articles without labels.
3. Predict the next token in internet text.
4. Train a robot from a scalar reward after each trajectory.
5. Fine-tune a pretrained encoder on 500 labeled examples.
""",
                None,
            ),
            (
                "Write the learning pipeline",
                """
Choose one real problem you care about and fill in:

`raw data -> features x -> model h_theta -> prediction -> loss/feedback -> parameter update -> evaluation`

State which arrows happen during training and which happen during deployment.
""",
                None,
            ),
            (
                "Represent a tiny dataset",
                r"""
Create five examples with three features. Store the design matrix as
$X\in\mathbb{R}^{5\times3}$ and labels as $y\in\mathbb{R}^5$.
Print every shape and explain what one row and one column mean.
""",
                "create X and y, then print and explain their shapes",
            ),
            (
                "Separate parameters from data",
                r"""
Create a parameter vector $\theta\in\mathbb{R}^3$ and compute $X\theta$.
Write down which quantities are observed, which are chosen by you, and which are learned.
""",
                "create theta, compute X @ theta, and verify the output shape",
            ),
            (
                "Loss versus evaluation metric",
                """
For a binary classifier, compare cross-entropy, accuracy, precision, recall, and F1.
Which one supplies a training gradient? Which one should be reported to a stakeholder?
Construct a case where accuracy is misleading.
""",
                "construct an imbalanced label/prediction example and compute accuracy by hand",
            ),
            (
                "Design a data split",
                """
For your chosen problem, specify train, development, and test sets. Explain what would
count as leakage and whether random, temporal, grouped, or geographic splitting is appropriate.
""",
                None,
            ),
            (
                "Estimate scaling",
                r"""
Suppose a computation costs $O(nd)$ for $n$ examples and $d$ features. Make a small
table for three values of $n$ and $d$. Then explain why modern foundation-model training
forces us to care about data, parameter, memory, and compute scaling simultaneously.
""",
                "calculate several n*d operation estimates and display them in a bounded table",
            ),
        ],
        "checks": """
Without notes, answer:

1. What are the data, hypothesis, parameters, objective, and evaluation metric?
2. Why is a loss not automatically a good real-world metric?
3. What is the difference between fitting, validation, and final testing?
4. Why does reinforcement learning need interaction or generated trajectories?
5. Name one failure caused by bad problem formulation rather than a bad optimizer.
""",
        "next_steps": "Lecture 2 turns this pipeline into a concrete supervised-learning program using linear regression.",
    },
    {
        "number": 2,
        "title": "Supervised Learning Setup",
        "video_url": "https://www.youtube.com/watch?v=cmNIMjPYdgM",
        "notes": "Part I; Chapter 1 — Linear Regression",
        "goal": """
Derive and implement linear regression from the CS229 objective. Compare batch gradient
descent, stochastic gradient descent, mini-batch training, and the normal equations, then
reproduce the same computation with PyTorch autograd.
""",
        "setup": "Use only NumPy, Matplotlib, and PyTorch. Do not use `sklearn`.",
        "steps": [
            (
                "Build the design matrix",
                r"""
Use five points generated by $y=1+2x_1$. Add $x_0=1$, so
$X\in\mathbb{R}^{5\times2}$, $y\in\mathbb{R}^5$, and $\theta\in\mathbb{R}^2$.
""",
                "create X, y, and zero-initialized theta; print all shapes",
            ),
            (
                "Implement the hypothesis",
                r"""
Vectorize $h_\theta(X)=X\theta$. Change one coordinate of $\theta$ and predict which
part of the fitted line changes.
""",
                "compute y_hat = X @ theta and run the intercept/slope check",
            ),
            (
                "Implement the sum objective",
                r"""
Use $J(\theta)=\frac12\sum_{i=1}^n(h_\theta(x^{(i)})-y^{(i)})^2$.
Do not silently replace the sum with a mean.
""",
                "define loss_sum(X, y, theta)",
            ),
            (
                "Derive the batch gradient",
                r"""
Show on paper that $\nabla_\theta J(\theta)=X^T(X\theta-y)$ and explain why its shape
must match $\theta$.
""",
                "define batch_gradient(X, y, theta) without autograd",
            ),
            (
                "Run batch gradient descent",
                r"""
Apply $\theta\leftarrow\theta-\alpha\nabla J(\theta)$ for 100–500 updates.
Track and plot the objective; the solution should approach $[1,2]$.
""",
                "implement Batch GD, save losses, print theta, and plot the curve",
            ),
            (
                "Run classical SGD",
                r"""
Use one randomly sampled example for each parameter update, but evaluate the full training
objective after every update. Use a seeded generator.
""",
                "implement single-example SGD and plot its full-objective trajectory",
            ),
            (
                "Run shuffled mini-batch GD",
                """
Shuffle indices once per epoch, split into batches of size 2, and update once per batch.
Record the number of updates per epoch and compare with sampling with replacement.
""",
                "implement shuffled mini-batch GD and compare its curve with Batch GD and SGD",
            ),
            (
                "Solve the normal equations",
                r"""
Derive $X^TX\theta=X^Ty$. Use `solve` when possible and `pinv` as a robust comparison.
Do not form an explicit inverse just to solve a linear system.
""",
                "solve for theta with np.linalg.solve and np.linalg.pinv; compare both with GD",
            ),
            (
                "Recover the probabilistic interpretation",
                r"""
Assume $y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}$ with independent
$\epsilon^{(i)}\sim\mathcal{N}(0,\sigma^2)$. Derive why maximum likelihood minimizes
the squared-error sum. Write the derivation before coding.
""",
                "compute residuals and estimate sigma squared from the fitted model",
            ),
            (
                "Match the gradient with PyTorch",
                """
Create tensor versions of `X`, `y`, and `theta`. Let only `theta` require gradients.
Compute the same sum objective, call `backward`, and compare with the manual NumPy gradient.
""",
                "use torch.autograd and verify the gradient numerically",
            ),
            (
                "Train with PyTorch",
                """
First update a tensor manually inside `torch.no_grad()`, then train `torch.nn.Linear(1, 1)`
with `torch.optim.SGD`. Explain why gradients must be cleared.
""",
                "implement both the manual PyTorch loop and the nn.Linear optimizer loop",
            ),
            (
                "Compare batch sizes",
                """
Use batch sizes 1, 2, and 5. Compare updates per epoch, noise in the objective, and final
parameter estimates while keeping the number of epochs and random seed visible.
""",
                "run the controlled batch-size experiment and plot three labeled curves",
            ),
        ],
        "checks": r"""
Explain each expression and every shape:

$$h_\theta(x)=\theta^Tx,\qquad
J(\theta)=\frac12\sum_i(h_\theta(x^{(i)})-y^{(i)})^2,$$

$$\nabla J(\theta)=X^T(X\theta-y),\qquad
\theta_{t+1}=\theta_t-\alpha\nabla J(\theta_t).$$

Then distinguish example, batch, update/iteration, and epoch; explain when normal equations
are preferable and why SGD can be noisy but useful.
""",
        "next_steps": "Lecture 3 moves from global least squares to local weighting and from regression to classification.",
    },
    {
        "number": 3,
        "title": "Weighted Least Squares",
        "video_url": "https://www.youtube.com/watch?v=uJF_gL3jhxI",
        "notes": "Chapters 1.4 and 2 — Locally Weighted Regression and Logistic Regression",
        "goal": """
Implement locally weighted linear regression, logistic regression, and Newton's method.
Understand how a local weighting bandwidth changes bias/variance and why classification
uses a likelihood rather than ordinary squared error.
""",
        "setup": "Use NumPy, Matplotlib, and PyTorch only. Generate all toy data locally.",
        "steps": [
            (
                "Construct nonlinear regression data",
                r"""
Generate one-dimensional inputs and noisy targets from a nonlinear function. Keep
$X=[\mathbf{1},x]$ as a two-column design matrix.
""",
                "generate and plot a seeded nonlinear regression dataset",
            ),
            (
                "Build local weights",
                r"""
For query $x_q$, use $w^{(i)}=\exp(-(x^{(i)}-x_q)^2/(2\tau^2))$ and diagonal matrix $W$.
Check that nearby examples receive larger weights.
""",
                "implement local_weights(x_train, x_query, tau) and inspect several queries",
            ),
            (
                "Solve weighted least squares",
                r"""
Derive $\theta=(X^TWX)^{-1}X^TWy$, then implement it with a linear solve and a small
ridge term for numerical stability.
""",
                "implement locally_weighted_predict without explicitly inverting X.T @ W @ X",
            ),
            (
                "Run the bandwidth experiment",
                """
Plot predictions for at least three values of `tau`. Describe underfitting, overfitting,
and computation at prediction time.
""",
                "plot three LWR curves and write the bias/variance conclusion",
            ),
            (
                "Implement a stable sigmoid",
                r"""
For binary classification use $h_\theta(x)=\sigma(\theta^Tx)$. Test very large positive
and negative logits without overflow.
""",
                "define a numerically stable sigmoid and test extreme logits",
            ),
            (
                "Implement Bernoulli negative log-likelihood",
                r"""
Use $-\sum_i[y^{(i)}\log p^{(i)}+(1-y^{(i)})\log(1-p^{(i)})]$ with safe clipping.
Explain why the label space differs from linear regression.
""",
                "define binary_nll(X, y, theta) and test it on a tiny dataset",
            ),
            (
                "Derive the logistic gradient",
                r"""
Show that $\nabla_\theta\mathrm{NLL}=X^T(\sigma(X\theta)-y)$, then compare the analytic
gradient with finite differences.
""",
                "implement the analytic gradient and a central-difference gradient check",
            ),
            (
                "Fit logistic regression with GD",
                """
Create a two-feature binary dataset, add an intercept, optimize the NLL, and plot the
decision boundary. Track both loss and accuracy.
""",
                "train logistic regression from scratch and plot its decision boundary",
            ),
            (
                "Implement Newton's method",
                r"""
Use $H=X^TRX$, where $R_{ii}=p^{(i)}(1-p^{(i)})$, and update by solving
$H\Delta=\nabla\mathrm{NLL}$ followed by $\theta\leftarrow\theta-\Delta$.
""",
                "implement damped Newton updates and compare iteration counts with GD",
            ),
            (
                "Reproduce with PyTorch",
                """
Use a single linear layer that outputs logits and `binary_cross_entropy_with_logits`.
Compare learned probabilities and parameters with the NumPy implementation.
""",
                "train the equivalent PyTorch binary classifier without applying sigmoid twice",
            ),
        ],
        "checks": """
1. Why does small `tau` have low local bias and high variance?
2. Why should you solve a linear system instead of forming an inverse?
3. Why is a logit not a probability?
4. What makes Newton's method a second-order method?
5. Under what condition can the Newton system become ill-conditioned?
""",
        "next_steps": "Lecture 4 unifies linear and logistic regression through exponential families and GLMs.",
    },
    {
        "number": 4,
        "title": "Exponential Family and GLM Classification",
        "video_url": "https://www.youtube.com/watch?v=8gVi4Rk21Eg",
        "notes": "Chapters 2.3 and 3 — Multiclass Classification and Generalized Linear Models",
        "goal": """
Recognize exponential-family distributions, construct generalized linear models, and
implement stable multiclass softmax regression from first principles and in PyTorch.
""",
        "setup": "Use NumPy, Matplotlib, and PyTorch. Do not use a library classifier.",
        "steps": [
            (
                "Identify exponential-family pieces",
                r"""
Match Bernoulli, Gaussian with fixed variance, Poisson, and categorical distributions to
$p(y;\eta)=b(y)\exp(\eta^TT(y)-a(\eta))$. Record $T$, $\eta$, $a$, and the mean.
""",
                None,
            ),
            (
                "Verify the log-partition derivative",
                r"""
For Bernoulli and Poisson, differentiate $a(\eta)$ and verify that
$\mathbb{E}[T(y)]=a'(\eta)$ numerically over a grid of natural parameters.
""",
                "implement both mean functions and compare analytic derivatives with finite differences",
            ),
            (
                "State the GLM assumptions",
                """
For a chosen input `x`, write the random component, linear predictor, and response/link
relationship. Explain which assumptions produce ordinary least squares, logistic regression,
and Poisson regression.
""",
                None,
            ),
            (
                "Implement stable softmax",
                r"""
Given logits $z\in\mathbb{R}^{n\times K}$, subtract the row maximum before exponentiating.
Verify non-negativity and row sums of one, including extreme logits.
""",
                "define stable_softmax(logits) and test normal and extreme inputs",
            ),
            (
                "Encode multiclass labels",
                r"""
Convert integer labels to one-hot matrix $Y\in\{0,1\}^{n\times K}$. Verify that each row
contains exactly one positive entry.
""",
                "implement one_hot(y, num_classes) and validate bounds and shapes",
            ),
            (
                "Implement multiclass cross-entropy",
                r"""
For $P=\mathrm{softmax}(X\Theta)$, use $J(\Theta)=-\sum_{i,k}Y_{ik}\log P_{ik}$.
Keep the sum/mean convention explicit.
""",
                "define softmax_nll(X, y, Theta) with stable log probabilities",
            ),
            (
                "Derive the matrix gradient",
                r"""
Show that $\nabla_\Theta J=X^T(P-Y)$. Annotate the dimensions of all four matrices and
check the result with finite differences on a tiny problem.
""",
                "implement the gradient and run a small central-difference check",
            ),
            (
                "Train softmax regression",
                """
Generate three 2D classes, add an intercept, fit with gradient descent, and visualize the
decision regions. Report train accuracy and NLL separately.
""",
                "train multiclass softmax regression from scratch and plot decision regions",
            ),
            (
                "Add L2 regularization",
                r"""
Add $\frac\lambda2\|\Theta_{1:}\|_F^2$ without regularizing the intercept row. Compare
parameter norms and boundaries for several values of $\lambda$.
""",
                "add L2 regularization and run a controlled lambda experiment",
            ),
            (
                "Match PyTorch cross-entropy",
                """
Train `torch.nn.Linear` with `torch.nn.CrossEntropyLoss`. Pass raw logits, not softmax
probabilities, and compare the learned predictions with NumPy.
""",
                "implement the PyTorch multiclass model and verify its loss convention",
            ),
        ],
        "checks": """
1. What three design choices define a GLM?
2. Why is the log-partition function important?
3. Why does subtracting the maximum not change softmax probabilities?
4. What are the shapes of `X`, `Theta`, logits, probabilities, and the gradient?
5. Why should `CrossEntropyLoss` receive logits?
""",
        "next_steps": "Lecture 5 switches from discriminative conditional models to generative class-conditional models.",
    },
    {
        "number": 5,
        "title": "Gaussian Discriminant Analysis",
        "video_url": "https://www.youtube.com/watch?v=zRdE8A4UZes",
        "notes": "Chapter 4 — Generative Learning Algorithms",
        "goal": """
Estimate and use Gaussian discriminant analysis, compare it with logistic regression, and
implement Naive Bayes with Laplace smoothing for count-based text features.
""",
        "setup": "Use NumPy and Matplotlib. PyTorch is optional for the final comparison.",
        "steps": [
            (
                "Generate class-conditional data",
                r"""
Sample two classes from Gaussians with different means and a shared covariance. Keep true
$\phi$, $\mu_0$, $\mu_1$, and $\Sigma$ for later comparison.
""",
                "generate a seeded two-class Gaussian dataset and plot it",
            ),
            (
                "Implement the multivariate Gaussian log-density",
                r"""
Use a linear solve and `slogdet` to evaluate $\log\mathcal{N}(x;\mu,\Sigma)$ without an
explicit matrix inverse or determinant underflow.
""",
                "define multivariate_normal_logpdf(X, mean, covariance) with shape checks",
            ),
            (
                "Fit GDA by maximum likelihood",
                r"""
Estimate $\phi$, $\mu_0$, $\mu_1$, and the shared covariance $\Sigma$ from labeled data.
Be explicit about whether covariance normalization uses $n$ or $n-1$.
""",
                "implement fit_gda(X, y) and compare estimates with the generating parameters",
            ),
            (
                "Compute posterior class probabilities",
                r"""
Combine $\log p(x\mid y)$ and $\log p(y)$, normalize with log-sum-exp, and predict by the
larger posterior. Verify probabilities sum to one.
""",
                "implement gda_predict_proba and gda_predict",
            ),
            (
                "Visualize the GDA boundary",
                """
Evaluate predictions on a grid and plot the boundary. Explain why shared covariance gives
a linear boundary and class-specific covariance would give a quadratic boundary.
""",
                "plot GDA probabilities and the decision boundary",
            ),
            (
                "Compare generative and discriminative fits",
                """
Fit the logistic regression implementation from Lecture 3 on the same data. Compare
assumptions, sample efficiency, and robustness when the Gaussian assumption is wrong.
""",
                "run GDA and logistic regression on matched and misspecified datasets",
            ),
            (
                "Build count features for text",
                """
Create a tiny vocabulary and document-term count matrix for two classes. Keep tokenization
rules visible and include one word absent from a class during training.
""",
                "construct a small bag-of-words matrix and integer labels by hand",
            ),
            (
                "Fit multinomial Naive Bayes",
                r"""
Estimate class priors and token probabilities. Use log probabilities during prediction and
explain the conditional-independence assumption.
""",
                "implement multinomial Naive Bayes without sklearn",
            ),
            (
                "Apply Laplace smoothing",
                r"""
Use $\hat\phi_{k\mid y}=(N_{k,y}+\alpha)/(N_y+\alpha|V|)$. Compare predictions at
$\alpha=0$, $1$, and a large value.
""",
                "add smoothing and demonstrate the zero-probability failure it prevents",
            ),
            (
                "Sample from the fitted model",
                """
Generate synthetic token counts or feature vectors conditional on a sampled class. Explain
why the ability to model `p(x, y)` makes the model generative.
""",
                "sample several synthetic examples from either fitted generative model",
            ),
        ],
        "checks": """
1. What distributions does GDA model directly?
2. Why does shared covariance lead to a linear log-odds function?
3. When might logistic regression outperform GDA?
4. What independence assumption does Naive Bayes make?
5. Why does Laplace smoothing matter most for unseen events?
""",
        "next_steps": "Lecture 6 focuses on evaluation, generalization, regularization, and practical ML decisions.",
    },
    {
        "number": 6,
        "title": "Dataset Split and ML Advice",
        "video_url": "https://www.youtube.com/watch?v=llnEgyyuYkQ",
        "notes": "Chapters 8–9 — Generalization, Regularization, and Model Selection",
        "goal": """
Build an evidence-safe model-development loop: split data correctly, diagnose bias and
variance, select regularization only on validation data, and perform structured error analysis.
""",
        "setup": "Use NumPy and Matplotlib. Build split and evaluation logic yourself.",
        "steps": [
            (
                "Create deterministic splits",
                """
Generate a dataset and split it into train, development, and test subsets with a seeded
permutation. Assert disjoint indices and full coverage.
""",
                "implement split_indices and validate no overlap or missing examples",
            ),
            (
                "Choose the correct split strategy",
                """
For IID images, time series, multiple rows per patient, and geographically clustered data,
choose random, temporal, grouped, or geographic splitting and justify each choice.
""",
                None,
            ),
            (
                "Demonstrate leakage",
                """
Construct a preprocessing statistic using all data, then redo it using train data only.
Show how leakage can make validation look better even without changing model code.
""",
                "build a small normalization leakage experiment and compare both pipelines",
            ),
            (
                "Visualize bias and variance",
                """
Fit polynomial regression models of increasing degree using NumPy linear algebra. Plot train
and development error versus degree.
""",
                "implement polynomial features, fit several degrees, and plot both errors",
            ),
            (
                "Draw learning curves",
                """
Train the same model on increasing fractions of the training set. Diagnose high bias, high
variance, and insufficient data from the gap and level of the curves.
""",
                "compute seeded learning curves with repeated subsamples",
            ),
            (
                "Implement ridge regression",
                r"""
Minimize $\|X\theta-y\|_2^2+\lambda\|\theta_{1:}\|_2^2$ while excluding the intercept
from regularization. Compare parameter norms.
""",
                "solve ridge regression for a grid of lambdas",
            ),
            (
                "Select a hyperparameter safely",
                """
Choose `lambda` using development error only, lock it, and evaluate the test set exactly once.
Record all three errors and explain why selecting on test is invalid.
""",
                "implement the train-dev-select-test protocol and print a compact result table",
            ),
            (
                "Implement K-fold cross-validation",
                """
Build fold indices yourself. Ensure every example is validation exactly once and compare
the mean and standard deviation across folds.
""",
                "implement K-fold CV for ridge regression without sklearn",
            ),
            (
                "Perform error analysis",
                """
Create at least three meaningful error buckets. Report count and error rate per bucket rather
than reading random failures one by one.
""",
                "build a bounded error-analysis table from your model predictions",
            ),
            (
                "Audit distribution shift",
                """
Create a shifted test distribution and compare feature summaries and model error with the IID
test set. State what can and cannot be inferred from this toy experiment.
""",
                "simulate covariate shift, compare summaries, and measure the error change",
            ),
        ],
        "checks": """
1. Which decisions may use the development set? Which may use the test set?
2. What do high train and high development error suggest?
3. What does a large train-development gap suggest?
4. Why is preprocessing part of the fitted pipeline?
5. When is random splitting actively misleading?
6. Write a three-step plan for the largest observed error bucket.
""",
        "next_steps": "Lecture 7 replaces linear predictors with composable nonlinear neural-network modules.",
    },
    {
        "number": 7,
        "title": "Neural Networks 1 — Architecture",
        "video_url": "https://www.youtube.com/watch?v=fRM41w9jzQo",
        "notes": "Chapter 7.1–7.3 — Nonlinear Models, Neural Networks, and Modules",
        "goal": """
Construct neural networks as compositions of affine maps and nonlinear modules. Implement
a multilayer perceptron forward pass in NumPy, reason about dimensions and initialization,
and build the equivalent architecture with PyTorch.
""",
        "setup": "Use NumPy, Matplotlib, and PyTorch. Do not use a high-level training framework.",
        "steps": [
            (
                "Draw the computation graph",
                r"""
For a two-layer MLP, annotate every edge and shape in
$z^{[1]}=XW^{[1]}+b^{[1]}$, $a^{[1]}=\sigma(z^{[1]})$,
$z^{[2]}=a^{[1]}W^{[2]}+b^{[2]}$.
""",
                None,
            ),
            (
                "Implement activation functions",
                """
Implement sigmoid, tanh, ReLU, and GELU or a stated approximation. Plot function values and
local slopes on the same input grid and identify saturated regions.
""",
                "implement and plot four activation functions and their derivatives",
            ),
            (
                "Implement an affine module",
                r"""
Write a function that maps $X\in\mathbb{R}^{n\times d_{in}}$ to
$Z\in\mathbb{R}^{n\times d_{out}}$. Test broadcasting of the bias deliberately.
""",
                "define affine_forward(X, W, b) with assertions for all dimensions",
            ),
            (
                "Build a NumPy MLP forward pass",
                """
Compose affine -> ReLU -> affine for a three-class problem. Return both logits and a cache
containing the intermediate tensors that backpropagation will need.
""",
                "implement mlp_forward and print every cached shape",
            ),
            (
                "Count parameters",
                """
Derive a formula for the number of trainable scalars in an MLP. Check it on at least three
architectures and distinguish parameters from activations.
""",
                "write parameter_count(layer_sizes) and verify it against explicit arrays",
            ),
            (
                "Compare initialization scales",
                """
Propagate the same random batch through deep ReLU networks initialized with tiny Gaussian,
large Gaussian, and He initialization. Track activation mean and variance by layer.
""",
                "run the initialization experiment and plot layerwise activation variance",
            ),
            (
                "Create a PyTorch module",
                """
Implement the same MLP by subclassing `torch.nn.Module`. Print named parameters and verify
that the output shape and parameter count match NumPy.
""",
                "define a two-layer PyTorch MLP and run a forward-pass equivalence check",
            ),
            (
                "Inspect hidden representations",
                """
For a 2D toy classification dataset, plot the input and the hidden-layer representation before
training. State what a useful representation would need to change.
""",
                "create a toy batch, extract hidden activations, and plot them",
            ),
            (
                "Optional convolution shape drill",
                """
Compute output height/width and parameter count for several 2D convolutions. Then verify one
case with `torch.nn.Conv2d`; focus on shapes rather than building a full CNN.
""",
                "verify one hand-computed Conv2d output and parameter count",
            ),
        ],
        "checks": """
1. Why does stacking only affine layers remain an affine model?
2. Which axis is batch, feature, hidden unit, and class?
3. What is the difference between a parameter and an activation?
4. Why does initialization scale matter more as depth grows?
5. What does `forward` compute, and what does it not yet compute?
""",
        "next_steps": "Lecture 8 derives backpropagation so these architectures can learn.",
    },
    {
        "number": 8,
        "title": "Neural Networks 2 — Backpropagation",
        "video_url": "https://www.youtube.com/watch?v=ne2ngVAoMG8",
        "notes": "Chapter 7.4–7.5 — Backpropagation and Vectorization",
        "goal": """
Derive reverse-mode differentiation from local derivatives, implement backpropagation for a
two-layer MLP, gradient-check it, and reconcile the manual gradients with PyTorch autograd.
""",
        "setup": "Use NumPy and PyTorch. Keep all gradient checks tiny and deterministic.",
        "steps": [
            (
                "Backpropagate through a scalar graph",
                r"""
For $u=ab$, $v=u+c$, and $L=v^2$, compute the forward values and all reverse derivatives
by hand. Label the upstream and local derivative at each node.
""",
                "implement the scalar graph and compare manual derivatives with torch.autograd",
            ),
            (
                "Derive affine-layer gradients",
                r"""
Given $Z=XW+b$ and upstream $G=\partial L/\partial Z$, derive gradients for $X$, $W$, and
$b$. Annotate the sum over the batch dimension.
""",
                "implement affine_backward(G, X, W) and check every returned shape",
            ),
            (
                "Derive activation gradients",
                """
Implement backward rules for ReLU, sigmoid, and tanh. Test values at positive, negative, and
near-zero pre-activations and state the convention used by ReLU at zero.
""",
                "implement three activation backward functions and local tests",
            ),
            (
                "Implement softmax cross-entropy backward",
                r"""
Starting from logits and integer labels, show that the derivative is $(P-Y)/n$ for a mean
loss. Make the sum-versus-mean convention explicit.
""",
                "implement a stable softmax_cross_entropy_with_grad function",
            ),
            (
                "Backpropagate through a two-layer MLP",
                """
Reuse Lecture 7's cache. Return gradients with names matching the parameters and assert that
each gradient shape equals its parameter shape.
""",
                "implement mlp_backward for affine -> ReLU -> affine -> cross-entropy",
            ),
            (
                "Run finite-difference checks",
                r"""
Use central differences $(J(\theta+\epsilon)-J(\theta-\epsilon))/(2\epsilon)$ on every
parameter of a tiny network. Report relative error, not just absolute error.
""",
                "gradient-check W1, b1, W2, and b2 with a small epsilon sweep",
            ),
            (
                "Train the manual network",
                """
Fit a nonlinear 2D dataset using the NumPy forward/backward functions. Plot loss, accuracy,
and the final decision regions.
""",
                "write the complete NumPy training loop and visualize learning",
            ),
            (
                "Match PyTorch autograd",
                """
Copy the NumPy parameters into a PyTorch model, run one identical batch, and compare all
gradients before either optimizer updates the parameters.
""",
                "perform a one-step manual-versus-autograd gradient reconciliation",
            ),
            (
                "Diagnose gradient accumulation",
                """
Call `backward()` twice with and without clearing gradients. Explain the observed values and
why accumulation can be intentional in large-batch simulation.
""",
                "demonstrate accumulation, zeroing, and intentional micro-batch accumulation",
            ),
            (
                "Inspect vanishing and exploding gradients",
                """
Measure parameter-gradient norms in deeper sigmoid and ReLU networks. Change initialization
and depth one factor at a time.
""",
                "plot layerwise gradient norms for two activations and two initializations",
            ),
        ],
        "checks": """
1. What is an upstream gradient?
2. Why does the bias gradient sum across examples?
3. Why is reverse mode efficient for a scalar loss with many parameters?
4. What relative error would make you distrust a gradient implementation?
5. Why must the manual and autograd comparisons happen before an optimizer step?
""",
        "next_steps": "Lecture 9 leaves labeled prediction and studies structure in unlabeled data.",
    },
    {
        "number": 9,
        "title": "K-Means and GMM — Non-EM View",
        "video_url": "https://www.youtube.com/watch?v=bSmIGBCoffA",
        "notes": "Chapters 10 and 11.1 — K-Means and Gaussian Mixtures",
        "goal": """
Implement hard clustering with K-means, understand its objective and initialization failures,
then move to soft probabilistic cluster assignments under a Gaussian mixture model.
""",
        "setup": "Use NumPy and Matplotlib. Do not call a clustering library.",
        "steps": [
            (
                "Generate unlabeled clusters",
                """
Sample several 2D clusters with unequal sizes and spreads. Keep true labels only for plotting,
never for fitting.
""",
                "generate and plot a seeded unlabeled mixture dataset",
            ),
            (
                "Implement the assignment step",
                r"""
Compute all squared distances between $n$ points and $K$ centroids without a Python loop over
examples. Return cluster index $c^{(i)}$ for each point.
""",
                "implement assign_clusters(X, centroids) using broadcasting",
            ),
            (
                "Implement the centroid update",
                r"""
Compute $\mu_k$ as the mean of assigned points. Define and test a policy for an empty cluster.
""",
                "implement update_centroids(X, assignments, K) with an empty-cluster policy",
            ),
            (
                "Track the distortion objective",
                r"""
Use $J(c,\mu)=\sum_i\|x^{(i)}-\mu_{c^{(i)}}\|_2^2$. Run K-means and verify that the
objective never increases beyond numerical tolerance.
""",
                "implement K-means with objective history and a monotonicity assertion",
            ),
            (
                "Test initialization sensitivity",
                """
Run many random initializations, plot the distribution of final distortion, and save the best
run. Then implement the K-means++ seeding idea.
""",
                "compare random seeding with K-means++ over repeated runs",
            ),
            (
                "Stress-test K-means assumptions",
                """
Apply K-means to elongated, unequal-variance, and non-convex clusters. Describe which geometry
the squared Euclidean objective prefers.
""",
                "construct at least two failure datasets and visualize the assignments",
            ),
            (
                "Evaluate a Gaussian component",
                """
Reuse the stable multivariate Gaussian log-density from Lecture 5. Evaluate several component
densities and verify their shapes.
""",
                "compute component log densities for fixed means and covariances",
            ),
            (
                "Compute soft responsibilities",
                r"""
For fixed GMM parameters compute
$r_{ik}=p(z^{(i)}=k\mid x^{(i)})$ using Bayes' rule and log-sum-exp. Do not update parameters yet.
""",
                "implement gmm_responsibilities for fixed mixture parameters",
            ),
            (
                "Compare hard and soft assignments",
                """
Visualize maximum-responsibility labels and uncertainty near overlapping regions. Find points
for which K-means and GMM disagree and explain why.
""",
                "plot responsibility contours and inspect disagreement examples",
            ),
            (
                "Select K cautiously",
                """
Plot K-means distortion for several values of `K`. Explain why training distortion alone always
rewards more clusters and what additional evidence is needed.
""",
                "build an elbow plot and state its limitations",
            ),
        ],
        "checks": """
1. What objective does K-means optimize?
2. Why can K-means converge to different solutions?
3. What happens to an empty cluster?
4. What does a GMM responsibility mean probabilistically?
5. Which K-means assumptions are relaxed by a full-covariance GMM?
""",
        "next_steps": "Lecture 10 learns the GMM parameters with EM and derives PCA for dimensionality reduction.",
    },
    {
        "number": 10,
        "title": "GMM with EM and PCA",
        "video_url": "https://www.youtube.com/watch?v=sUS-eTa0l6s",
        "notes": "Chapters 11–12 — EM Algorithms and Principal Components Analysis",
        "goal": """
Implement expectation-maximization for Gaussian mixtures, monitor likelihood correctly, and
derive PCA both from covariance eigenvectors and from singular value decomposition.
""",
        "setup": "Use NumPy and Matplotlib. Keep datasets small enough to inspect visually.",
        "steps": [
            (
                "Implement log-sum-exp",
                """
Write a stable reduction over a chosen axis and compare it with the naive computation on
moderate and extreme inputs.
""",
                "implement logsumexp and verify it against safe test cases",
            ),
            (
                "Initialize a GMM",
                """
Initialize mixture weights, means, and positive-definite covariance matrices. State all shapes
and use K-means centroids or a seeded random choice.
""",
                "create deterministic GMM initialization with covariance regularization",
            ),
            (
                "Implement the E-step",
                r"""
Compute responsibilities $r_{ik}$ from current parameters in log space. Assert non-negativity
and that each row sums to one.
""",
                "implement the vectorized E-step and its invariants",
            ),
            (
                "Implement the M-step",
                r"""
Use $N_k=\sum_i r_{ik}$ to update $\pi_k$, $\mu_k$, and $\Sigma_k$. Add a small diagonal
term and explain its numerical role.
""",
                "implement the M-step with full covariance matrices",
            ),
            (
                "Run EM and audit likelihood",
                """
Alternate E and M steps until a stated convergence criterion. Compute observed-data log
likelihood after every full iteration and check it does not decrease materially.
""",
                "fit a GMM, plot log likelihood, and visualize final responsibilities",
            ),
            (
                "Test local optima and degeneracy",
                """
Run several seeds and compare final likelihoods. Construct or discuss a covariance-collapse
case and show how regularization changes it.
""",
                "compare repeated EM runs and one near-degenerate case",
            ),
            (
                "Center data for PCA",
                r"""
Compute the feature mean and centered matrix $X_c$. Demonstrate what goes wrong if PCA is run
without centering on a translated dataset.
""",
                "center a 2D/3D dataset and verify the centered feature means",
            ),
            (
                "Derive PCA from covariance",
                """
Compute covariance eigenpairs, sort them, and project onto the top components. Use `eigh` for
a symmetric matrix.
""",
                "implement fit_pca_eigh and transform_pca with explicit shapes",
            ),
            (
                "Derive PCA from SVD",
                r"""
Compute $X_c=U\Sigma V^T$ and show that the principal directions match covariance eigenvectors
up to sign. Compare eigenvalues with squared singular values.
""",
                "implement PCA via SVD and reconcile both methods numerically",
            ),
            (
                "Measure reconstruction and explained variance",
                """
For several retained dimensions, reconstruct the data and plot reconstruction error and
cumulative explained-variance ratio.
""",
                "run a dimensionality sweep with labeled plots",
            ),
            (
                "Compare clustering before and after PCA",
                """
Add noisy dimensions to clustered data, reduce with PCA, and compare GMM fit quality and runtime.
Do not claim improvement unless the measured outputs support it.
""",
                "run the PCA-plus-GMM experiment and report bounded comparisons",
            ),
        ],
        "checks": """
1. What hidden variable does a GMM introduce?
2. What does the E-step estimate and what does the M-step optimize?
3. Why is observed-data likelihood the right monitored quantity?
4. Why must PCA center the data?
5. How are covariance eigenvectors related to the right singular vectors?
6. What information is lost when dimensions are dropped?
""",
        "next_steps": "Lecture 11 turns gradual Gaussian noising and denoising into a modern generative model.",
    },
    {
        "number": 11,
        "title": "Diffusion Models",
        "video_url": "https://www.youtube.com/watch?v=dqUMCzWjZSI",
        "notes": "Chapter 14 — Diffusion Models",
        "goal": """
Implement the forward diffusion process, understand the reverse-model parameterization and
ELBO connection, train a tiny noise predictor, and sample from a toy denoising process.
""",
        "setup": "Use NumPy, Matplotlib, and PyTorch. Use a tiny 2D distribution rather than a large image model.",
        "steps": [
            (
                "Create a toy data distribution",
                """
Generate a 2D mixture such as a ring or several moons using only NumPy. Plot clean samples and
record the empirical mean and covariance.
""",
                "generate a seeded 2D distribution and display bounded summary statistics",
            ),
            (
                "Define a noise schedule",
                r"""
Choose $\beta_t\in(0,1)$ and compute $\alpha_t=1-\beta_t$ and
$\bar\alpha_t=\prod_{s=1}^t\alpha_s$. Plot all three schedules.
""",
                "construct and validate beta, alpha, and cumulative-alpha arrays",
            ),
            (
                "Sample the forward process directly",
                r"""
Use $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ with
$\epsilon\sim\mathcal{N}(0,I)$. Verify the shape and limiting behavior.
""",
                "implement q_sample(x0, t, noise) with batchwise timesteps",
            ),
            (
                "Visualize progressive noising",
                """
Plot the same clean batch at multiple timesteps. Track empirical covariance and explain why the
distribution approaches a standard Gaussian under a suitable schedule.
""",
                "visualize at least five timesteps and plot covariance diagnostics",
            ),
            (
                "Check the Markov and closed-form views",
                """
Sample `x_t` by repeatedly applying one-step transitions and directly from `x_0`. Compare
sample moments over many draws, not individual noisy samples.
""",
                "compare iterative and closed-form forward sampling statistically",
            ),
            (
                "Build a timestep-conditioned noise predictor",
                """
Create a small PyTorch MLP that receives `x_t` and a simple timestep embedding and predicts
the injected noise. Print every tensor shape.
""",
                "define a tiny timestep-conditioned epsilon model",
            ),
            (
                "Implement the simplified training loss",
                r"""
Sample $x_0$, $t$, and $\epsilon$, then minimize
$\mathbb{E}\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2$. Keep timestep sampling visible.
""",
                "write one training step and then a short deterministic toy training loop",
            ),
            (
                "Implement reverse sampling",
                """
Starting from Gaussian noise, apply the reverse mean update from the notes and inject the
appropriate variance except at the final step. Save intermediate samples.
""",
                "implement a toy reverse sampler and plot its trajectory through time",
            ),
            (
                "Trace the ELBO connection",
                """
On paper, label observed and latent variables in the diffusion chain and identify the KL terms
in the variational lower bound. Explain which distribution is fixed and which is learned.
""",
                None,
            ),
            (
                "Audit a failed sampler",
                """
Deliberately use a mismatched schedule or omit timestep conditioning. Record what changes in
the loss and samples, and distinguish an observed failure from a theoretical explanation.
""",
                "run one controlled ablation and compare plots side by side",
            ),
        ],
        "checks": r"""
1. What are $q(x_t\mid x_{t-1})$, $q(x_t\mid x_0)$, and $p_\theta(x_{t-1}\mid x_t)$?
2. Why is the forward process not learned?
3. What does the network predict in the epsilon parameterization?
4. Why must the model know the timestep?
5. Which part of your toy result demonstrates learning rather than merely adding noise?
""",
        "next_steps": "Lecture 12 studies reusable representations, adaptation, and parameter-efficient fine-tuning.",
    },
    {
        "number": 12,
        "title": "Representation Learning",
        "video_url": "https://www.youtube.com/watch?v=_kREM2UAiJ8",
        "notes": "Chapters 14.4–16 and 15 — Diffusion Wrap-up, Foundation Models, and Representation Learning",
        "goal": """
Finish the diffusion objective, then build and evaluate reusable representations through
pretraining, contrastive learning, linear probing, fine-tuning, and LoRA-style adaptation.
""",
        "setup": "Use NumPy, Matplotlib, and PyTorch. Use small synthetic or built-in toy data only.",
        "steps": [
            (
                "Finish the diffusion loss simplification",
                r"""
Starting from the relevant KL term, identify which quantities are constant with respect to
$\theta$ and explain why noise-prediction MSE is a practical training target.
""",
                None,
            ),
            (
                "Define a representation",
                r"""
Write a feature map $\phi_\theta(x)\in\mathbb{R}^d$ for a chosen raw input. Distinguish
the encoder, representation, task head, and final prediction.
""",
                "create a tiny encoder and classifier head with explicit output shapes",
            ),
            (
                "Pretrain and transfer",
                """
Create a source task and a related target task. Pretrain an encoder on the source, freeze it,
and train only a new linear head on a small target dataset.
""",
                "implement the pretrain-then-linear-probe experiment with fixed seeds",
            ),
            (
                "Compare linear probing and fine-tuning",
                """
Starting from identical pretrained weights, compare a frozen encoder with full fine-tuning.
Report trainable parameter counts, target-data size, and validation performance.
""",
                "run a controlled linear-probe versus fine-tune comparison",
            ),
            (
                "Measure embedding similarity",
                r"""
Implement cosine similarity and pairwise similarity matrix $S=ZZ^T$ after normalizing rows.
Check the diagonal and inspect same-class versus different-class values.
""",
                "compute and visualize a normalized embedding similarity matrix",
            ),
            (
                "Construct positive and negative pairs",
                """
Create two stochastic views of each example using simple augmentations. Define which pairs are
positive, which are negatives, and what accidental false negatives would mean.
""",
                "build a seeded paired-view batch and verify pair indices",
            ),
            (
                "Implement an InfoNCE-style loss",
                r"""
Use normalized embeddings, temperature $\tau$, and cross-entropy over pair similarities.
Explain how temperature changes the logits and gradients.
""",
                "implement a symmetric contrastive loss and test several temperatures",
            ),
            (
                "Train a tiny contrastive encoder",
                """
Train on paired toy views, then visualize embeddings before and after. Evaluate with a linear
probe rather than claiming quality from a visually attractive plot alone.
""",
                "train the encoder briefly and evaluate the frozen representation",
            ),
            (
                "Implement a LoRA layer",
                r"""
Replace $W$ with $W+\frac\alpha r BA$, where $A\in\mathbb{R}^{r\times d_{in}}$ and
$B\in\mathbb{R}^{d_{out}\times r}$. Freeze $W$ and train only $A,B$.
""",
                "define a LoRALinear module and verify which parameters require gradients",
            ),
            (
                "Count adaptation cost",
                """
Compare full fine-tuning and LoRA trainable parameters for several layer sizes and ranks.
Include the adapter storage needed for many downstream users.
""",
                "produce a compact parameter/storage comparison table",
            ),
        ],
        "checks": """
1. What makes a representation transferable rather than merely predictive on pretraining data?
2. What evidence distinguishes linear probing from fine-tuning?
3. Why normalize embeddings before cosine contrastive loss?
4. How do false negatives affect contrastive learning?
5. Why can LoRA reduce adaptation storage even when the base-model forward pass remains large?
""",
        "next_steps": "Lecture 13 applies representations to retrieval and introduces autoregressive language-model loss.",
    },
    {
        "number": 13,
        "title": "LLMs and Next-Token Prediction Loss",
        "video_url": "https://www.youtube.com/watch?v=lNTajqxxOn4",
        "notes": "Chapters 16.3–17.2 — Retrieval, RAG, Tokenization, and Autoregressive Models",
        "goal": """
Build a small semantic-retrieval pipeline, tokenize sequences into shifted input/target pairs,
implement next-token cross-entropy and perplexity, and train a tiny bigram language model.
""",
        "setup": "Use the Python standard library, NumPy, Matplotlib, and PyTorch. No external tokenizer or LLM API.",
        "steps": [
            (
                "Build a toy corpus",
                """
Write a small corpus with repeated structure and several short documents for retrieval. Keep a
separate held-out sequence and record the corpus version in the notebook.
""",
                "create the training text, held-out text, and retrieval documents",
            ),
            (
                "Implement tokenization",
                """
Implement character-level and whitespace-level tokenizers with explicit vocabulary, unknown,
and end-of-sequence handling. Compare vocabulary size and sequence length.
""",
                "build encode/decode functions for two tokenization schemes",
            ),
            (
                "Create next-token examples",
                r"""
For tokens $(x_1,\ldots,x_T)$, create inputs $(x_1,\ldots,x_{T-1})$ and targets
$(x_2,\ldots,x_T)$. Batch fixed-length windows and verify there is no off-by-one error.
""",
                "build deterministic context-target windows and print one decoded example",
            ),
            (
                "Implement stable token cross-entropy",
                r"""
Given logits of shape `(batch, time, vocab)`, compute mean negative log probability of target
tokens. Compare with `torch.nn.functional.cross_entropy` on flattened dimensions.
""",
                "implement token_cross_entropy and reconcile it with PyTorch",
            ),
            (
                "Compute perplexity",
                r"""
Use $\mathrm{PPL}=\exp(\text{mean NLL})$. Explain the units and why train perplexity alone
does not establish generalization.
""",
                "compute train and held-out perplexity for fixed toy predictions",
            ),
            (
                "Fit a count-based bigram model",
                """
Count next-token transitions with smoothing, normalize rows, and evaluate held-out NLL. Inspect
the most likely continuation of several tokens.
""",
                "implement and evaluate a smoothed count bigram language model",
            ),
            (
                "Fit a neural bigram model",
                """
Treat a trainable `(vocab, vocab)` matrix as next-token logits. Optimize it with PyTorch and
compare learned probabilities with the count model.
""",
                "train the neural bigram model and plot its loss curve",
            ),
            (
                "Sample autoregressively",
                """
Generate one token at a time. Compare greedy decoding, temperature sampling, and top-k sampling
while keeping the random generator fixed.
""",
                "implement three decoding strategies and display several short samples",
            ),
            (
                "Implement semantic retrieval",
                """
Create simple document and query embeddings, normalize them, compute cosine similarity, and
return top-k documents with scores. Make ties deterministic.
""",
                "implement top_k_retrieve and inspect success and failure queries",
            ),
            (
                "Sketch a RAG data path",
                """
Write the concrete pipeline `query -> embedding -> retrieved documents -> prompt/context ->
next-token model`. Mark which components are trained, frozen, indexed, or generated.
""",
                None,
            ),
            (
                "Audit memorization and leakage",
                """
Place a held-out sentence in the retrieval index but not language-model training data. Explain
why retrieval success is not the same as parametric memorization.
""",
                "construct the controlled retrieval-versus-generation example",
            ),
        ],
        "checks": """
1. Why are input and target token sequences shifted by one position?
2. What axes does token cross-entropy reduce?
3. What does perplexity measure and what does it not measure?
4. How do temperature and top-k change sampling?
5. What is stored in a retrieval index versus language-model parameters?
""",
        "next_steps": "Lecture 14 replaces the bigram context with transformer self-attention and studies in-context learning.",
    },
    {
        "number": 14,
        "title": "Transformers and In-Context Learning",
        "video_url": "https://www.youtube.com/watch?v=pwQ0l4hFCVI",
        "notes": "Chapter 17.3 and 17.6 — Transformer Architecture and In-Context Learning",
        "goal": """
Implement causal scaled dot-product attention, multi-head attention, and a small transformer
block while tracking every tensor shape, computational cost, and autoregressive constraint.
""",
        "setup": "Use NumPy for the first attention calculation and PyTorch for modules. Keep sequence lengths tiny.",
        "steps": [
            (
                "Project queries, keys, and values",
                r"""
For $X\in\mathbb{R}^{B\times T\times d_{model}}$, compute $Q=XW_Q$, $K=XW_K$,
and $V=XW_V$. Annotate batch, time, and feature axes.
""",
                "create a tiny batch and compute Q, K, V with explicit shape assertions",
            ),
            (
                "Implement scaled dot-product attention",
                r"""
Compute $\mathrm{softmax}(QK^T/\sqrt{d_k})V$. Explain why scaling is needed as $d_k$ grows
and verify attention weights sum to one over keys.
""",
                "implement single-head attention in NumPy and return weights plus outputs",
            ),
            (
                "Apply a causal mask",
                """
Construct the triangular mask for autoregressive prediction. Verify that changing a future token
cannot change an earlier output.
""",
                "add a causal mask and run the future-token invariance test",
            ),
            (
                "Split and merge multiple heads",
                """
Reshape projections into `(batch, heads, time, head_dim)`, attend independently, concatenate,
and project. Assert `d_model = heads * head_dim`.
""",
                "implement split_heads, multi_head_attention, and merge_heads",
            ),
            (
                "Add positional information",
                """
Implement sinusoidal positional encodings or a learned embedding table. Compare two sequences
with the same tokens in different orders.
""",
                "add positional encodings and demonstrate order sensitivity",
            ),
            (
                "Build a pre-norm transformer block",
                """
Compose layer norm, causal attention, residual connection, feed-forward network, and a second
residual connection. Print shapes after each sublayer.
""",
                "define a small PyTorch transformer block without using nn.Transformer",
            ),
            (
                "Reconcile with PyTorch attention",
                """
Copy tiny hand-built projection weights into a PyTorch attention module or implement the same
equations in PyTorch and compare outputs numerically.
""",
                "run a NumPy-versus-PyTorch attention equivalence check",
            ),
            (
                "Count time and memory cost",
                r"""
Calculate the size of $T\times T$ attention scores for several sequence lengths. Separate
training activation cost from autoregressive KV-cache cost.
""",
                "produce a compact sequence-length cost table",
            ),
            (
                "Simulate KV caching",
                """
During token-by-token generation, cache prior keys and values and compute only the newest query.
Compare outputs with recomputing the full prefix.
""",
                "implement a tiny cached decoding step and verify equality with full recomputation",
            ),
            (
                "Design an in-context learning prompt",
                """
Create zero-shot, one-shot, and few-shot prompts for the same toy mapping. Identify which tokens
are instructions, demonstrations, query, and predicted continuation. Do not update parameters.
""",
                "construct and print three prompt variants with token counts",
            ),
        ],
        "checks": """
1. What do Q, K, and V each do?
2. Which axis receives softmax?
3. Why is a causal mask required for next-token training?
4. What changes with more heads and what remains fixed?
5. Why does KV caching save computation but consume memory?
6. How is in-context learning different from fine-tuning?
""",
        "next_steps": "Lecture 15 studies attention efficiency, mixture-of-experts layers, prompting modes, and SFT.",
    },
    {
        "number": 15,
        "title": "Attention Variants, MoE, and SFT",
        "video_url": "https://www.youtube.com/watch?v=hHC-SF3utxg",
        "notes": "Chapters 17.4–17.8 — Attention Variants, MoE, Prompting, and SFT",
        "source_note": "This is playlist item 15. YouTube currently mislabels it as Lecture 16; the public transcript confirms the content used here.",
        "goal": """
Explore attention/system tradeoffs, implement grouped-query and sliding-window attention ideas,
route tokens through a toy mixture-of-experts layer, and construct a masked supervised
fine-tuning objective.
""",
        "setup": "Use NumPy and PyTorch. Keep all attention tensors small enough to inspect.",
        "steps": [
            (
                "Compare MHA, MQA, and GQA shapes",
                """
Write the query/key/value head counts for multi-head, multi-query, and grouped-query attention.
Calculate projection parameters and KV-cache elements for each.
""",
                "build a shape/parameter/cache table for several configurations",
            ),
            (
                "Implement grouped-query expansion",
                """
Create fewer key/value heads than query heads, repeat or map them to query-head groups, and
verify the final attention output shape.
""",
                "implement a tiny grouped-query attention forward pass",
            ),
            (
                "Build a sliding-window mask",
                """
Allow each position to attend only to a fixed number of previous positions. Visualize the mask
and compare the number of allowed edges with full causal attention.
""",
                "construct and plot full-causal and sliding-window masks",
            ),
            (
                "Measure the context tradeoff",
                """
Change a token outside the attention window and verify local outputs remain unchanged. Explain
what long-range information the architecture can lose.
""",
                "run an outside-window perturbation test",
            ),
            (
                "Route tokens through experts",
                r"""
For router logits $g(x)$, compute expert probabilities, select top-k experts, renormalize, and
combine expert outputs. Track token and expert axes.
""",
                "implement a toy top-k mixture-of-experts layer",
            ),
            (
                "Audit expert utilization",
                """
Count tokens routed to each expert and define a simple load-balancing penalty. Construct a
collapsed-router case and a balanced case.
""",
                "measure expert loads and compare collapsed versus balanced routing",
            ),
            (
                "Compare zero-shot and few-shot prompting",
                """
Write prompt templates for the same task with no examples and with demonstrations. Separate
formatting changes from parameter updates.
""",
                "create prompt records and count input/target tokens",
            ),
            (
                "Format SFT examples",
                """
Represent instruction, optional context, and response as token sequences. Create labels that
ignore padding and, if chosen, instruction tokens.
""",
                "build a padded SFT batch with an explicit loss mask",
            ),
            (
                "Implement masked SFT loss",
                """
Compute next-token cross-entropy only where the loss mask is active. Verify ignored positions
produce zero contribution and zero logit gradient.
""",
                "implement masked_sft_loss and its invariants",
            ),
            (
                "Separate adaptation choices",
                """
Make a comparison table for prompting, in-context learning, full SFT, and LoRA SFT: training
data, changed parameters, inference context cost, and stored artifacts.
""",
                None,
            ),
        ],
        "checks": """
1. What memory does MQA/GQA reduce?
2. What capability can sliding-window attention sacrifice?
3. Why can an MoE have many total parameters but limited compute per token?
4. What is router collapse?
5. Which tokens should contribute to SFT loss, and why must that choice be explicit?
6. Which adaptation methods change model weights?
""",
        "next_steps": "Lecture 16 introduces sequential decision-making and derives REINFORCE/policy gradient.",
    },
    {
        "number": 16,
        "title": "RL Basics and Policy Gradient",
        "video_url": "https://www.youtube.com/watch?v=xveNBYVTrqw",
        "notes": "Chapters 19 and 21.1 — Reinforcement Learning and REINFORCE",
        "source_note": "This is playlist item 16. YouTube currently gives it an incorrect GMM/PCA title; the public transcript begins the RL unit and derives policy gradient.",
        "goal": """
Represent sequential decision-making as an MDP, sample trajectories from a stochastic policy,
compute returns, derive the log-derivative policy-gradient estimator, and train a toy policy
with REINFORCE.
""",
        "setup": "Use NumPy, Matplotlib, and PyTorch. Implement a tiny environment locally; no Gym dependency.",
        "steps": [
            (
                "Specify a finite MDP",
                r"""
Define states $\mathcal{S}$, actions $\mathcal{A}$, transition probabilities $P$, reward $R$,
initial-state distribution, horizon $T$, and discount $\gamma$ for a small line world.
""",
                "encode the line-world transition and reward tables with validation assertions",
            ),
            (
                "Roll out a fixed policy",
                """
Given a table of action probabilities, sample complete trajectories and record state, action,
reward, and action log-probability at each step.
""",
                "implement rollout(policy, rng) and print one bounded trajectory",
            ),
            (
                "Compute returns and return-to-go",
                r"""
Compute total discounted return and $G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k$. Test by hand on
a short known reward sequence.
""",
                "implement discounted_returns and verify against a manual example",
            ),
            (
                "Parameterize a stochastic policy",
                """
Use state-dependent logits followed by softmax. Sample actions and return differentiable log
probabilities for the chosen actions.
""",
                "define a small PyTorch categorical policy and inspect its probabilities",
            ),
            (
                "Derive the score-function identity",
                r"""
Show on paper how $\nabla_\theta\mathbb{E}_{\tau\sim p_\theta}[R(\tau)]$ becomes
$\mathbb{E}[R(\tau)\nabla_\theta\log p_\theta(\tau)]$. Identify which dynamics terms drop out.
""",
                None,
            ),
            (
                "Implement REINFORCE loss",
                r"""
Use $L=-\sum_t G_t\log\pi_\theta(a_t\mid s_t)$ so gradient descent performs reward ascent.
Verify the sign using a one-state, two-action example.
""",
                "implement reinforce_loss and run the one-state sign check",
            ),
            (
                "Train the line-world policy",
                """
Collect trajectories, update policy parameters, and track average return over a moving window.
Keep evaluation episodes separate from training updates.
""",
                "train with REINFORCE and plot return plus final action probabilities",
            ),
            (
                "Measure estimator variance",
                """
At fixed policy parameters, estimate gradients from many individual trajectories and from
larger trajectory batches. Compare mean and componentwise variance.
""",
                "sample gradient estimates at several batch sizes and plot variance",
            ),
            (
                "Add a constant baseline",
                r"""
Replace $G_t$ with $G_t-b$. Demonstrate empirically that a baseline can reduce variance without
systematically changing the expected gradient when used correctly.
""",
                "compare no baseline with a running-mean baseline",
            ),
            (
                "Discuss exploration",
                """
Vary policy entropy or initialization and observe whether the agent prematurely commits to one
action. Separate this observed behavior from the broader exploration/exploitation theory.
""",
                "run one controlled low-entropy versus high-entropy initialization experiment",
            ),
        ],
        "checks": """
1. What makes a decision problem sequential?
2. What information does a reward provide that an action label does not?
3. Why is the environment transition probability absent from the policy gradient?
4. Why is REINFORCE an on-policy Monte Carlo estimator?
5. Why can a baseline reduce variance without introducing bias?
""",
        "next_steps": "Lecture 17 improves policy updates with advantages and PPO, then applies RL to verifiable LLM reasoning.",
    },
    {
        "number": 17,
        "title": "PPO, RLVR, and Long-Chain Reasoning",
        "video_url": "https://www.youtube.com/watch?v=J7CossjMvEg",
        "notes": "Chapters 18 and 21.2 — Reasoning in LLMs, RLVR, and PPO",
        "source_note": "This is playlist item 17 and the final lecture. YouTube currently gives it an incorrect GMM/PCA title; the public transcript covers PPO and RL for verifiable LLM reasoning.",
        "goal": """
Turn REINFORCE into an advantage-based update, implement PPO's clipped surrogate objective,
monitor policy drift, and model the generation/reward/training pipeline used by RL with
verifiable rewards for long-chain reasoning.
""",
        "setup": "Use NumPy, Matplotlib, and PyTorch. Reuse the tiny policy and environment from Lecture 16.",
        "steps": [
            (
                "Reproduce the policy-gradient estimator",
                """
Load or recreate the Lecture 16 toy policy. For a fixed trajectory batch, save old action log
probabilities before any optimizer step.
""",
                "collect a fixed on-policy batch with states, actions, returns, and old log probabilities",
            ),
            (
                "Estimate a value baseline",
                r"""
Fit a small value function $V_\phi(s)$ to observed returns. Keep policy and value parameters
separate and report value prediction error.
""",
                "train a tiny value network on the fixed rollout batch",
            ),
            (
                "Compute advantages",
                r"""
Start with $\hat A_t=G_t-V_\phi(s_t)$. Normalize advantages within the batch and inspect the
effect on scale and sign.
""",
                "compute raw and normalized advantages with assertions",
            ),
            (
                "Compute importance ratios",
                r"""
Use $r_t(\theta)=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{old}(a_t\mid s_t))$.
Verify every ratio equals one before the first update.
""",
                "implement probability_ratio and its pre-update identity check",
            ),
            (
                "Implement the PPO clipped objective",
                r"""
Implement $L^{clip}=\mathbb{E}[\min(r_t\hat A_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$.
Test positive and negative advantages separately.
""",
                "implement ppo_policy_loss and unit-test clipping cases by hand",
            ),
            (
                "Run multiple PPO epochs",
                """
Reuse the same rollout batch for several minibatch epochs. Track policy loss, value loss,
entropy, approximate KL, and clipping fraction.
""",
                "implement a short PPO update and plot all diagnostics",
            ),
            (
                "Compare PPO with unclipped updates",
                """
Starting from identical policy weights and rollout data, compare clipped PPO against the
unclipped surrogate. Measure policy drift and return after evaluation rollouts.
""",
                "run a controlled clipped-versus-unclipped comparison",
            ),
            (
                "Build a verifiable-reward task",
                """
Create small arithmetic prompts with deterministic answer checking. Define generated response,
parsed answer, verifier output, and scalar reward separately.
""",
                "implement a tiny arithmetic prompt generator and strict verifier",
            ),
            (
                "Model an RLVR batch",
                """
For each prompt, produce several candidate token/action sequences, assign verifier rewards,
compute group-relative or baseline-adjusted advantages, and record the training fields.
""",
                "construct a small RLVR-style batch table without calling an external LLM",
            ),
            (
                "Audit reward hacking",
                """
Create at least one malformed response that exploits a weak verifier, then strengthen the
verifier. Explain why verifiability is an assumption about the reward channel, not the model.
""",
                "demonstrate a weak-verifier exploit and a corrected check",
            ),
            (
                "Account for compute phases",
                """
Separate generation, reward evaluation, and gradient training. Estimate token counts and major
stored tensors for each phase; identify why generation can dominate wall-clock time.
""",
                "produce a compact phase-by-phase compute and memory table",
            ),
            (
                "Compare SFT and RLVR",
                """
State what supervision each method needs, what objective it optimizes, how data is obtained,
and what failure modes remain. Include the role of long chain-of-thought trajectories.
""",
                None,
            ),
        ],
        "checks": """
1. Why must old log probabilities be frozen for a PPO rollout batch?
2. How do advantages differ from raw returns?
3. What behavior does clipping discourage?
4. Why monitor KL and clipping fraction even when the loss decreases?
5. What makes a reward verifiable?
6. How can a model exploit a verifier?
7. Why are generation, reward calculation, and training distinct compute phases?
""",
        "next_steps": "You have reached the end of the public Spring 2026 lecture sequence. Revisit weak checks, complete one notebook at a time, and commit your own solutions separately.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing notebooks (commit your completed work first)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    built = []
    skipped = []
    for lecture in LECTURES:
        number = lecture["number"]
        folder = ROOT / f"lecture{number:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        filename = f"cs229_spring2026_lecture{number:02d}_{slugify(lecture['title'])}_practice.ipynb"
        path = folder / filename
        if path.exists() and not args.force:
            skipped.append(path)
            continue
        nbf.write(build_notebook(lecture), path)
        built.append(path)

    old_lecture2 = ROOT / "lecture02" / "cs229_lecture2_handtype_pytorch_practice_fixed.ipynb"
    if old_lecture2.exists() and args.force:
        old_lecture2.unlink()

    print(f"Built {len(built)} notebooks; skipped {len(skipped)} existing notebooks")
    for path in built:
        print(path.relative_to(ROOT))
    if skipped:
        print("Existing notebooks were preserved. Use --force only to rebuild blank starters.")


if __name__ == "__main__":
    main()
