# Capstone Project: Budget-Constrained Bayesian Optimisation

### 1. Overview
As part of the _Professional Certificate in Machine Learning and Artificial Intelligence_ offered by Imperial College London, a capstone project defining a real-world problem must be completed as a final assessed component of the programme. In my cohort, it was the **Black-Box Optimisation (BBO)** project. The goal of this project was to maximise eight unknown black-box functions $f(\mathbf{x})=y$ under strict evaluation budgets. These black-box functions represented real-world optimisation problems commonly encountered in machine learning and scientific computing, such as the hyperparameter tuning of deep-learning models, the maximisation of log-likelihood functions, and identifying optimal compound combinations in drug discovery. In such scenarios, the functional form of $f(\mathbf{x})=y$ is unknown, gradients are unavailable, and each function evaluation is computationally or financially expensive. As a result, only a small number of evaluations can be performed.

### 2. Challenge Objectives

The objective of the challenge was to maximise each black-box function $f(\mathbf{x})$. While the number of known evaluations of varied between each function, there was a budget of only 13 additional evaluations for each function. It is important to note that the goal was not necessarily to identify the global maximum, but rather to locate a sufficiently high-quality local maximum within the constrained budget. In deep-learning hyperparameter optimisation this often the case as searching the entire feature space for global optima is computationally unfeasible. 

| Function | Input | Output | Domain | Number of Known Evaluations | Description |
|----------|--------|---------|--------|-----------------------------|-------------|
| 1 | $(x_1, x_2)$ | $y$ |$[0, 1]^2$| 10 | $f(\mathbf{x})$ describes the intensity of a radiation field with multiple sources where $x_1$ and $x_2$ represent the 2D spatial coordinates.  |
| 2 | $(x_1, x_2)$ | $y$ |$[0, 1]^2$| 10 | $f(\mathbf{x})$ emulates the log-likelihood function $l(\mathbf{\theta})$ of a statistical model whose parameters are $\mathbf{\theta} = (x_1, x_2)$.  |
3 | $(x_1, x_2, x_3)$ | $y$ |$[0, 1]^3$| 15 | Each feature of the black-box function $x_1$, $x_2$ and $x_3$ are the amounts of each respective compound that form a new medical drug. The output $f(\mathbf{x})=y$ quantifies the side-effects as a negative number such that $y<0$. Smaller outputs convey stronger negative side-off of the drug. This function aims to imitate drug discovery projects. |
4 | $(x_1, x_2, x_3, x_4)$ | $y$ |$[0, 1]^4$| 30 | $f(\mathbf{x})$ represents a machine learning model with the features $x_1, x_2, x_3$ and $x_4$. The model places products across warehouses for a business with high online sales. As output, $f(\mathbf{x})$ returns the difference from the expensive baseline. |
5 | $(x_1, x_2, x_3, x_4)$ | $y$ |$[0, 1]^4$| 20 | $f(\mathbf{x})$ returns the chemical yield of an industrial process that is directly affected by the parameters $x_1, x_2, x_3$ and $x_4$. The distribution is unimodal with a global peak where the yield is maximial. | 
6 | $(x_1, x_2, x_3, x_4, x_5)$ | $y$ |$[0, 1]^5$| 20 | Each feature $x_i$ quantifies the amount of an ingredient of a cake (flour, sugar, eggs, butter and milk). The function $f(\mathbf{x}) < 0$ returns a negative score where scores closer to zero are favoured. |
7 | $(x_1, x_2, x_3, x_4, x_5, x_6)$ | $y$ |$[0, 1]^6$| 30 | The black-box function $f(\mathbf{x})$ represents the performance metric of deep-learning model. Each feature $x_i$ is a hyperparameter to the model. |
6 | $(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8)$ | $y$ |$[0, 1]^8$| 40 | The black-box function $f(\mathbf{x})$ represents the performance metric (validation accuracy) of deep-learning model. Each feature $x_i$ is a hyperparameter to the model. |

### 4. Technical Approach

Our approach was to perform Bayesian optimisation using the canonical setup that involves a surrogate model paired with an acquisition function. We implemented this optimisation procedure using the `skopt` library. Specifically, a Gaussian Process (GP) surrogate model was used to approximate the unknown black-box function $f(\mathbf{x})$. The GP provides both a posterior mean $\mu(\mathbf{x})$ that directly approximates $f(\mathbf{x})$ and provides an associated uncertainty $\sigma(\mathbf{x})$. With the aid of an acquisition function of choice, new points to evaluate are determined. These new evaluations are used to update the surrogate model, and this process repeats until the budget has been spent or a suitable optimum has been found. The canonical BO process is outlined by the pseudocode below,

```text
Algorithm: Bayesian Optimisation (BO)

Input:
    - Black-box objective function f(x)
    - Surrogate model S
    - Acquisition function α(x)
    - Initial dataset D0 = {(x_i, y_i)} for i = 1,..., n0
    - Evaluation budget T

Output:
    - Best observed solution x*

1:  D ← D0
2:  for t = n0 + 1 to T do
3:      Fit surrogate model S to dataset D
4:      Define acquisition function α(x; S)
5:      x_t ← argmax_x α(x; S)
6:      y_t ← f(x_t)
7:      D ← D ∪ {(x_t, y_t)}
8:  end for
9:  x* ← argmax_(x_i, y_i ∈ D) y_i
10: return x*
```

An important aspect of the GP process is the kernel $\mathbf{K}$ or covariance function which allows for the creation of the surrogate model by measuring the similarity between data points. It encodes the assumptions about the black-box function $f(\mathbf{x})$ regarding smoothness and noise. For each black-box function, the challenge was to determine the approapriate kernel $\mathbf{K}$, acquistion function $\alpha{\mathbf{x}}$ and exploration/exploitation trade-off parmaeter values ($\kappa$, $\xi$, etc.). 
```