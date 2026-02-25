# Capstone Project: Budget-Constrained Bayesian Optimisation

### 1. Overview
As part of the _Professional Certificate in Machine Learning and Artificial Intelligence_ offered by Imperial College London, a capstone project defining a real-world problem must be completed as a final assessed component of the programme. In my cohort, it was the **Black-Box Optimisation (BBO)** project. The goal of this project was to maximise eight unknown black-box functions $f(\mathbf{x})=y$ under strict evaluation budgets. These black-box functions represented real-world optimisation problems commonly encountered in machine learning and scientific computing, such as the hyperparameter tuning of deep-learning models, the maximisation of log-likelihood functions, and identifying optimal compound combinations in drug discovery. In such scenarios, the functional form of $f(\mathbf{x})=y$ is unknown, gradients are unavailable, and each function evaluation is computationally or financially expensive. As a result, only a small number of evaluations can be performed.

### 2. Challenge Objectives

The objective of the challenge was to maximise each black-box function $f(\mathbf{x})$. While the number of known evaluations of varied between each function, there was a budget of only 13 additional evaluations for each function. It is important to note that the goal was not necessarily to identify the global maximum, but rather to locate a sufficiently high-quality local maximum within the constrained budget. In deep-learning hyperparameter optimisation this often the case as searching the entire feature space for global optima is computationally unfeasible. 

### 3. Function Specifications

| Function | Input | Output | Domain | Number of Known Evaluations | Description |
|----------|--------|---------|--------|-----------------------------|-------------|
| 1 | $(x_1, x_2)$ | $y$ |$[0, 1]^2$| 10 | $f(\mathbf{x})$ describes the intensity of a radiation field with multiple sources where $x_1$ and $x_2$ represent the 2D spatial coordinates.  |
| 2 | $(x_1, x_2)$ | $y$ |$[0, 1]^2$| 10 | $f(\mathbf{x})$ emulates the log-likelihood function $l(\mathbf{\theta})$ of a statistical model whose parameters are \mathbf{\theta} = ($x_1$, $x_2$).  |
 3 | $(x_1, x_2, x_3)$ | $y$ |$[0, 1]^3$| 15 | Each feature of the black-box function $x_1$, $x_2$ and $x_3$ are the amounts of each respective compound that form a new medical drug. The output $f(\mathbf{x})=y$ quantifies the side-effects as a negative number such that $y<0$. Smaller outputs convey stronger negative side-off of the drug. This function aims to imitate drug discovery projects. |

### 4. Technical Approach

Our approach was to perform Bayesian optimisation using the canonical setup that involes a surrogate model paired with an acquisition function. We implemented this optimisation procedure using the `skopt` library. Specifically, a Gaussian Process (GP) surrogate model was used to approximate the unknown black-box function $f(\mathbf{x})$. The GP provides both a posterior mean $\mu(\mathbf{x})$ that directly approximates $f(\mathbf{x})$ and provides an associated uncertainty $\sigma(\mathbf{x})$. With the aid of an acquistion function of choice, 


```algorithm
Algorithm: Bayesian Optimisation (BO)

Input:
    - Black-box objective function $f(\mathbf{x})$
    - Surrogate model $S$
    - Acquisition function $\alpha(\mathbf{x})$
    - Initial dataset $D_0 = {(x_i, y_i)}$ for $i = 1,..., n_0$
    - Evaluation budget $T$

Output:
    - Best observed solution $\mathbf{x}*$.

1:  $D \Longleftarrow D_0$
2:  for $t = n_0 + 1$ to $T$ do
3:      Fit surrogate model $S$ to dataset $D$
4:      Define acquisition function $\alpha(\mathbf{x}; S)$
5:      $x_t \Longleftarrow \text{argmax}_{\mathbf{x}} \alpha(\mathbf{x}; S)$
6:      $y_t \Longleftarrow f(\mathbf{x}_t)$               
7:      $D \Longleftarrow D ∪ {(x_t, y_t)}$
8:  end for
9:  $\mathbf{x}* \Longleftarrow \text{argmax}_{(x_i, y_i) \in D} y_i$
10: return $\mathbf{x}*$