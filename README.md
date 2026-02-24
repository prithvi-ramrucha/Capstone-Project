# Capstone Project: Budget-Constrained Bayesian Optimisation

### 1. Overview
As part of the _Professional Certificate in Machine Learning and Artificial Intelligence_ offered by Imperial College London, a capstone project defining a real-world problem must be completed as a final assessed component of the programme. In my cohort, it was the **Black-Box Optimisation (BBO)** project. The goal of this project was to maximise eight unknown black-box functions $f(\mathbf{x})=y$. These black-box functions 

### 2. Function Specifications

| Function | Input | Output | Domain | Number of Known Evaluations | Description |
|----------|--------|---------|--------|-----------------------------|-------------|
| 1 | $(x_1, x_2)$ | $y$ |$[0, 1]^2$| 10 | $f(\mathbf{x})$ describes the intensity of a radiation field with multiple sources where $x_1$ and $x_2$ represent the 2D spatial coordinates.  |
| 2 | $(x_1, x_2)$ | $y$ |$[0, 1]^2$| 10 | $f(\mathbf{x})$ emulates the log-likelihood function $l(\mathbf{\theta})$ of a statistical model whose parameters are \mathbf{\theta} = ($x_1$, $x_2$).  |
 3 | $(x_1, x_2, x_3)$ | $y$ |$[0, 1]^3$| 15 | Each feature of the black-box function $x_1$, $x_2$ and $x_3$ are the amounts of each respective compound that form a new medical drug. The output $f(\mathbf{x})=y$ quantifies the side-effects as a negative number such that $y<0$. Smaller outputs convey stronger negative side-off of the drug. This function aims to imitate drug discovery projects. |