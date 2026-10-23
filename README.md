# Linear Regression from Scratch using NumPy

## Overview
This project implements simple and multiple linear regression from scratch using NumPy.
It demonstrates the hypothesis function, mean squared error (MSE), gradient descent,
and manual evaluation of predictions.

## Dataset
- Dataset: California Housing
- Total samples: 20,640
- Input features for multiple regression: 8
- Training samples: 16,512
- Test samples: 4,128
- Split: 80% training, 20% testing

Missing values in `total_bedrooms` were replaced with the training-independent
column median before splitting. Numerical input features were standardized using
statistics calculated from the training set only. The target was also standardized
using training-set statistics for stable gradient descent.

## Mathematical Implementation

### Hypothesis Function
The model predicts the target using:

h_theta(X) = X theta

A column of ones is added to X to learn the bias (intercept).

### Mean Squared Error
The cost function is:

J(theta) = (1/m) * sum((prediction - actual)^2)

### Gradient Descent
Parameters are updated using:

theta = theta - learning_rate * gradient

The gradient is calculated using NumPy matrix operations. The cost is recorded
at every iteration to visualize convergence.

## Results

### Multiple Linear Regression
- Features: 8
- Parameters: 9 (8 weights + bias)
- Learning rate: 0.01
- Iterations: 20,000
- Test MSE: 5,053,696,426.68
- Test RMSE: 71,089.36
- Test R²: 0.614342

### Simple Linear Regression
- Feature: `median_income`
- Parameters: 2 (1 weight + bias)
- Learning rate: 0.01
- Iterations: 20,000
- Test MSE: 7,091,157,771.77
- Test RMSE: 84,209.01
- Test R²: 0.458859

### Scikit-learn Comparison
The multiple regression model was compared with scikit-learn's `LinearRegression`
on the same train/test split.

| Metric | NumPy implementation | Scikit-learn |
|---|---:|---:|
| Test MSE | 5,053,696,426.68 | 5,059,928,371.17 |
| Test R² | 0.614342 | 0.613866 |

The implementations produce similar test metrics. Gradient descent is an iterative
optimization method, so its parameters may differ slightly from the least-squares
solution computed by scikit-learn.

## Cost Convergence
The plot below shows the cost decreasing during gradient descent training.

![Gradient Descent Convergence](results/cost_convergence.png)

## Technologies
- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn (train/test split and final comparison only)

## How to Run
1. Clone or download this repository.
2. Install the required packages.
3. Open `linear_regression.ipynb` in Jupyter Notebook.
4. Run the notebook cells in order.

## Dependencies
- numpy
- pandas
- matplotlib
- scikit-learn

## Conclusion
This project demonstrates how linear regression works internally by implementing
the prediction function, cost function, and gradient descent without using
scikit-learn for model training.
