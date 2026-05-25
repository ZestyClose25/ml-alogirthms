# Gradient Descent Visualizer

This repository contains a simple, visual Python script that demonstrates the **Gradient Descent** algorithm using `matplotlib`. It is designed to help beginners build an intuition for how gradient descent operates under the hood through real-time plotting animations.

The code is divided into two main visual demonstrations:

## 1. Finding the Minima of a Function
This section visualizes a point stepping down the slope of a mathematical function (a sine wave, `y = sin(x)`) to find the local minimum.
* **Cost Function:** `y = sin(x)`
* **Derivative:** `y' = cos(x)`
* **Visualization:** Animates a red dot descending the curve step-by-step using a defined learning rate, clearing and redrawing the plot to create a smooth animation.

## 2. Linear Regression (Line of Best Fit)
This section applies gradient descent to a fundamental machine learning problem: finding the line of best fit for a 2D dataset.
* **Dataset:** Randomly generated scatter points conceptually grouped around the line `y = 3x + 5` (with added noise).
* **Algorithm:** Iteratively updates the slope (`m`) and y-intercept (`b`) by calculating the gradients of the Mean Squared Error (MSE) cost function.
* **Visualization:** Animates the regression line shifting and adjusting as it "learns" the optimal parameters to fit the data points over 100 epochs.

## Prerequisites

To run this code, you need Python 3.x installed on your system along with the following libraries:

Code output
success

```bash
pip install numpy matplotlib
Running the Code
Simply execute the Python script. A Matplotlib interactive window will open, showing the animations in real-time.

Bash
python gradient_descent.py
(Make sure to name your python file appropriately or update the command above)

How it Works
Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.

It starts with an initial set of parameters (e.g., m = 0, b = 0).

It calculates the derivative (gradient) of the cost function at the current point.

It takes a small step (determined by the learning rate) in the opposite direction of the gradient to reduce the error.

This process repeats for a set number of epochs or until the algorithm converges to a minimum.
