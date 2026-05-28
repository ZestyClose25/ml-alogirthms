from matplotlib import pyplot as plot
import numpy as np

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# This part focuses on finding minima of cost function to get the right parameters value 

def y_function(x):
    return np.sin(x)

def y_derivative(x):
    return np.cos(x)

x = np.arange(-5, 5, 0.1)
y = y_function(x)

current_pos = (1.4, y_function(1.4))
learning_rate = 0.01

for i in range(1000):
    new_x = current_pos[0] - ( learning_rate * y_derivative(current_pos[0]) )
    new_y = y_function(new_x)
    current_pos = (new_x, new_y)

    plot.plot(x,y, color='blue')
    plot.scatter(current_pos[0], current_pos[1], color='red')
    plot.pause(0.01)
    plot.clf()

# ------------------------------------------------------------------------------------------------------------------
# This part focuses on finding the line of best fit/Regression line for the given dataset

# --- same simple dataset ---
np.random.seed(42)
X = 2 * np.random.rand(100)
y = 3 * X + 5 + np.random.randn(100) * 0.5

# parameters
m = 0
b = 0
#hyperparamter 
learningRate = 0.01

n = len(X)

plot.ion()

for epoch in range(100):
    Yhat = m * X + b
    error = Yhat - y

    # gradients ( derivative of cost function ) & dividing by n to average out all of them
    dm = (2/n) * np.dot(error, X)
    db = (2/n) * np.sum(error)

    m = m - (learningRate * dm)
    b = b - (learningRate * db)

    loss = np.mean(error**2)

    print(f"for {epoch} m:{m} b:{b} with loss:{loss}")
    plot.clf()
    plot.scatter(X, y, color='blue', label='Data')
    yline = m * X + b
    plot.plot(X, yline, color='red', label='Regression Line')
    plot.legend()
    plot.pause(0.1)

plot.ioff()
plot.show()
print(f"Final => m:{m}, b:{b}")
