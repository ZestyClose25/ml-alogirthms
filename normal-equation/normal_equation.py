import numpy as np
from matplotlib import pyplot as plt

# Creating random data samples
np.random.seed(42)
X = np.random.randn(500,1)
y = 2*X + 1 + 1.2*np.random.randn(500,1) 

def find_theta(X, y):
    m = X.shape[0]

    # Appending a column of ones to add bias term
    X = np.append(X, np.ones((m,1)), axis=1)

    # Reshaping y to (m,1)
    y = y.reshape(m,1)

    # Applying normal equation
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return theta

def predict(X, theta):
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

    preds = np.dot(X, theta)
    return preds

theta = find_theta(X, y)
preds = predict(X, theta)

fig = plt.figure(figsize=(8,6))
plt.plot(X, y, 'b.')
plt.plot(X, preds, 'c-')
plt.xlabel('X - Input')
plt.ylabel('Y - Target')
plt.show()