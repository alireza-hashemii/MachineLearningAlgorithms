import numpy as np
import matplotlib.pyplot as plt

# sigmoid(logistic) function which returns a probablity
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# calculate z: it is calculated by multiplying each feature to its related coefficient
def compute_prediction(X , weights):
    z = np.dot(X , weights)
    return sigmoid(z)


# finding the appropriate coefficient using gradient descent
def update_weights_gd(X_train, y_train, weights, learning_rate):
    predictions = compute_prediction(X_train, weights)
    delta_weight = np.dot(X_train.T, (y_train - predictions))
    m = y_train.shape[0]
    weights += learning_rate / float(m) * delta_weight
    return weights


# calculate the cost (J(w)) 
def compute_cost(X , y , weights):
    predictions = compute_prediction(X, weights)
    # predictions = predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    cost = np.mean(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
    return cost


# train logistic model
def train_logistic_model(X_train, y_train, *,max_iter , learning_rate , fit_intercept=False):
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
        weights = np.zeros(X_train.shape[1])
    
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train ,y_train, weights , learning_rate)
    
        if iteration % 100 == 0:
           print(compute_cost(X_train, y_train, weights))

    return weights


def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))

    return compute_prediction(X, weights)



