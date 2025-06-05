class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """Calculate the sigmoid function."""
        return 1 / (1 + self.exp(-z))

    def exp(self, x):
        """Calculate the exponential function (exp)."""
        # Using Taylor series expansion for exp(x)
        # exp(x) = 1 + x + x^2/2! + x^3/3! + ...
        term = 1.0  # First term in the expansion
        result = 1.0  # Start with 1 (the first term)
        for n in range(1, 100):  # Use 100 terms in the series
            term *= x / n  # Calculate the next term
            result += term  # Add the term to the result
        return result

    def fit(self, X, y):
        """Train the model using gradient descent."""
        num_samples, num_features = len(X), len(X[0])
        self.weights = [0] * num_features
        self.bias = 0

        for _ in range(self.num_iterations):
            # Compute linear model
            linear_model = [sum(x_i * w for x_i, w in zip(x, self.weights)) + self.bias for x in X]
            y_predicted = [self.sigmoid(z) for z in linear_model]

            # Compute gradients
            dw = [0] * num_features
            db = 0
            for i in range(num_samples):
                error = y_predicted[i] - y[i]
                for j in range(num_features):
                    dw[j] += (1 / num_samples) * error * X[i][j]
                db += (1 / num_samples) * error

            # Update weights and bias
            for j in range(num_features):
                self.weights[j] -= self.learning_rate * dw[j]
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """Make predictions using the trained model."""
        linear_model = [sum(x_i * w for x_i, w in zip(x, self.weights)) + self.bias for x in X]
        y_predicted = [self.sigmoid(z) for z in linear_model]
        return [1 if prob > 0.5 else 0 for prob in y_predicted]