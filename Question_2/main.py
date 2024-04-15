import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

class DataPreprocessor:
    def __init__(self, X, y, training_size=0.8, random_seed=1000):
        self.X = X
        self.y = y
        self.training_size = training_size
        self.random_seed = random_seed

    def standard_scaler(self):
        means = np.mean(self.X, axis=0)
        stds = np.std(self.X, axis=0)
        return (self.X - means) / stds

    def l2_normalizer(self, X):
        l2_norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / l2_norms

    def shuffle_data(self, X, y):
        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def encode_labels(self):
        return np.array([-1 if label == "Osmancik" else 1 for label in self.y.to_numpy()])

    def split_data(self, X, y):
        split_idx = int(self.training_size * len(X))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def preprocess(self):
        X_scaled = self.standard_scaler()
        X_normalized = self.l2_normalizer(X_scaled)
        X_shuffled, y_shuffled = self.shuffle_data(X_normalized.to_numpy(), self.y.to_numpy())
        y_encoded = self.encode_labels()
        return self.split_data(X_shuffled, y_encoded)

class GradientDescent:
    def __init__(self, X, y, learning_rate=0.01, iterations=1000, reg_param=0.1):
        self.X = X
        self.y = y.reshape(-1, 1)
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.reg_param = reg_param
        self.w = np.zeros((self.X.shape[1], 1))

    def calculate_gradient(self):
        predictions = sigmoid(np.dot(self.X, self.w))
        errors = self.y - predictions
        gradient = -np.dot(self.X.T, errors) / len(self.X)
        if self.reg_param:
            gradient += self.reg_param * self.w
        return gradient

    def fit(self):
        for _ in range(self.iterations):
            gradient = self.calculate_gradient()
            self.w -= self.learning_rate * gradient
        return self.w

    def predict(self, X):
        probabilities = sigmoid(np.dot(X, self.w))
        return np.where(probabilities > 0.5, 1, -1)

if __name__ == "__main__":
    dataset = fetch_ucirepo(id=545)
    X, y = dataset.data.features, dataset.data.targets

    preprocessor = DataPreprocessor(X, y)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    model = GradientDescent(X_train, y_train)
    model.fit()
    y_pred = model.predict(X_test)
    print(f"Accuracy: {calculate_accuracy(y_test, y_pred)}")
