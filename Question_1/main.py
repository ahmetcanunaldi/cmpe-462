import numpy as np
import matplotlib.pyplot as plt

def check_missclassified(x):
    return np.where(x > 0, 1, -1)

def plot_result(X, y, weights, bias):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o', edgecolors='k', alpha=0.7)

    # Calculating the x-values for the decision boundary line
    x_values = np.array([np.min(X[:, 0]), np.max(X[:, 0])])

    # adding bias term
    weights = [bias, weights[0], weights[1]]
    y_values = -(weights[1] / weights[2]) * x_values - (weights[0] / weights[2])

    # Plotting the decision boundary line
    plt.plot(x_values, y_values, 'k')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('PLA Decision Boundary')
    plt.show()

class PLA:
    def __init__(self, data, label, iter=100):
        self.x = data
        self.y = label
        self.iter = iter
        self.w = 0
        self.b = 0
        self.n_iter = 0

    def fit(self):
        n_row, n_colm = self.x.shape
        self.w = np.zeros(n_colm) + 3
        iteration = 0

        while iteration < self.iter:
            error = False
            for idx, x_i in enumerate(self.x):
                W_dot_x = np.dot(x_i, self.w) + self.b
                y_predicted = check_missclassified(W_dot_x)

                if self.y[idx] * y_predicted <= 0:
                    self.w += self.y[idx] * x_i
                    self.b += self.y[idx]
                    error = True

            if error == False:
                self.n_iter = iteration
                break
            iteration += 1
        self.n_iter = iteration

    def predict(self):
        output = np.dot(self.x, self.w) + self.b
        y_predicted = np.where(output > 0, 1, -1)
        return y_predicted


if __name__ == "__main__":

    # Question 1
    data = np.load("PLA_Data/data_small.npy")
    data = np.delete(data, 0, 1)
    label = np.load("PLA_Data/label_small.npy")
    perceptron = PLA(data, label, 5000)
    perceptron.fit()
    predictions = perceptron.predict()

    plot_result(perceptron.x, perceptron.y, perceptron.w, perceptron.b)

    print("Weights:", perceptron.w)
    print("Bias:", perceptron.b)
    # print("Predictions:", predictions)
    # print("Labels:     ", check_missclassified(label))
    print("Iteration = ", perceptron.n_iter)
