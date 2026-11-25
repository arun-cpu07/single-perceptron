import numpy as np


def step(ws):
    if ws > 0:
        return 1
    else:
        return 0


class NumpyPerceptron:
    def __init__(self, max_epoch=100, verbose=False):
        self.weights = []
        self.bias = 0.0
        self.lr = 0.1
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.training_result = "not trained"

    def fit(self, x_inputs, y_labels):
        update, iteration = True, 1
        x = np.atleast_2d(np.asarray(x_inputs))
        y = +np.asarray(y_labels).flatten()
        self.weights = np.array([0.0] * np.size(x[0]))
        self.bias = 0.0
        while update and iteration <= self.max_epoch:
            update = False
            for i in range(np.shape(x)[0]):
                ws = np.dot(x[i], self.weights)
                ws += self.bias
                res = step(ws)
                if y[i] != res:
                    correction_var = self.lr * (y[i] - res)
                    self.weights += x[i] * correction_var
                    self.bias += correction_var
                    update = True
                    if self.verbose:
                        print(f"weights {self.weights}, bias : {self.bias}")
            iteration += 1
        if update:
            if self.verbose:
                print(
                    f"The training didn't converge after {iteration - 1} runs. The weights resolved are {self.weights}"
                )
            self.training_result = "not converged"
        else:
            if self.verbose:
                print(
                    f"The training converged after {iteration - 1} runs. The weights are {self.weights} and bias is {self.bias}"
                )
            self.training_result = "converged"

    def predict(self, x_inputs):
        predictions = np.array([])
        x = np.atleast_2d(np.asarray(x_inputs))
        if self.training_result == "converged":
            ws = x @ self.weights + self.bias
            predictions = (ws > 0).astype(int)
            return predictions
        elif self.training_result == "not converged":
            return "the training didn't converge. If it a lineraly seperable dataset try increasing max_epoch"
        else:
            return "run the training first so the perceptron can learn the weights "


n = NumpyPerceptron(verbose=True)
n.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1])
print(n.predict([[0, 0], [0, 1], [1, 0], [1, 1]]))
