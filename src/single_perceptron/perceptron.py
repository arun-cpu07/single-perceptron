def step(ws):
    if ws > 0:
        return 1
    else:
        return 0


# Perceptron class
class Perceptron:
    def __init__(self, max_epoch=100, verbose=False):
        self.weights = []
        self.bias = 0
        self.lr = 0.1
        self.training_result = ""
        self.max_epoch = max_epoch
        self.verbose = verbose

    def fit(self, x_inputs, labels):
        self.x_inputs = x_inputs
        self.y = labels
        weighted_sum = 0
        update, iteration = 1, 1
        self.weights = [0] * len(self.x_inputs[0])
        self.bias = 0
        while update > 0 and iteration <= self.max_epoch:
            update = 0
            for i in range(len(self.x_inputs)):
                weighted_sum = 0
                x = self.x_inputs[i]
                y = self.y[i]
                for xn, wn in zip(x, self.weights):
                    weighted_sum += xn * wn
                weighted_sum += self.bias
                result = step(weighted_sum)
                if result != y:
                    correction_var = self.lr * (y - result)
                    self.weights = [
                        w + (a * correction_var) for w, a in zip(self.weights, x)
                    ]
                    self.bias += correction_var
                    update += 1
            iteration += 1
        else:
            if update != 0:
                if self.verbose:
                    print(
                        "not converged. If you are sure that the problem is lineraly seperable then try including an max_epoch count greater than the initial value 100"
                    )
                self.training_result = "not converged"
            else:
                if self.verbose:
                    print("converged")
                    print(f"the weights are {self.weights}, and bias is {self.bias}")
                self.training_result = "converged"

    def predict(self, x_inputs):
        if self.training_result == "not converged":
            return (
                "the training didnt converge. can't be predict by a sinlge perceptron"
            )
        else:
            ws = 0
            for i, x1 in enumerate(x_inputs):
                ws += x1 * self.weights[i]
            ws += self.bias
            return step(ws)
