def step(ws):
    if ws > 0:
        return 1
    else:
        return 0


# Perceptron class
class Perceptron:
    def __init__(
        self,
        max_epoch=100,
    ):
        self.weights = []
        self.bias = 0
        self.lr = 0.1
        self.training_result = ""
        self.max_epoch = max_epoch

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
                print(
                    "not converged. If you are sure that the problem is lineraly seperable then try including an max_epoch count greater than the initial value 100"
                )
                self.training_result = "not converged"
            else:
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


if __name__ == "__main__":
    p = Perceptron()
    x1_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 0, 0, 1]
    print("and gate: ")
    p.fit(x1_inputs, labels)
    print("and gate: ")
    print("1 1 ", p.predict([1, 1]))
    print("0 1 ", p.predict([0, 1]))
    print("1 0 ", p.predict([1, 0]))
    print("0 0 ", p.predict([0, 0]))

    q = Perceptron()
    x1_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 1, 1, 1]
    print("or gate: ")
    q.fit(x1_inputs, labels)
    print("1 1 ", q.predict([1, 1]))
    print("0 1 ", q.predict([0, 1]))
    print("1 0 ", q.predict([1, 0]))
    print("0 0 ", q.predict([0, 0]))

    r = Perceptron()
    x1_inputs = [[0], [1]]
    labels = [1, 0]
    print("or gate: ")
    r.fit(x1_inputs, labels)
    print("1 ", r.predict([1]))
    print("0 ", r.predict([0]))

    s = Perceptron()
    x1_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [1, 1, 1, 0]
    print("nand gate: ")
    s.fit(x1_inputs, labels)
    print("1, 1", s.predict([1, 1]))
    print("0, 1", s.predict([0, 1]))
    print("1, 0", s.predict([1, 0]))
    print("0, 0", s.predict([0, 0]))

    t = Perceptron()
    x1_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [1, 0, 0, 0]
    print("nor gate: ")
    t.fit(x1_inputs, labels)
    print("1, 1", t.predict([1, 1]))
    print("0, 1", t.predict([0, 1]))
    print("1, 0", t.predict([1, 0]))
    print("0, 0", t.predict([0, 0]))

    u = Perceptron()
    x1_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 1, 1, 0]
    print("xor gate: ")
    u.fit(x1_inputs, labels)
    print("1, 1", u.predict([1, 1]))
    print("0, 1", u.predict([0, 1]))
    print("1, 0", u.predict([1, 0]))
    print("0, 0", u.predict([0, 0]))

    q = Perceptron(max_epoch=500)
    marks_input = [[15], [28], [35], [36], [34], [33], [39], [40], [52], [68], [88]]
    marks_labels = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]
    q.fit(marks_input, marks_labels)
    print("36", "pass" if q.predict([36]) else "fail")
    print("30", "pass" if q.predict([36]) else "fail")
    print("3", "pass" if q.predict([36]) else "fail")
    print("86", "pass" if q.predict([36]) else "fail")
    print("98", "pass" if q.predict([36]) else "fail")

    v = Perceptron(max_epoch=1000000)
    income_and_debt = [
        [2, 0],  # Difference = 20000
        [2.5, 0.5],  # Difference = 20000
        [2.9, 0],  # Difference = 29000
        [5, 4],  # Difference = 10000
        [6, 5],  # Difference = 10000
        [7, 5.5],  # Difference = 15000
        [8, 6],  # Difference = 20000
        [9, 6.5],  # Difference = 25000
        [10, 7.1],  # Difference = 29000
        [10, 7],  # Difference = 30000
        [3, 0],  # Difference = 30000
        [4, 1],  # Difference = 30000
        [4, 0.5],  # Difference = 35000
        [10, 6.9],  # Difference = 31000
        [12, 8],  # Difference = 40000
        [15, 8],  # Difference = 70000
        [18, 9],  # Difference = 90000
        [20, 10],  # Difference = 100000
    ]
    approval_status = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    v.fit(income_and_debt, approval_status)
    v.predict([7, 4.1])
