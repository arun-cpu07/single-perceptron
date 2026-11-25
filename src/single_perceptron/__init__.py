from .perceptron import Perceptron

__all__ = ["Perceptron"]

try:
    from .numpy_perceptron import NumpyPerceptron

    has_numpy = True
    __all__.append("NumpyPerceptron")
except ImportError:
    has_numpy = False


def main() -> None:
    print("Hello from single perceptron!")
    p = Perceptron()
    x1_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 0, 0, 1]
    print("and gate: ")
    p.fit(x1_inputs, labels)
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
    print("30", "pass" if q.predict([30]) else "fail")
    print("3", "pass" if q.predict([3]) else "fail")
    print("86", "pass" if q.predict([86]) else "fail")
    print("98", "pass" if q.predict([98]) else "fail")

    v = Perceptron(max_epoch=5000)
    print("Loan approval based on difference between income and debt:")
    income_and_debt = [
        # Input loan and debt amounts are scaled to 10000
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
    print(f"your loan has been {'approved' if v.predict([7, 4.1]) else 'not approved'}")
    print(f"your loan has been {'approved' if v.predict([5, 2]) else 'not approved'}")
    print(f"your loan has been {'approved' if v.predict([8, 4]) else 'not approved'}")
    print(f"your loan has been {'approved' if v.predict([6, 5]) else 'not approved'}")
    print(f"your loan has been {'approved' if v.predict([10, 4]) else 'not approved'}")
    print(f"your loan has been {'approved' if v.predict([25, 10]) else 'not approved'}")

    if has_numpy:
        pass
    else:
        print(
            "\nThe perceptron using NumPy has been ignored... Refer README.md for instructions..."
        )
