from single_perceptron.perceptron import Perceptron


def test_and_gate():
    p = Perceptron()
    p.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1])
    assert p.predict([1, 1]) == 1
    assert p.predict([0, 1]) == 0
    assert p.predict([1, 0]) == 0
    assert p.predict([0, 0]) == 0


def test_or_gate():
    p = Perceptron()
    p.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 1])
    assert p.predict([1, 1]) == 1
    assert p.predict([0, 1]) == 1
    assert p.predict([1, 0]) == 1
    assert p.predict([0, 0]) == 0


def test_not_gate():
    p = Perceptron()
    p.fit([[0], [1]], [1, 0])
    assert p.predict([1]) == 0
    assert p.predict([0]) == 1


def test_nand_gate():
    p = Perceptron()
    p.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [1, 1, 1, 0])
    assert p.predict([1, 1]) == 0
    assert p.predict([0, 1]) == 1
    assert p.predict([1, 0]) == 1
    assert p.predict([0, 0]) == 1


def test_nor_gate():
    p = Perceptron()
    p.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [1, 0, 0, 0])
    assert p.predict([1, 1]) == 0
    assert p.predict([0, 1]) == 0
    assert p.predict([1, 0]) == 0
    assert p.predict([0, 0]) == 1


def test_xor_gate():
    p = Perceptron()
    p.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])
    assert (
        p.predict([1, 1])
        == "the training didnt converge. can't be predict by a sinlge perceptron"
    )
    assert (
        p.predict([0, 1])
        == "the training didnt converge. can't be predict by a sinlge perceptron"
    )
    assert (
        p.predict([1, 0])
        == "the training didnt converge. can't be predict by a sinlge perceptron"
    )
    assert (
        p.predict([0, 0])
        == "the training didnt converge. can't be predict by a sinlge perceptron"
    )


def test_pass_scores():
    q = Perceptron(max_epoch=500)
    marks_input = [[15], [28], [35], [36], [34], [33], [39], [40], [52], [68], [88]]
    marks_labels = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]
    q.fit(marks_input, marks_labels)
    assert q.predict([36]) == 1
    assert q.predict([30]) == 0
    assert q.predict([3]) == 0
    assert q.predict([86]) == 1
    assert q.predict([96]) == 1


def test_loan_approval():
    v = Perceptron(max_epoch=5000)
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
    assert v.predict([5, 2]) == 1
    assert v.predict([8, 4]) == 1
    assert v.predict([10, 7]) == 1  # edge-case
    assert v.predict([9, 6]) == 1  # edge-case
    assert v.predict([15, 10]) == 1
    assert v.predict([7, 4.1]) == 0  # edge-case
    assert v.predict([10, 7.1]) == 0  # edge-case
    assert v.predict([12, 9.5]) == 0
    assert v.predict([6, 5]) == 0
    assert v.predict([3, 1]) == 0
    assert v.predict([20, 18]) == 0
    assert v.predict([10, 5]) == 1
