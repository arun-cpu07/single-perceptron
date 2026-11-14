from perceptron import Perceptron


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
    v = Perceptron(max_epoch=500)
    marks_input = [[15], [28], [35], [36], [34], [33], [39], [40], [52], [68], [88]]
    marks_labels = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]
    v.fit(marks_input, marks_labels)
    print("36", v.predict([36]))
    print("30", v.predict([30]))
    print("3", v.predict([3]))
    print("86", v.predict([86]))
    print("98", v.predict([96]))
