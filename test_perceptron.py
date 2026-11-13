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
