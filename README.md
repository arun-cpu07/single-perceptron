# Single Perceptron

A from-scratch Python implementation of a single perceptron, the building block of a neural network.

- **perceptron.py** this version uses only the python standard libraries.
- **numpy_perceptron.py** this version uses NumPy for faster matrix multiplications


Neither uses scikit-learn or TensorFlow.

Repository: <https://github.com/arun-cpu07/single-perceptron>

## What is a perceptron?

A perceptron works like a very simple neuron: it receives inputs and "fires" (outputs `1`) when a certain threshold is reached, otherwise it outputs `0`.

Two things decide when it fires:

- **Weights:** how much each input matters.
- **Bias:** how easy or hard it is to reach the threshold.

**Training** (also called fitting) is the process of adjusting the weights and bias. You give the perceptron inputs along with the output it *should* produce, and every time it gets one wrong, it nudges the weights and bias to be a little more correct.

## Installation

Clone the repository and set up the environment (the project uses [uv](https://github.com/astral-sh/uv) and needs Python 3.14 or newer):

```bash
git clone https://github.com/arun-cpu07/single-perceptron
cd single-perceptron
uv sync
```

To use it as a dependency in your own project instead:

```bash
uv add git+https://github.com/arun-cpu07/single-perceptron
```

## Usage

The `Perceptron` class provides two methods:

- `fit(X, y)`: trains the perceptron on the training data.
- `predict(x)`: returns `1` or `0` for a given input, based on what it learned.

Example using an AND gate:

```python
from single_perceptron import Perceptron

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]

p = Perceptron(max_epoch=100, verbose=True)
p.fit(X, y)

print(p.predict([1, 1]))  # 1
print(p.predict([0, 1]))  # 0
```

To use it in your own program, create a `main.py` in the root folder and add `from single_perceptron import Perceptron`.

To see the built-in demo (logic gates, pass/fail marks and loan approval), run:

```bash
uv run single-perceptron
```

## Parameters

| Parameter   | Default | Description                                                                                          |
|-------------|---------|------------------------------------------------------------------------------------------------------|
| `max_epoch` | `100`   | Maximum number of times the perceptron goes through the training data before giving up.              |
| `verbose`   | `False` | When `True`, prints whether training converged or not, along with the final weights and bias.        |

Some `max_epoch` values that work for the examples in this repo:

| Example                       | Epochs needed to converge | `max_epoch` used |
|-------------------------------|---------------------------|------------------|
| Logic gates (AND, OR, NOT...) | well under 100            | `100` (default)  |
| Pass/fail from marks          | 157                       | `500`            |
| Loan approval (scaled data)   | 1479                      | `5000`           |

## Limitations

### Linear separability (XOR)

A single perceptron can only solve **linearly separable** problems. This means that if you plot the inputs, the ones that should fire and the ones that shouldn't must be separable by a single straight line.

- AND, OR, NOT, NOR and NAND gates are linearly separable, so the perceptron can learn them.
- XOR is **not** linearly separable. No single straight line can divide the points (0,0), (1,1) from (1,0), (0,1), so a single perceptron will never converge on it.

For data that is linearly separable, the perceptron is mathematically guaranteed to find a solution. For data that isn't, it never will.

### Slow convergence and unscaled data (loan example)

Even for linearly separable problems, the perceptron may take an enormous number of steps to converge.

`tests/test_perceptron.py` contains a loan approval dataset. The numbers there are **scaled down by 10000**, because with the raw values the perceptron did not converge (meaning it could not find the line that separates the inputs) even after 1,000,000 iterations.

The result is also not perfectly precise near the boundary: with a loan limit of 30000, values just below it (for example a difference of 29500 or 29950) are still accepted by the perceptron, because it stops at the first line that separates the training data and does not look for the best one. A model that finds the maximum-margin boundary, such as an **SVM**, handles this kind of problem better.

**Tip:** always scale or normalize your inputs before training.

## Project structure

```
src/single_perceptron/
    __init__.py       # exports Perceptron and the demo main()
    perceptron.py     # the step function and the Perceptron class
tests/
    test_perceptron.py
```

## Running tests

```bash
uv run pytest
```
