# Import modules
from random import choice
import numpy as np

# Initialise weight vector
w = np.random.rand(9)

# Input vector
# training_data = [
#     (array([0,0,-1]), 0),
#     (array([0,1,-1]), 1),
#     (array([1,0,-1]), 1),
#     (array([1,1,-1]), 0),
# ]

# Training data
training_data = np.loadtxt("data.txt", delimiter=',')
print(training_data)

# Activation function
act_step = lambda x: 0 if x < 0 else 1

# Errors
errors = []

# Learning rate
ni = 0.2

# Learning iterations
n = 3000

# Forward function
for i in range(n):
    tmp = choice(training_data)
    x = np.append(tmp[:-1], -1)
    expected = tmp[-1]
    print("Input:", x)
    print("Expected:", expected)
    print("Weights:", w)
    result = np.dot(w, x)
    error = expected - act_step(result)
    errors.append(error)
    w += ni * error * x

# Print result
for x in training_data:
    result = np.dot(x, w)
    print("{}: {} -> {}".format(x[-1], result, act_step(result)))

# Compute recall and precision
tp, fp, fn = 0, 0, 0
for e in errors:
    tp += 1 if e == 0 else 0
    fp += 1 if e == 1 else 0
    fn += 1 if e == -1 else 0

print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("Precision:", float(tp)/(tp+fp))
print("Recall:", float(tp)/(tp+fn))

# Plot error graph
from pylab import plot, ylim
ylim([-1,1])
plot(errors)
