import numpy as np


# Sigmoid activation and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# Simple neuron with 2 inputs
weights = np.array([0.5, -0.3])   # initial weights
bias = 0.1                         # initial bias
inputs = np.array([1.0, 2.0])      # example input
target = 1.0                        # target output (what it should predict)

# ---- Forward pass ----
weighted_sum = np.dot(weights, inputs) + bias
output = sigmoid(weighted_sum)

# Compute loss (MSE)
loss = 0.5 * (output - target)**2
print("Forward pass:")
print(f"Weighted sum: {weighted_sum}")
print(f"Output: {output}")
print(f"Loss: {loss}")

# ---- Backward pass (Gradient calculation) ----
# Step 1: derivative of loss w.r.t output
dL_dout = output - target

# Step 2: derivative of output w.r.t weighted sum
dout_dz = sigmoid_derivative(output)

# Step 3: derivative of weighted sum w.r.t weights
dz_dw = inputs  # because z = w1*x1 + w2*x2 + b

# Step 4: gradient of loss w.r.t weights
grad_w = dL_dout * dout_dz * dz_dw

# Step 5: gradient of loss w.r.t bias
grad_b = dL_dout * dout_dz * 1

print("\nGradients:")
print(f"dL/dw: {grad_w}")
print(f"dL/db: {grad_b}")

# ---- Update weights manually ----
learning_rate = 0.1
weights -= learning_rate * grad_w
bias -= learning_rate * grad_b

print("\nUpdated weights and bias:")
print(f"Weights: {weights}")
print(f"Bias: {bias}")
