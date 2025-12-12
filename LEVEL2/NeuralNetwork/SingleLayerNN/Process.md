Input X → shape: (samples, features)

# Forward pass

Z = X·W + b
A = activation(Z)
ŷ = A
L = loss(ŷ, y)

# Backward pass

dZ = dL/dA \* f'(Z)
dW = Xᵀ · dZ
db = sum(dZ along all samples)

# update parameter

W -= lr _ dW
b -= lr _ db

# Test

Z_test = X_test · W + b
ŷ_test = activation(Z_test)
