[Step 1] Forward Pass:
Input X (features)
z = X·w + b # Weighted sum (pre-activation)
a = activation(z) # Apply activation function (sigmoid, ReLU, etc.)
ŷ = a # Predicted output

[Step 2] Compute Loss:
L = loss(ŷ, y) # Compare prediction ŷ to target y

[Step 3] Backward Pass (Compute Gradients):

# Goal: know how changing w,b changes loss

dz = dL/da _ f'(z) # derivative of loss w.r.t z
dw = X _ dz # derivative w.r.t weight
db = dz # derivative w.r.t bias

[Step 4] Update Parameters:
w -= learning_rate _ dw
b -= learning_rate _ db

[Step 5] Testing:
z_test = X_test·w + b
ŷ_test = activation(z_test)
