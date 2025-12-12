compute z and activaiton for all layers and pred y will be equal to activation of last or output layer. ->
then calculate single loss -> backwardpropagation find derivative of loss wrt to all parameters.
and then update all parameters.

# ForwardPass

Layer 1 ->
Z1 = X·W1 + b1
A1 = f(Z1)

Layer 2 ->
Z2 = A1·W2 + b2
A2 = f(Z2)

Output Layer ->
Z3 = A2·W3 + b3
A3 = f(Z3) = ŷ

# Loss

L = loss(ŷ, y)

# BackwardPropagation ->

Derivative or gradient of loss wrt to all parameters

# Update Parameters

Wk -= lr _ dWk , k=1,2,3
bk -= lr _ dbk

# Testing

Run forward pass only.
ŷ_test = model(X_test)
