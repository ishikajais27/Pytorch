import torch
import torch.nn as nn
import torch.optim as optim

# Example data
X = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]], dtype=torch.float32)      # (2 samples, 3 features)

y = torch.tensor([[1., 0., 1., 0.],
                  [0., 1., 0., 1.]], dtype=torch.float32)  # 4-neuron output

# ---------------------------------------------------------
# SINGLE LAYER NEURAL NETWORK (multiple neurons)
# ---------------------------------------------------------
class SingleLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.layer = nn.Linear(input_size, output_size)  # W: (3×4), b: (1×4)
        self.activation = nn.Sigmoid()                  # apply activation on full layer

    def forward(self, x):
        z = self.layer(x)      # X·W + b
        a = self.activation(z) # activation for all 4 neurons
        return a               # predicted outputs
        

# Model, Loss, Optimizer
model = SingleLayerNN(input_size=3, output_size=4)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
for epoch in range(100):
    optimizer.zero_grad()        # reset gradients to zero

    output = model(X)            # forward pass
    loss = criterion(output, y)  # compute loss

    loss.backward()              # backward pass: compute gradients
    optimizer.step()             # update weights and bias

print("Trained Weights:\n", model.layer.weight)
print("Trained Bias:\n", model.layer.bias)

# ---------------------------------------------------------
# TESTING
# ---------------------------------------------------------
X_test = torch.tensor([[2., 3., 4.]], dtype=torch.float32)
y_test = model(X_test)

print("Prediction:", y_test)
