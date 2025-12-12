import torch
import torch.nn as nn
import torch.optim as optim

# Example data
X = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]], dtype=torch.float32)

y = torch.tensor([[1., 0.],
                  [0., 1.]], dtype=torch.float32)  # 2 output neurons

# ---------------------------------------------------------
# MULTI-LAYER PERCEPTRON (MLP)
# ---------------------------------------------------------
class MultiLayerNN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden1)  # W1, b1
        self.fc2 = nn.Linear(hidden1, hidden2)     # W2, b2
        self.fc3 = nn.Linear(hidden2, output_size) # W3, b3

        self.activation = nn.ReLU()
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.activation(z1)

        z2 = self.fc2(a1)
        a2 = self.activation(z2)

        z3 = self.fc3(a2)
        a3 = self.output_act(z3)

        return a3


# Create model
model = MultiLayerNN(input_size=3, hidden1=4, hidden2=3, output_size=2)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
for epoch in range(200):
    optimizer.zero_grad()

    output = model(X)
    loss = criterion(output, y)

    loss.backward()     # Computes gradients for ALL layers automatically
    optimizer.step()    # Updates ALL layer weights

print("Weights Layer 1:\n", model.fc1.weight)
print("Bias Layer 1:\n", model.fc1.bias)

print("Weights Layer 2:\n", model.fc2.weight)
print("Bias Layer 2:\n", model.fc2.bias)

print("Weights Layer 3:\n", model.fc3.weight)
print("Bias Layer 3:\n", model.fc3.bias)

# ---------------------------------------------------------
# TESTING
# ---------------------------------------------------------
X_test = torch.tensor([[2., 3., 4.]], dtype=torch.float32)
y_test = model(X_test)

print("Prediction:", y_test)
