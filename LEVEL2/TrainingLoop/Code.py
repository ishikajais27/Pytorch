import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------
# DATA (2 samples, 3 features each)
# ---------------------------------------
X = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]], dtype=torch.float32)

y = torch.tensor([[1.],
                  [0.]], dtype=torch.float32)


# ---------------------------------------
# MODEL (Single Neuron)
# ---------------------------------------
class SingleNeuron(nn.Module):
    def __init__(self, input_size):
        super().__init__()                      # activate nn.Module internal setup
        self.linear = nn.Linear(input_size, 1)  # one neuron: output = w·x + b

    def forward(self, x):
        return self.linear(x)


# Initialize model, loss, optimizer
model = SingleNeuron(input_size=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# ---------------------------------------
# TRAINING LOOP
# ---------------------------------------
for epoch in range(100):

    optimizer.zero_grad()       # 1. Clear old gradients

    output = model(X)           # 2. Forward pass → predictions

    loss = criterion(output, y) # 3. Compute loss (ŷ vs y)

    loss.backward()             # 4. Backward pass → compute gradients

    optimizer.step()            # 5. Update weights using Adam

    # Optional: print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# ---------------------------------------
# AFTER TRAINING — SHOW FINAL PARAMETERS
# ---------------------------------------
print("\nTrained Weights:", model.linear.weight.data)
print("Trained Bias:", model.linear.bias.data)


# ---------------------------------------
# VALIDATION (no gradient calculation)
# ---------------------------------------
with torch.no_grad():
    predictions = model(X)
    print("\nFinal Predictions:", predictions)
