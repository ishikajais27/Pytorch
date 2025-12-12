import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------
# 1. XOR DATA
# ---------------------------------------
X = torch.tensor([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]], dtype=torch.float32)

y = torch.tensor([[0],
                  [1],
                  [1],
                  [0]], dtype=torch.float32)

# ---------------------------------------
# 2. DEFINE MODEL (MLP)
# ---------------------------------------
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)   # Hidden layer: 2 inputs → 4 neurons
        self.output = nn.Linear(4, 1)   # Output layer: 4 neurons → 1 output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

model = XORNet()

# ---------------------------------------
# 3. LOSS & OPTIMIZER
# ---------------------------------------
criterion = nn.BCELoss()             # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.1)

# ---------------------------------------
# 4. TRAINING LOOP
# ---------------------------------------
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()       # Clear old gradients
    output = model(X)           # Forward pass → predictions
    loss = criterion(output, y) # Compute loss
    loss.backward()             # Backward pass → compute gradients
    optimizer.step()            # Update weights

    # Print progress every 500 epochs
    if (epoch+1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ---------------------------------------
# 5. TEST MODEL
# ---------------------------------------
with torch.no_grad():
    predictions = model(X)
    print("\nPredictions:")
    print(predictions)

# ---------------------------------------
# 6. SAVE MODEL
# ---------------------------------------
torch.save(model.state_dict(), "xor_model.pth")
print("\nModel saved as xor_model.pth")

# ---------------------------------------
# 7. LOAD MODEL (Optional)
# ---------------------------------------
loaded_model = XORNet()
loaded_model.load_state_dict(torch.load("xor_model.pth"))
loaded_model.eval()

with torch.no_grad():
    print("\nLoaded model predictions:")
    print(loaded_model(X))
