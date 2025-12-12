import torch
import torch.nn as nn
import torch.optim as optim

# ---------------- 1. INPUT & TARGET ----------------
# XOR problem: inputs and corresponding outputs (target)
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# ---------------- 2. NEURAL NETWORK ----------------
# Layers + Activation Functions
class SimpleNN(nn.Module):
    def __init__(self):
        #nn.Module is the base class for all neural networks in PyTorch.
        #super().__init__() calls the constructor of the parent class (nn.Module)
        super().__init__()
        # Layer 1: input -> hidden (2 neurons)
        self.hidden = nn.Linear(2, 2)  
        # Layer 2: hidden -> output (1 neuron)
        self.output = nn.Linear(2, 1)  
        # Activation function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))  # hidden layer + activation
        x = self.sigmoid(self.output(x))  # output layer + activation
        return x

# Instantiate the model
model = SimpleNN()

# ---------------- 3. LOSS FUNCTION ----------------
# Mean Squared Error (how far predictions are from target)
criterion = nn.MSELoss()

# ---------------- 4. OPTIMIZER (GRADIENT DESCENT) ----------------
# Stochastic Gradient Descent updates weights using computed gradients
optimizer = optim.SGD(model.parameters(), lr=0.5)

# ---------------- 5. FORWARD PASS ----------------
pred = model(X)                 # model makes predictions
loss = criterion(pred, y)       # calculate how wrong it is
print("Initial loss:", loss.item())

# ---------------- 6. BACKPROPAGATION ----------------
optimizer.zero_grad()  # 1. Clear previous gradients
loss.backward()        # 2. Compute new gradients automatically
optimizer.step()       # 3. Update weights using gradients

# Forward pass after one weight update
pred = model(X)
loss = criterion(pred, y)
print("Loss after one backprop step:", loss.item())
