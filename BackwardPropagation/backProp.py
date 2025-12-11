import torch
import torch.nn as nn
import torch.optim as optim

# Input and target
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)  #input
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)  # output

# Define a tiny 2-layer neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)  # 2 inputs -> 2 hidden neurons
        self.output = nn.Linear(2, 1)  # 2 hidden -> 1 output neuron
        self.sigmoid = nn.Sigmoid()   
    
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))   #self.hidden(x) → multiply input by hidden layer weights + bias.  self.sigmoid(...) → apply sigmoid to introduce non-linearity.
        x = self.sigmoid(self.output(x))   #self.output(x) → multiply hidden outputs by output layer weights + bias.
        return x  #predicted output
 
# Instantiate model
model = SimpleNN() #Creates an instance of the network, initializing random weights and biases.

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Forward pass
pred = model(X)
loss = criterion(pred, y)
print("Initial loss:", loss.item())

# ---------------- BACKPROPAGATION ----------------
optimizer.zero_grad()   # Clear previous gradients

# This explanation is highlighting the importance of resetting gradients to zero before computing the
# gradients for the current batch during backpropagation in PyTorch.
# During backpropagation, when you call loss.backward(), PyTorch adds the new gradient to whatever is already stored in the .grad attribute of each 
# parameter.
# If you don’t reset, gradients will keep growing incorrectly across batches.

loss.backward()         # Compute gradients automatically
optimizer.step()        # Update weights using gradients

# Forward pass after one step
pred = model(X)
loss = criterion(pred, y)
print("Loss after one backprop step:", loss.item())
