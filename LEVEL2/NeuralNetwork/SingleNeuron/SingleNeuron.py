import torch
import torch.nn as nn
import torch.optim as optim

# Example data (2 samples(rows), 3 features(cloumns))
X = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]], dtype=torch.float32)
y = torch.tensor([[1.],
                  [0.]], dtype=torch.float32)

# Single neuron model
class SingleNeuron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
#nn.Module is the parent class
#SingleNeuron is the child class
#super().__init__() runs the __init__() of nn.Module
        
        self.linear = nn.Linear(input_size, 1)  # single output
        self.activation = nn.Sigmoid()          # activation

    def forward(self, x):
        z = self.linear(x)
        a = self.activation(z)
        return a  #predicted output
 
# Model, loss, optimizer
model = SingleNeuron(input_size=3)
criterion = nn.BCELoss()          # binary cross-entropy(claculate loss function)
optimizer = optim.SGD(model.parameters(), lr=0.01)  #learning rate decide how long step model should take

# Training loop
for epoch in range(100):
    optimizer.zero_grad()  #set the gradient value 0 before calculating it
    output = model(X)      #get predicted o/p 
    loss = criterion(output, y)  #calculate loss
    loss.backward()           # PyTorch computes backward automatically calculates the gradient of the loss with
    #respect to every trainable parameter
    optimizer.step() #uses the gradients to update the parameters.
    
print("Trained weights:", model.linear.weight)
print("Trained bias:", model.linear.bias)

# Testing
X_test = torch.tensor([[2., 3., 4.]], dtype=torch.float32)
y_test = model(X_test)
print("Prediction:", y_test)
