import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------------------
# 1. Define a simple model
# --------------------------------------------
class SingleNeuron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

model = SingleNeuron(input_size=3)
optimizer = optim.Adam(model.parameters(), lr=0.01)


# --------------------------------------------
# 2. SAVE MODEL (Best Method: Save State Dict)
# --------------------------------------------
torch.save(model.state_dict(), "model_weights.pth")
print("Model weights saved!")


# --------------------------------------------
# 3. LOAD MODEL (Create same model again)
# --------------------------------------------
loaded_model = SingleNeuron(input_size=3)

# Load weights
loaded_model.load_state_dict(torch.load("model_weights.pth"))

# Switch to evaluation mode
loaded_model.eval()

print("Model weights loaded successfully!")
print("Loaded Weights:", loaded_model.linear.weight.data)
print("Loaded Bias:", loaded_model.linear.bias.data)


# --------------------------------------------
# 4. SAVE CHECKPOINT (model + optimizer + epoch)
# --------------------------------------------
torch.save({
    "epoch": 50,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict()
}, "checkpoint.pth")

print("\nCheckpoint saved!")


# --------------------------------------------
# 5. LOAD CHECKPOINT (resume training)
# --------------------------------------------
checkpoint = torch.load("checkpoint.pth")

model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"]

print("\nCheckpoint loaded!")
print("Resuming training from epoch:", start_epoch)
