import numpy as np
import torch

print("==== 1. CREATING TENSORS ====")

# From Python list
t1 = torch.tensor([1, 2, 3])
print("From list:", t1)

# From NumPy array
np_array = np.array([4, 5, 6])
t2 = torch.from_numpy(np_array)
print("From NumPy:", t2)

# Random tensors
t3 = torch.rand(2, 3)   # Uniform [0,1]
t4 = torch.randn(2, 3)  # Normal distribution
print("\nRandom uniform (2x3):\n", t3)
print("Random normal (2x3):\n", t4)

# Special tensors
t5 = torch.zeros(3, 4)
t6 = torch.ones(2, 2)
t7 = torch.eye(3)
print("\nZeros (3x4):\n", t5)
print("Ones (2x2):\n", t6)
print("Identity (3x3):\n", t7)
