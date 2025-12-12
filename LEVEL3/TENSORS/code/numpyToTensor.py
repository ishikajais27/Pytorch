import numpy as np
import torch

print("==== 5. NUMPY ↔ PYTORCH CONVERSION ====")

# PyTorch → NumPy
torch_tensor = torch.tensor([1.0, 2.0, 3.0])
numpy_array = torch_tensor.numpy()
print("PyTorch tensor:", torch_tensor)
print("Converted to NumPy array:", numpy_array)

# NumPy → PyTorch
np_array = np.array([4.0, 5.0, 6.0])
torch_from_np = torch.from_numpy(np_array)
print("\nNumPy array:", np_array)
print("Converted to PyTorch tensor:", torch_from_np)

# Check shared memory
torch_tensor[0] = 10
print("\nAfter modifying torch_tensor[0] = 10")
print("Torch tensor:", torch_tensor)
print("NumPy array:", numpy_array)
