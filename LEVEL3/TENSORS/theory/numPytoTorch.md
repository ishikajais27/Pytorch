Inputs: Tensor or NumPy array
Outputs: NumPy array or tensor

PyTorch → NumPy

torch_tensor = torch.tensor([1.0,2.0,3.0])
numpy_array = torch_tensor.numpy()

NumPy → PyTorch

np_array = np.array([4.0,5.0,6.0])
torch_tensor = torch.from_numpy(np_array)

Note: They share memory; changing one affects the other.
