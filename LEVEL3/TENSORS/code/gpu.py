import torch

print("==== 4. GPU TENSORS ====")

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

if torch.cuda.is_available():
    a_gpu = a.cuda()
    print("CPU tensor:\n", a)
    print("GPU tensor:\n", a_gpu)
    print("GPU device:", a_gpu.device)
else:
    print("GPU not available, using CPU")
