What it is: Tensors can be moved to GPU for faster computation.

Inputs: CPU tensor
Outputs: GPU tensor

if torch.cuda.is_available():
a_gpu = a.cuda()

GPU tensors are faster for neural networks.

Switch back to CPU: a_cpu = a_gpu.cpu()
