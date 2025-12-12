import torch

print("==== 2. TENSOR OPERATIONS ====")

# -----------------------------
# Basic operations
# -----------------------------
# Sample tensors
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print("Tensor a:\n", a)
print("Tensor b:\n", b)

# Element-wise operations
print("\na + b:\n", a + b)
print("a * b (element-wise):\n", a * b)

# Matrix multiplication
print("\na @ b:\n", a @ b)
print("torch.matmul(a, b):\n", torch.matmul(a, b))

# Reduction operations
print("\nSum of a:", a.sum())
print("Mean of a:", a.mean())
print("Max of a:", a.max())
print("Sum along columns:", a.sum(dim=0))
print("Sum along rows:", a.sum(dim=1))

# -----------------------------
# Advanced visualization
# -----------------------------
def visualize_tensor_operations():
    """Visualize how tensor operations work"""
    print("\n" + "="*60)
    print("VISUALIZING TENSOR OPERATIONS")
    print("="*60)
    
    # Matrix multiplication example
    A = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    
    B = torch.tensor([[7, 8],
                      [9, 10],
                      [11, 12]])
    
    print(f"A shape: {A.shape}")  # (2, 3)
    print(f"B shape: {B.shape}")  # (3, 2)
    
    print(f"\nA =\n{A}")
    print(f"\nB =\n{B}")
    
    C = A @ B  # (2, 3) @ (3, 2) = (2, 2)
    print(f"\nMatrix multiplication A @ B =\n{C}")
    print(f"C shape: {C.shape}")
    
    print("\n💡 How matrix multiplication works:")
    print("  C[0,0] = 1×7 + 2×9 + 3×11 = 58")
    print("  C[0,1] = 1×8 + 2×10 + 3×12 = 64")
    print("  C[1,0] = 4×7 + 5×9 + 6×11 = 139")
    print("  C[1,1] = 4×8 + 5×10 + 6×12 = 154")
    
    # Broadcasting example
    print("\n📊 BROADCASTING EXAMPLE:")
    X = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])  # (2, 3)
    y = torch.tensor([10, 20, 30])  # (3,)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"\nX =\n{X}")
    print(f"\ny = {y}")
    
    Z = X + y  # y broadcast to (2, 3)
    print(f"\nX + y (broadcasting) =\n{Z}")
    
    # Reshaping example
    print("\n🔄 RESHAPING TENSORS:")
    flat = torch.arange(12)  # [0, 1, 2, ..., 11]
    print(f"Flat tensor: {flat}, shape: {flat.shape}")
    
    reshaped = flat.reshape(3, 4)
    print(f"\nReshaped to (3, 4):\n{reshaped}")
    
    reshaped2 = flat.reshape(2, 3, 2)
    print(f"\nReshaped to (2, 3, 2):\n{reshaped2}")
    
    print("\n⚠️ VIEW VS RESHAPE:")
    print("  - view(): Shares memory, fast but requires contiguous tensor")
    print("  - reshape(): Can copy if needed, more flexible")
    
    return A, B, C

# Call visualization function
A, B, C = visualize_tensor_operations()
