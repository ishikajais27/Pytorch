import torch


def autograd_demo():
    print("="*60)
    print("AUTOGRAD: AUTOMATIC DIFFERENTIATION")
    print("="*60)
    
    # Compare: Manual vs Automatic gradient calculation
    print("\n📚 COMPARISON: MANUAL VS AUTOGRAD")
    
    # -------------------------
    # MANUAL CALCULATION
    # -------------------------
    print("\n1. MANUAL GRADIENT CALCULATION (NumPy style):")
    
    def manual_gradient():
        # Function: y = 3x² + 2x + 1
        x = 2.0 
        y = 3*x**2 + 2*x +1  #y=f(x)
        
        # Manual derivative: dy/dx = 6x + 2  df(x)/dx
        # At x=2: 6*2 + 2 = 14
        grad = 6*x + 2
        
        return y, grad
    
    y_manual, grad_manual = manual_gradient()
    print(f"   y = 3x² + 2x + 1 at x=2")
    print(f"   y = {y_manual}")
    print(f"   dy/dx = {grad_manual} (calculated manually)")
    
    # -------------------------
    # AUTOGRAD CALCULATION
    # -------------------------
    print("\n2. AUTOGRAD CALCULATION (PyTorch magic):")
    
    def autograd_example():
        # Create tensor with requires_grad=True
        x = torch.tensor(2.0, requires_grad=True)
        
        # Define computation after forward pass and loss direct call y.backward()
        y = 3*x**2 + 2*x + 1
        
        # PyTorch automatically tracks operations!
        # Compute gradient
        y.backward()  # This computes dy/dx
        
        # Access gradient
        grad = x.grad
        
        return y.item(), grad.item()
    
    y_auto, grad_auto = autograd_example()
    print(f"   y = {y_auto}")
    print(f"   dy/dx = {grad_auto} (computed by autograd)")
    
    print(f"\n✅ Results match: {grad_manual == grad_auto}")
    
    # -------------------------
    # MORE COMPLEX EXAMPLE
    # -------------------------
    print("\n3. COMPLEX CHAIN RULE EXAMPLE:")
    
    # Function: z = (x + y)²
    # Where: x = 2, y = 3
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    
    # Forward pass
    z = (x + y) ** 2
    
    print(f"   x = {x.item()}")
    print(f"   y = {y.item()}")
    print(f"   z = (x + y)² = {z.item()}")
    
    # Backward pass
    z.backward()
    
    print(f"\n   ∂z/∂x = {x.grad.item()} (should be 2*(x+y) = 10)")
    print(f"   ∂z/∂y = {y.grad.item()} (should be 2*(x+y) = 10)")
    
    # -------------------------
    # NEURAL NETWORK EXAMPLE
    # -------------------------
    print("\n4. NEURAL NETWORK LAYER WITH AUTOGRAD:")
    
    # Simulate a neural network layer
    # y = ReLU(W·x + b)
    
    # Input
    x_nn = torch.tensor([1.0, 2.0, 3.0])
    
    # Weights and biases (parameters to learn)
    W = torch.tensor([[0.1, 0.2, 0.3],
                      [0.4, 0.5, 0.6]], requires_grad=True)
    
    b = torch.tensor([0.1, 0.2], requires_grad=True)
    
    # Forward pass
    z_nn = torch.matmul(W, x_nn) + b
    y_nn = torch.relu(z_nn)
    
    print(f"\n   Input x: {x_nn}")
    print(f"   Weight W:\n{W}")
    print(f"   Bias b: {b}")
    print(f"   Linear output z = W·x + b: {z_nn}")
    print(f"   After ReLU: {y_nn}")
    
    # Define a simple loss
    target = torch.tensor([1.0, 2.0])
    loss = torch.sum((y_nn - target) ** 2)
    
    print(f"   Target: {target}")
    print(f"   Loss: {loss.item()}")
    
    # Backward pass (automatic gradients!)
    loss.backward()
    
    print(f"\n   Gradients computed automatically:")
    print(f"   ∂loss/∂W:\n{W.grad}")
    print(f"   ∂loss/∂b: {b.grad}")
    
    # -------------------------
    # GRADIENT ACCUMULATION
    # -------------------------
    print("\n5. GRADIENT ACCUMULATION WARNING:")
    
    # Important: PyTorch accumulates gradients!
    x_accum = torch.tensor(2.0, requires_grad=True)
    
    # First computation
    y1 = x_accum ** 2
    y1.backward()
    print(f"   After first backward(): grad = {x_accum.grad}")
    
    # Second computation (gradients accumulate!)
    y2 = x_accum ** 3
    y2.backward()
    print(f"   After second backward(): grad = {x_accum.grad} (accumulated!)")
    
    # How to zero gradients
    x_accum.grad.zero_()  # Zero the gradients
    print(f"   After zero_grad(): grad = {x_accum.grad}")
    
    return x, y, z

# Run autograd demo
autograd_demo()