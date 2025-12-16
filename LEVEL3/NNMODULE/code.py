import torch
import torch.nn as nn
import torch.nn.functional as F


def nn_module_basics():
    print("="*60)
    print("NN.MODULE: BUILDING NEURAL NETWORKS")
    print("="*60)
    
    # -------------------------
    # COMPARISON: Manual vs nn.Module
    # -------------------------
    print("\n📚 COMPARISON: MANUAL VS NN.MODULE")
    
    print("\n1. MANUAL NEURAL NETWORK (NumPy style):")
   #in init we calculate weigth and biases, in forward we calculate predicted output and then backward
    print("""
    class ManualNN:
        def __init__(self):   
            self.W1 = np.random.randn(784, 128)
            self.b1 = np.zeros(128)
            self.W2 = np.random.randn(128, 10)
            self.b2 = np.zeros(10)
        
        def forward(self, x):
            z1 = x @ self.W1 + self.b1
            a1 = relu(z1)
            z2 = a1 @ self.W2 + self.b2
            return softmax(z2)
        
        def backward(self, x, y, lr):  # 🤯 Manual gradients!
            # ... 50 lines of calculus ...
    """)
    
    print("\n2. PYTORCH NEURAL NETWORK (nn.Module style):")
    print("""
    class PyTorchNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(784, 128)  #  Built-in
            self.layer2 = nn.Linear(128, 10)   #  Built-in
        
        def forward(self, x):
            x = F.relu(self.layer1(x))  # Built-in
            x = self.layer2(x)          # Automatic
            return F.softmax(x, dim=1)  # Built-in
        
        # NO backward() method needed! 🤯
    """)
    
    # -------------------------
    # BUILDING YOUR FIRST NN.MODULE
    # -------------------------
    print("\n\n3. BUILDING YOUR FIRST NN.MODULE NETWORK:")
    
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()  # Always call parent init!
            
            # Define layers
            self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1 calculate w1,b1
            self.fc2 = nn.Linear(hidden_size, output_size)  # Fully connected layer 2 calculate w2,b2
            
            # You can add other components
            self.dropout = nn.Dropout(p=0.3)  # Dropout for regularization
            self.batchnorm = nn.BatchNorm1d(hidden_size)  # Batch normalization
            
            print(f"   Created SimpleNN:")
            print(f"     Input: {input_size}")
            print(f"     Hidden: {hidden_size}")
            print(f"     Output: {output_size}")
        
        def forward(self, x):
            # Forward pass through network
            x = self.fc1(x)  # Linear transformation
            x = self.batchnorm(x)  # Batch normalization optional
            x = F.relu(x)  # Activation function
            x = self.dropout(x)  # Dropout (only during training) optional
            x = self.fc2(x)  # Output layer
            
            # No softmax here - CrossEntropyLoss includes it
            return x
    
    # Create instance
    model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
    
    print(f"\n   Model architecture:\n{model}")
    
    # Check parameters
    print(f"\n   Parameters count: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # List all parameters
    print("\n   Parameter names and shapes:")
    for name, param in model.named_parameters():
        print(f"     {name}: {param.shape}")
    
    # -------------------------
    # USING THE MODEL
    # -------------------------
    print("\n\n4. USING THE MODEL:")
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 784)
    print(f"   Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(dummy_input)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output (first sample):\n{output[0]}")
    
    # Get predictions
    predictions = torch.argmax(output, dim=1)
    print(f"   Predictions: {predictions}")
    
    # -------------------------
    # MODEL MODES: TRAIN VS EVAL
    # -------------------------
    print("\n\n5. MODEL MODES (TRAIN vs EVAL):")
    
    print(f"   Initial mode: {model.training}")
    
    # Set to training mode
    model.train()
    print(f"   After model.train(): {model.training}")
    
    # Dropout/BatchNorm behave differently in training
    output_train = model(dummy_input)
    print(f"   Output in train mode (with dropout):\n{output_train[0]}")
    
    # Set to evaluation mode
    model.eval()
    print(f"   After model.eval(): {model.training}")
    
    # Dropout/BatchNorm behave differently in evaluation
    with torch.no_grad():
        output_eval = model(dummy_input)
    print(f"   Output in eval mode (no dropout):\n{output_eval[0]}")
    
    # Check difference
    diff = torch.abs(output_train - output_eval).mean()
    print(f"   Average difference between train/eval outputs: {diff.item():.6f}")
    
    return model

# Create first nn.Module network
simple_model = nn_module_basics()