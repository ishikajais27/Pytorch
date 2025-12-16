import torch
import torch.nn as nn
import torch.nn.functional as F


def different_network_architectures():
    """Show different ways to build neural networks in PyTorch"""
    print("\n" + "="*60)
    print("DIFFERENT WAYS TO BUILD NEURAL NETWORKS")
    print("="*60)
    
    # -------------------------
    # METHOD 1: SEQUENTIAL (Simplest)
    # -------------------------
    print("\n1. METHOD 1: NN.SEQUENTIAL (Simplest)")
    
    sequential_model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    print(f"   Sequential Model:\n{sequential_model}")
    
    # Test it
    dummy_input = torch.randn(4, 784)
    output_seq = sequential_model(dummy_input)
    print(f"   Output shape: {output_seq.shape}")
    
    # -------------------------
    # METHOD 2: CUSTOM NN.MODULE (More Control)
    # -------------------------
    print("\n\n2. METHOD 2: CUSTOM NN.MODULE (More Control)")
    
    class CustomNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
            self.extra_layer = nn.Linear(10, 5)  # Additional processing
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            x = self.extra_layer(x)  # Can add custom logic here
            return x
    
    custom_model = CustomNN()
    print(f"   Custom Model:\n{custom_model}")
    
    output_custom = custom_model(dummy_input)
    print(f"   Output shape: {output_custom.shape}")
    
    # -------------------------
    # METHOD 3: DYNAMIC ARCHITECTURE
    # -------------------------
    print("\n\n3. METHOD 3: DYNAMIC ARCHITECTURE")
    
    class DynamicNN(nn.Module):
        def __init__(self, layer_sizes, activations=None):
            """
            layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
            activations: List of activation functions
            """
            super().__init__()
            
            if activations is None:
                activations = ['relu'] * (len(layer_sizes) - 1)
            
            # Create layers dynamically
            self.layers = nn.ModuleList()
            for i in range(len(layer_sizes) - 1):
                self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            self.activations = activations
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                
                # Apply activation (except for last layer)
                if i < len(self.layers) - 1:
                    if self.activations[i] == 'relu':
                        x = F.relu(x)
                    elif self.activations[i] == 'sigmoid':
                        x = torch.sigmoid(x)
                    elif self.activations[i] == 'tanh':
                        x = torch.tanh(x)
                    
                    # Apply dropout to hidden layers
                    x = self.dropout(x)
            
            return x
    
    # Create dynamic model
    dynamic_model = DynamicNN(
        layer_sizes=[784, 256, 128, 64, 10],
        activations=['relu', 'relu', 'relu', None]  # Last layer no activation
    )
    
    print(f"   Dynamic Model:\n{dynamic_model}")
    
    output_dynamic = dynamic_model(dummy_input)
    print(f"   Output shape: {output_dynamic.shape}")
    
    # -------------------------
    # METHOD 4: RESIDUAL NETWORK (Skip Connections)
    # -------------------------
    print("\n\n4. METHOD 4: RESIDUAL BLOCKS (Skip Connections)")
    
    class ResidualBlock(nn.Module):
        """Residual block with skip connection"""
        def __init__(self, in_features, out_features):
            super().__init__()
            
            # Main path
            self.linear1 = nn.Linear(in_features, out_features)
            self.bn1 = nn.BatchNorm1d(out_features)
            self.linear2 = nn.Linear(out_features, out_features)
            self.bn2 = nn.BatchNorm1d(out_features)
            
            # Skip connection
            self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
            
            self.dropout = nn.Dropout(0.3)
        
        def forward(self, x):
            identity = self.skip(x)
            
            # Main path
            out = self.linear1(x)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.dropout(out)
            
            out = self.linear2(out)
            out = self.bn2(out)
            
            # Add skip connection
            out += identity
            out = F.relu(out)
            
            return out
    
    class ResNet(nn.Module):
        """Simple ResNet for tabular data"""
        def __init__(self):
            super().__init__()
            self.input_layer = nn.Linear(784, 256)
            self.res_block1 = ResidualBlock(256, 256)
            self.res_block2 = ResidualBlock(256, 256)
            self.output_layer = nn.Linear(256, 10)
        
        def forward(self, x):
            x = F.relu(self.input_layer(x))
            x = self.res_block1(x)
            x = self.res_block2(x)
            x = self.output_layer(x)
            return x
    
    resnet_model = ResNet()
    print(f"   ResNet Model:\n{resnet_model}")
    
    output_resnet = resnet_model(dummy_input)
    print(f"   Output shape: {output_resnet.shape}")
    
    return sequential_model, custom_model, dynamic_model, resnet_model

# Create different network architectures
models = different_network_architectures()