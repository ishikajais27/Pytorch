import matplotlib.pyplot as plt
import numpy as np


def visualize_activations():
    """Show all activation functions"""
    
    # Generate x values from -5 to 5
    x = np.linspace(-5, 5, 100)
    
    # Define activation functions
    def relu(x):
        return np.maximum(0, x)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(x):
        return np.tanh(x)
    
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    # Calculate
    y_relu = relu(x)
    y_sigmoid = sigmoid(x)
    y_tanh = tanh(x)
    y_leaky = leaky_relu(x)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot each
    axes[0,0].plot(x, y_relu, 'b-', linewidth=2)
    axes[0,0].set_title('ReLU Activation', fontsize=14)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlabel('Input')
    axes[0,0].set_ylabel('Output')
    axes[0,0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0,0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[0,1].plot(x, y_sigmoid, 'r-', linewidth=2)
    axes[0,1].set_title('Sigmoid Activation', fontsize=14)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xlabel('Input')
    axes[0,1].set_ylabel('Output')
    axes[0,1].axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    axes[0,1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[1,0].plot(x, y_tanh, 'g-', linewidth=2)
    axes[1,0].set_title('Tanh Activation', fontsize=14)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlabel('Input')
    axes[1,0].set_ylabel('Output')
    axes[1,0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1,0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[1,1].plot(x, y_leaky, 'm-', linewidth=2)
    axes[1,1].set_title('Leaky ReLU (α=0.01)', fontsize=14)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlabel('Input')
    axes[1,1].set_ylabel('Output')
    axes[1,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1,1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print explanations
    print("\n" + "="*60)
    print("ACTIVATION FUNCTIONS EXPLAINED")
    print("="*60)
    
    print("\n1. ReLU (Rectified Linear Unit):")
    print("   - Output = max(0, input)")
    print("   - Fastest to compute")
    print("   - Most common in hidden layers")
    print("   - Problem: 'Dying ReLU' (outputs 0 forever if input negative)")
    
    print("\n2. Sigmoid:")
    print("   - Output = 1 / (1 + e^(-input))")
    print("   - Output range: 0 to 1")
    print("   - Good for probability/classification")
    print("   - Problem: Vanishing gradient (slow learning)")
    
    print("\n3. Tanh (Hyperbolic Tangent):")
    print("   - Output range: -1 to 1")
    print("   - Better than sigmoid for most cases")
    print("   - Output centered at 0")
    
    print("\n4. Leaky ReLU:")
    print("   - Output = input if input > 0, else α×input")
    print("   - Fixes 'dying ReLU' problem")
    print("   - Small negative slope instead of 0")

# Run visualization
visualize_activations()