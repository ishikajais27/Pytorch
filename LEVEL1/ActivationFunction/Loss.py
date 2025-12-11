import matplotlib.pyplot as plt
import numpy as np


def visualize_loss_functions():
    """Show how different loss functions work"""
    
    # For MSE: Show squared error
    x = np.linspace(-2, 2, 100)
    y_mse = x**2  # Squared error
    
    # For cross-entropy: Show loss for binary classification
    predictions = np.linspace(0.01, 0.99, 100)
    loss_target1 = -np.log(predictions)  # When target=1
    loss_target0 = -np.log(1 - predictions)  # When target=0
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot MSE
    ax1.plot(x, y_mse, 'b-', linewidth=2)
    ax1.set_title('Mean Squared Error (MSE)', fontsize=14)
    ax1.set_xlabel('Prediction Error (prediction - target)')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add annotations
    ax1.annotate('Small error → Tiny loss', xy=(0.5, 0.25), 
                 xytext=(1, 1.5), arrowprops=dict(arrowstyle='->'))
    ax1.annotate('Big error → Huge loss', xy=(1.5, 2.25), 
                 xytext=(1, 3), arrowprops=dict(arrowstyle='->'))
    
    # Plot Binary Cross-Entropy
    ax2.plot(predictions, loss_target1, 'r-', linewidth=2, label='Target=1 (Yes)')
    ax2.plot(predictions, loss_target0, 'b-', linewidth=2, label='Target=0 (No)')
    ax2.set_title('Binary Cross-Entropy Loss', fontsize=14)
    ax2.set_xlabel('Prediction Probability')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axvline(x=0.5, color='k', linestyle='--', alpha=0.3)
    
    # Add annotations
    ax2.annotate('Confident and correct → Small loss', 
                 xy=(0.9, 0.1), xytext=(0.6, 1.5), 
                 arrowprops=dict(arrowstyle='->'))
    ax2.annotate('Confident but wrong → Huge loss', 
                 xy=(0.1, 2.3), xytext=(0.3, 3), 
                 arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.show()
    
    # Print practical examples
    print("\n" + "="*60)
    print("WHEN TO USE WHICH LOSS FUNCTION")
    print("="*60)
    
    examples = [
        {
            "problem": "House Price Prediction",
            "type": "Regression (predicting number)",
            "loss": "Mean Squared Error (MSE)",
            "reason": "We care about actual price differences"
        },
        {
            "problem": "Spam Email Detection", 
            "type": "Binary Classification (yes/no)",
            "loss": "Binary Cross-Entropy",
            "reason": "We want probability estimates"
        },
        {
            "problem": "Digit Recognition (0-9)",
            "type": "Multi-class Classification",
            "loss": "Categorical Cross-Entropy", 
            "reason": "Multiple exclusive classes"
        },
        {
            "problem": "Stock Price Trend",
            "type": "Regression with outliers",
            "loss": "Mean Absolute Error (MAE)",
            "reason": "Less sensitive to extreme errors"
        }
    ]
    
    for ex in examples:
        print(f"\n📊 {ex['problem']}:")
        print(f"   Type: {ex['type']}")
        print(f"   Loss Function: {ex['loss']}")
        print(f"   Why: {ex['reason']}")

# Run it
visualize_loss_functions()