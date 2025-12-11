They make neural networks able to learn complex things
They introduce non-linearity (bends, curves, decisions)
They turn neurons into decision-makers
Without them, networks are useless (just linear math)

Types of Activation Functions

1. ReLU (Rectified Linear Unit) - The Simplest,in modern deep networks,Hidden Layer,“Only act if the signal is positive”
   f(x)=max(0,x)
   Output: 0 → ∞

2. Sigmoid - The Probability Calculator,Convert any number to probability (0 to 1),0 = Not confident at all,1 = 100% confident
   σ(x)=1/(1+e^-x)

3. Tanh - The Balanced Calculator, Convert any number to -1 to 1,tanh(0) = 0(neutral),tanh(1) ≈ 0.76(positive),tanh(-1) ≈ -0.76 (negative),Stronger gradients than sigmoid for small inputs
4. Softmax- Converts raw scores to probabilities across multiple classes,Probabilistic interpretation → easier to understand predictions

---

#LOSS FUNCTION -
->Number that says "how wrong" the prediction was
->tells us how good or bad our model is performing.
\*Types of loss functions:-

1. Common Loss Function: Mean Squared Error (MSE)->
   MSE = 1/n[∑(yi-y'i)^2] , n= number of examples,yi - actual values, y'i - predicted value
   Average of squared differences
   Good for regression (predicting numbers)
   For Regression(predicting numbers)
2. Binary Cross-Entropy
   For classification(predicting categories)
   ![alt text]({8D8F2E9F-2243-4D06-8DFE-E8AE305A52F0}.png)
