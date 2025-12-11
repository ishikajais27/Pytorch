decision maker based in i/p,weight,biases and activation functions.
Inputs: [Sunny=1, Weekend=1, Temperature=30]
Decision: "Should I go to beach?"

The neuron:

1. Looks at inputs
2. Weighs importance of each:
   - Sunny: Very important (weight=2)
   - Weekend: Important (weight=1.5)
   - Temperature: Somewhat important (weight=0.1)
3. Adds them up: (1×2) + (1×1.5) + (30×0.1) = 6.5
4. Makes decision: If >5 → "Go to beach!"

Output = Activation( (input₁ × weight₁) + (input₂ × weight₂) + ... + bias )
