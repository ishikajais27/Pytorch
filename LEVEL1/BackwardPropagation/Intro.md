Method used to update parameters(weigths and biases) using gradient.
Train model using backwardPropagation.

Think of it like a feedback system:-

Forward pass → get predictions.
Compare prediction to target → compute loss.
Backward pass → calculate gradient
Update weights → network learns.

Loss depends on output
Output depends on hidden layer  
Hidden depends on input

We can trace back: Loss → Output → Hidden → Input

Forward pass:
Inputs go through Layer 1 → activation → Layer 2 → output.
Output is compared to the target → loss calculated.

Backward pass (Backpropagation):
Compute gradient of loss w.r.t Layer 2 weights → tells Layer 2 how to adjust.
Use chain rule to propagate error back to Layer 1 → tells Layer 1 how to adjust.
Repeat until all weights and biases are updated.